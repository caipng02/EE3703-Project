"""The workbench tool dataset: 640 x 640 letterboxed images with YOLO-format labels and a manifest.

Layout: dataset/manifest.csv (one row per image: file_name, split, origin, source, class counts) and
dataset/<split>/images, <split>/labels, <split>/annotations.coco.json for train, valid and test.
"""

import csv
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
from .augment import AUGMENTATIONS, mosaic

# %% dataset
cv2.setNumThreads(0)  # OpenCV's own threads can hang inside DataLoader worker processes
TRAINING_SPLITS = ("train", "trainval")  # "trainval" = train + valid, only for the final retrain


def read_manifest(root):
    """manifest.csv as a list of dicts: one row per image with its split, origin, source and box counts."""
    with (Path(root) / "manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize(image):
    """RGB uint8 HWC array -> float tensor CHW with ImageNet normalization (what the model takes)."""
    tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255
    return (tensor - torch.tensor(IMAGENET_MEAN).view(3, 1, 1)) / torch.tensor(IMAGENET_STD).view(3, 1, 1)


def denormalize(tensor):
    """Undo normalize, for display."""
    image = tensor * torch.tensor(IMAGENET_STD).view(3, 1, 1) + torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    return (image.permute(1, 2, 0).clamp(0, 1) * 255).round().byte().numpy()


class ToolDataset(Dataset):
    """One split of the dataset. Items are (normalized image, target dict with pixel xyxy boxes and labels).

    The test split is locked unless allow_test=True, so it cannot leak into development by accident.
    """

    def __init__(self, root, split, augment=False, mosaic_p=0.0, origin=None, allow_test=False,
                 *, augmentation="full"):
        if split == "test" and not allow_test:
            raise ValueError("The test split is locked; pass allow_test=True only for the final evaluation.")
        if split not in TRAINING_SPLITS and (augment or mosaic_p):
            raise ValueError("Augmentation is for training data only.")
        if augmentation not in ("full", "light"):
            raise ValueError("augmentation must be 'full' or 'light'.")
        self.augmentation = augmentation  # looked up when an image is drawn (see data/augment.py)
        self.root, self.split, self.augment, self.mosaic_p = Path(root), split, augment, mosaic_p
        folders = ("train", "valid") if split == "trainval" else (split,)
        rows = [r for r in read_manifest(self.root)
                if r["split"] in folders and (origin is None or r["origin"] == origin)]
        self.names = [r["file_name"] for r in rows]
        self.origins = [r["origin"] for r in rows]
        self.folders = [r["split"] for r in rows]
        self.boxes, self.labels = [], []
        for name, folder in zip(self.names, self.folders):
            label_path = self.root / folder / "labels" / f"{Path(name).stem}.txt"
            lines = [line.split() for line in label_path.read_text(encoding="utf-8-sig").splitlines() if line]
            values = np.array([[float(v) * IMAGE_SIZE for v in line[1:]] for line in lines], np.float32).reshape(-1, 4)
            # Clip because labels are stored to 6 decimals, which can land a tiny fraction outside the image.
            self.boxes.append(np.stack([values[:, 0] - values[:, 2] / 2, values[:, 1] - values[:, 3] / 2,
                                        values[:, 0] + values[:, 2] / 2, values[:, 1] + values[:, 3] / 2],
                                       axis=1).clip(0, IMAGE_SIZE))
            self.labels.append(np.array([int(line[0]) for line in lines], np.int64))
        self.sample_weights = None  # set by the training sampler; mosaic partners follow the same weights

    def __len__(self):
        return len(self.names)

    def raw(self, index):
        """Image (RGB uint8), pixel xyxy boxes and labels, before augmentation and normalization."""
        image = cv2.cvtColor(cv2.imread(str(self.root / self.folders[index] / "images" / self.names[index])),
                             cv2.COLOR_BGR2RGB)
        return image, self.boxes[index].copy(), self.labels[index].copy()

    def __getitem__(self, index):
        image, boxes, labels = self.raw(index)
        if self.augment:
            # Seeded from torch so each worker and epoch draws a different, reproducible augmentation.
            rng = random.Random(int(torch.randint(0, 2**31 - 1, (1,)).item()))
            transform = AUGMENTATIONS[self.augmentation]
            if rng.random() < self.mosaic_p:
                others = rng.choices(range(len(self)), weights=self.sample_weights, k=3)
                image, boxes, labels = mosaic([transform(*self.raw(i), rng) for i in [index, *others]], rng)
            else:
                image, boxes, labels = transform(image, boxes, labels, rng)
        return normalize(image), {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "file_name": self.names[index],
            "origin": self.origins[index],
        }


def collate(batch):
    """Images stack into one tensor; targets stay a list because box counts differ between images."""
    return torch.stack([item[0] for item in batch]), [item[1] for item in batch]
