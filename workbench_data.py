"""Data loader and augmentation from Person 1’s EE3703_data notebook."""

import random

import cv2
import numpy as np

SIZE = 640
SCISSOR = 1
MIN_VISIBLE = 0.3  # a box must stay at least 30% visible, so at most 70% is hidden
MIN_SIDE = 4.0
GEOMETRIC_TRIES = 5
MOSAIC_TRIES = 10
CONFIG = {
    "horizontal_flip_p": 0.5,
    "rotate_deg": 15.0,
    "translate_fraction": 0.08,
    "scale_range": [0.8, 1.2],
    "brightness_contrast": 0.2,
    "gamma_range": [0.8, 1.25],
    "hue_shift_deg": 8,
    "saturation_range": [0.8, 1.2],
    "blur_p": 0.15,
    "noise_p": 0.15,
    "noise_sigma": 6.0,
    "cutout_p": 0.3,
    "cutout_fraction": 0.2,
    "scissor_scale_range": [0.7, 1.3],
    "scissor_occlusion_p": 0.5,
    "scissor_occlusion_area": [0.1, 0.3],
    "mosaic_p": 0.0,
    "minimum_visible_box_fraction": MIN_VISIBLE,
}


def _visible(boxes: np.ndarray, reference_area: np.ndarray, bounds: tuple[float, float, float, float] = (0, 0, SIZE, SIZE)) -> tuple[np.ndarray, np.ndarray]:
    """Return boxes clipped to bounds and a mask of those that stay usable."""
    clipped = boxes.copy()
    clipped[:, [0, 2]] = clipped[:, [0, 2]].clip(bounds[0], bounds[2])
    clipped[:, [1, 3]] = clipped[:, [1, 3]].clip(bounds[1], bounds[3])
    width, height = clipped[:, 2] - clipped[:, 0], clipped[:, 3] - clipped[:, 1]
    return clipped, (width >= MIN_SIDE) & (height >= MIN_SIDE) & (width * height >= MIN_VISIBLE * reference_area)


def _warp_boxes(boxes: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    corners = np.ones((len(boxes), 4, 3))
    corners[:, 0, :2] = boxes[:, [0, 1]]
    corners[:, 1, :2] = boxes[:, [2, 1]]
    corners[:, 2, :2] = boxes[:, [2, 3]]
    corners[:, 3, :2] = boxes[:, [0, 3]]
    moved = corners @ matrix.T
    return np.concatenate([moved.min(axis=1), moved.max(axis=1)], axis=1)


# Flip, then try up to 5 random rotation, scale and shift combinations that keep every box usable.
# If none works, only the flip is kept.
def geometric(image: np.ndarray, boxes: np.ndarray, labels: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if rng.random() < CONFIG["horizontal_flip_p"]:
        image = image[:, ::-1].copy()
        if len(boxes):
            boxes = boxes.copy()
            boxes[:, [0, 2]] = SIZE - boxes[:, [2, 0]]
    scale_range = CONFIG["scissor_scale_range"] if (labels == SCISSOR).any() else CONFIG["scale_range"]
    shift = CONFIG["translate_fraction"] * SIZE
    for _ in range(GEOMETRIC_TRIES):
        matrix = cv2.getRotationMatrix2D((SIZE / 2, SIZE / 2), rng.uniform(-CONFIG["rotate_deg"], CONFIG["rotate_deg"]), rng.uniform(*scale_range))
        matrix[:, 2] += (rng.uniform(-shift, shift), rng.uniform(-shift, shift))
        if not len(boxes):
            return cv2.warpAffine(image, matrix, (SIZE, SIZE), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114)), boxes, labels
        moved = _warp_boxes(boxes, matrix)
        clipped, usable = _visible(moved, (moved[:, 2] - moved[:, 0]) * (moved[:, 3] - moved[:, 1]))
        if usable.all():
            return cv2.warpAffine(image, matrix, (SIZE, SIZE), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114)), clipped.astype(np.float32), labels
    return image, boxes, labels


# Colour changes and noise only, so boxes don't move.
def photometric(image: np.ndarray, rng: random.Random) -> np.ndarray:
    limit = CONFIG["brightness_contrast"]
    output = image.astype(np.float32) * rng.uniform(1 - limit, 1 + limit) + 255 * rng.uniform(-limit, limit) / 2
    output = 255 * (output.clip(0, 255) / 255) ** rng.uniform(*CONFIG["gamma_range"])
    hsv = cv2.cvtColor(output.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-CONFIG["hue_shift_deg"], CONFIG["hue_shift_deg"]) / 2) % 180
    hsv[..., 1] = (hsv[..., 1] * rng.uniform(*CONFIG["saturation_range"])).clip(0, 255)
    output = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    if rng.random() < CONFIG["blur_p"]:
        output = cv2.GaussianBlur(output, (5, 5), 0)
    if rng.random() < CONFIG["noise_p"]:
        noise = np.random.default_rng(rng.randrange(2**32)).normal(0, CONFIG["noise_sigma"], output.shape)
        output = (output + noise).clip(0, 255).astype(np.uint8)
    return output


def _try_patch(image: np.ndarray, hidden: np.ndarray, boxes: np.ndarray, x: int, y: int, w: int, h: int) -> bool:
    """Grey out a rectangle unless the total hidden area of any box would pass the limit."""
    trial = hidden.copy()
    trial[y:y + h, x:x + w] = True
    for x1, y1, x2, y2 in boxes.astype(int):
        area = trial[y1:y2, x1:x2]
        if area.size and area.mean() > 1 - MIN_VISIBLE:
            return False
    hidden[y:y + h, x:x + w] = True
    image[y:y + h, x:x + w] = 114
    return True


def scissor_occlusion(image: np.ndarray, hidden: np.ndarray, boxes: np.ndarray, labels: np.ndarray, rng: random.Random) -> None:
    """Partly cover one scissor box, since thin blades and handles are often occluded."""
    scissors = boxes[labels == SCISSOR]
    if not len(scissors) or rng.random() >= CONFIG["scissor_occlusion_p"]:
        return
    x1, y1, x2, y2 = scissors[rng.randrange(len(scissors))]
    share = rng.uniform(*CONFIG["scissor_occlusion_area"])
    w, h = max(int((x2 - x1) * share ** 0.5), 1), max(int((y2 - y1) * share ** 0.5), 1)
    x = int(rng.uniform(x1, max(x2 - w, x1)))
    y = int(rng.uniform(y1, max(y2 - h, y1)))
    _try_patch(image, hidden, boxes, x, y, min(w, SIZE - x), min(h, SIZE - y))


# A gray square at a random place, skipped if it would hide too much of a box.
def cutout(image: np.ndarray, hidden: np.ndarray, boxes: np.ndarray, rng: random.Random) -> None:
    if rng.random() >= CONFIG["cutout_p"]:
        return
    side = int(SIZE * CONFIG["cutout_fraction"] * rng.uniform(0.5, 1))
    _try_patch(image, hidden, boxes, rng.randrange(SIZE - side), rng.randrange(SIZE - side), side, side)


# The full training augmentation for one image.
def augment(image: np.ndarray, boxes: np.ndarray, labels: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image, boxes, labels = geometric(image, boxes, labels, rng)
    image = photometric(image, rng).copy()
    hidden = np.zeros((SIZE, SIZE), dtype=bool)
    scissor_occlusion(image, hidden, boxes, labels, rng)
    cutout(image, hidden, boxes, rng)
    return image, boxes, labels


def mosaic(samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]], rng: random.Random) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Join four training images; call with samples from the training split only."""
    cx, cy = rng.randint(SIZE // 4, 3 * SIZE // 4), rng.randint(SIZE // 4, 3 * SIZE // 4)
    canvas = np.full((SIZE, SIZE, 3), 114, dtype=np.uint8)
    regions = [(0, 0, cx, cy), (cx, 0, SIZE, cy), (0, cy, cx, SIZE), (cx, cy, SIZE, SIZE)]
    all_boxes, all_labels = [np.zeros((0, 4), np.float32)], [np.zeros(0, np.int64)]
    for (x1, y1, x2, y2), (image, boxes, labels) in zip(regions, samples):
        width, height = x2 - x1, y2 - y1
        best = None
        # Choose the crop that leaves the fewest cut-off remnants without a label.
        for _ in range(MOSAIC_TRIES):
            offset_x, offset_y = rng.randint(0, SIZE - width), rng.randint(0, SIZE - height)
            moved = boxes.copy()
            moved[:, [0, 2]] += x1 - offset_x
            moved[:, [1, 3]] += y1 - offset_y
            clipped, usable = _visible(moved, (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), (x1, y1, x2, y2))
            touching = (moved[:, 2] > x1) & (moved[:, 0] < x2) & (moved[:, 3] > y1) & (moved[:, 1] < y2)
            remnants = int((touching & ~usable).sum())
            score = (remnants, -int(usable.sum()))
            if best is None or score < best[0]:
                best = (score, offset_x, offset_y, clipped[usable], labels[usable])
        _, offset_x, offset_y, kept, kept_labels = best
        canvas[y1:y2, x1:x2] = image[offset_y:offset_y + height, offset_x:offset_x + width]
        all_boxes.append(kept.astype(np.float32))
        all_labels.append(kept_labels)
    return canvas, np.concatenate(all_boxes), np.concatenate(all_labels)


import csv
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

# OpenCV threads can hang inside DataLoader worker processes.
cv2.setNumThreads(0)

SIZE = 640
NUM_CLASSES = 3
TRAINING_SPLITS = ("train", "trainval")
# ImageNet statistics, because both detectors use a standard ResNet-18 backbone.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
MAX_CLASS_WEIGHT = 2.0
# Starting values; recheck them on your detector, using validation only.
ORIGINAL_WEIGHT = 1.0
MOSAIC_P = 0.0


def normalize(image: np.ndarray) -> torch.Tensor:
    """Convert an RGB uint8 HWC array to the tensor both models take."""
    tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255
    return (tensor - torch.tensor(MEAN).view(3, 1, 1)) / torch.tensor(STD).view(3, 1, 1)


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Undo normalize for visual checks."""
    image = tensor * torch.tensor(STD).view(3, 1, 1) + torch.tensor(MEAN).view(3, 1, 1)
    return (image.permute(1, 2, 0).clamp(0, 1) * 255).round().byte().numpy()


def boxes_to_yolo(boxes: torch.Tensor) -> torch.Tensor:
    """Convert pixel xyxy boxes to normalized cx, cy, w, h."""
    return torch.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2, boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]], dim=1) / SIZE


class ToolDataset(Dataset):
    """Images and targets for one split; see the table above for the target format."""

    def __init__(self, root: str | Path, split: str, augment: bool = False, mosaic_p: float = 0.0, origin: str | None = None, allow_test: bool = False, *, augmentation_fn=augment) -> None:
        if split == "test" and not allow_test:
            raise ValueError("The test split is locked; pass allow_test=True only for the final evaluation.")
        if split not in TRAINING_SPLITS and (augment or mosaic_p):
            raise ValueError("Augmentation is for training data only.")
        self.augmentation_fn = augmentation_fn
        self.root, self.split, self.augment, self.mosaic_p = Path(root), split, augment, mosaic_p
        # "trainval" is only for the final retrain, after validation has fixed every setting.
        folders = ("train", "valid") if split == "trainval" else (split,)
        with (self.root / "manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
            rows = [r for r in csv.DictReader(handle) if r["split"] in folders and (origin is None or r["origin"] == origin)]
        self.names = [r["file_name"] for r in rows]
        self.origins = [r["origin"] for r in rows]
        self.folders = [r["split"] for r in rows]
        self.boxes, self.labels = [], []
        for name, folder in zip(self.names, self.folders):
            lines = [l.split() for l in (self.root / folder / "labels" / f"{Path(name).stem}.txt").read_text(encoding="utf-8-sig").splitlines() if l]
            values = np.array([[float(v) * SIZE for v in l[1:]] for l in lines], np.float32).reshape(-1, 4)
            # Clip because labels are stored to 6 decimals, which can land a tiny fraction outside the image.
            self.boxes.append(np.stack([values[:, 0] - values[:, 2] / 2, values[:, 1] - values[:, 3] / 2, values[:, 0] + values[:, 2] / 2, values[:, 1] + values[:, 3] / 2], axis=1).clip(0, SIZE))
            self.labels.append(np.array([int(l[0]) for l in lines], np.int64))
        self.sample_weights: list[float] | None = None

    def __len__(self) -> int:
        return len(self.names)

    def _raw(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        image = cv2.cvtColor(cv2.imread(str(self.root / self.folders[index] / "images" / self.names[index])), cv2.COLOR_BGR2RGB)
        return image, self.boxes[index].copy(), self.labels[index].copy()

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict]:
        image, boxes, labels = self._raw(index)
        if self.augment:
            # Seeded from torch so each worker and epoch draws a different, reproducible augmentation.
            rng = random.Random(int(torch.randint(0, 2**31 - 1, (1,)).item()))
            if rng.random() < self.mosaic_p:
                others = rng.choices(range(len(self)), weights=self.sample_weights, k=3)
                image, boxes, labels = mosaic([self.augmentation_fn(*self._raw(i), rng) for i in [index, *others]], rng)
            else:
                image, boxes, labels = self.augmentation_fn(image, boxes, labels, rng)
        return normalize(image), {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "file_name": self.names[index],
            "origin": self.origins[index],
        }


# Images stack into one tensor. Targets stay a list because box counts differ between images.
def collate(batch: list[tuple[torch.Tensor, dict]]) -> tuple[torch.Tensor, list[dict]]:
    return torch.stack([item[0] for item in batch]), [item[1] for item in batch]


def image_weights(dataset: ToolDataset, original_weight: float = ORIGINAL_WEIGHT) -> list[float]:
    """Per-image weights from training boxes only.

    Originals are upweighted first. Class weights are then the inverse of the resulting
    box frequency, capped at 2 times, so favouring originals does not unbalance the classes.
    """
    if dataset.split not in TRAINING_SPLITS:
        raise ValueError("Sampling weights are computed from training data only.")
    base = [original_weight if origin == "original" else 1.0 for origin in dataset.origins]
    counts = np.zeros(NUM_CLASSES)
    for weight, labels in zip(base, dataset.labels):
        counts += weight * np.bincount(labels, minlength=NUM_CLASSES)
    class_weight = np.minimum(counts.max() / counts, MAX_CLASS_WEIGHT)
    return [weight * (float(class_weight[labels].max()) if len(labels) else 1.0) for weight, labels in zip(base, dataset.labels)]


def expected_box_counts(dataset: ToolDataset, weights: list[float]) -> np.ndarray:
    """Boxes per class one epoch is expected to contain under the given weights."""
    scale = len(weights) / sum(weights)
    counts = np.zeros(NUM_CLASSES)
    for weight, labels in zip(weights, dataset.labels):
        counts += scale * weight * np.bincount(labels, minlength=NUM_CLASSES)
    return counts


def make_train_sampler(dataset: ToolDataset, original_weight: float = ORIGINAL_WEIGHT, generator: torch.Generator | None = None) -> WeightedRandomSampler:
    weights = image_weights(dataset, original_weight)
    dataset.sample_weights = weights
    return WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True, generator=generator)
