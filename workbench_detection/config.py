"""Project constants and the training configuration shared by every stage."""

import random
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import numpy as np
import torch

# %% constants
CLASSES = ("drill", "scissor", "screwdriver")  # class IDs 0, 1, 2; COCO category IDs are these plus 1
IMAGE_SIZE = 640                                # every image is letterboxed to 640 x 640
LETTERBOX_FILL = (114, 114, 114)
IMAGENET_MEAN = (0.485, 0.456, 0.406)           # the ResNet-18 backbone expects ImageNet statistics
IMAGENET_STD = (0.229, 0.224, 0.225)
SEED = 288275
MODEL_NAME = "YOLOv2"
AP_SCORE_FLOOR = 0.001                          # detections below this score are not exported for AP


def seed_everything(seed=SEED):
    """Seed Python, NumPy and PyTorch so runs repeat."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


# %% train-config
@dataclass(frozen=True)
class TrainConfig:
    """Every setting that affects training. Saved with each run and inside each checkpoint."""

    pretrained_backbone: bool = True   # ImageNet weights for the ResNet-18 backbone only
    augmentation: str = "full"         # "full": the proposal's pipeline; "light": flip, brightness, contrast
    mosaic: float = 0.5                # probability of a 4-image mosaic per training sample
    original_weight: float = 1.0       # sampling weight of original photos relative to public ones
    batch: int = 8
    lr: float = 5e-4                   # peak learning rate (AdamW)
    weight_decay: float = 1e-4
    epochs: int = 60                   # maximum epochs (proposal section 4.2)
    patience: int = 10                 # stop after this many epochs without a better validation AP50
    lr_schedule: str = "cosine"        # "cosine" decays to 5% of lr; "constant" keeps it fixed
    warmup_epochs: float = 5.0         # linear warmup from 1% of lr
    ema: bool = True                   # validate and save an exponential moving average of the weights
    objectness_prior: bool = True      # start objectness at 1% instead of 50%

    def __post_init__(self):
        checks = {
            "augmentation": self.augmentation in ("full", "light"),
            "mosaic": 0 <= self.mosaic <= 1,
            "original_weight": self.original_weight > 0,
            "batch": self.batch >= 1,
            "lr": self.lr > 0,
            "weight_decay": self.weight_decay >= 0,
            "epochs": self.epochs >= 1,
            "patience": self.patience >= 1,
            "lr_schedule": self.lr_schedule in ("cosine", "constant"),
            "warmup_epochs": self.warmup_epochs >= 0,
        }
        invalid = [name for name, ok in checks.items() if not ok]
        if invalid:
            raise ValueError(f"Invalid training settings: {', '.join(invalid)}")

    def to_dict(self):
        """The settings as a plain dict (saved in run folders and checkpoints)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        """Build from a dict saved with a run or checkpoint, ignoring unknown keys.

        Runs saved while the architecture was still a setting must have used this one (ARCHITECTURE); a setting
        missing from an old run had its original value then (ARCHITECTURE_BEFORE_SETTING).
        """
        values = {**ARCHITECTURE_BEFORE_SETTING, **dict(values)}
        different = [key for key, value in ARCHITECTURE.items() if values.get(key, value) != value]
        if "training_profile" in values or different:
            raise ValueError(f"Saved for a different YOLOv2 architecture ({', '.join(different) or 'old format'}).")
        known = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in known})

    def updated(self, **changes):
        """A copy with some settings changed, e.g. config.updated(lr=3e-4)."""
        return replace(self, **changes)


# The architecture in models/yolov2.py and models/loss.py, as the development code recorded it in run settings:
# 9 k-means anchors, the 40 x 40 grid (stride 16), the IoU objectness target and one anchor per object.
ARCHITECTURE = {"anchors": 9, "stride": 16, "objectness_target": "iou", "anchors_per_object": 1}
ARCHITECTURE_BEFORE_SETTING = {"anchors": 5, "stride": 32, "objectness_target": "iou", "anchors_per_object": 1}


def load_config(path):
    """Read a YAML training configuration; settings it leaves out take the current defaults."""
    import yaml

    with Path(path).open(encoding="utf-8") as stream:
        values = yaml.safe_load(stream) or {}
    known = {field.name for field in fields(TrainConfig)}
    return TrainConfig(**{key: value for key, value in values.items() if key in known})


def save_config(config, path):
    """Write a TrainConfig as YAML (the run folder's config.yaml)."""
    import yaml

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8")
