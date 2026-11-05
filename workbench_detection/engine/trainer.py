"""Training: data loading, the optimization loop, learning-rate schedule, weight averaging and checkpoints."""

import copy
import csv
import math
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..config import AP_SCORE_FLOOR, CLASSES, MODEL_NAME, SEED, TrainConfig, seed_everything
from ..data.dataset import ToolDataset, collate
from ..data.sampling import make_train_sampler
from ..evaluation.metrics import EvaluationConfig, coco_ap
from ..evaluation.records import load_ground_truth, load_json
from ..evaluation.report import plot_training_history
from ..models.loss import yolo_loss_on_cpu
from ..models.yolov2 import YOLOv2ResNet18, anchors_from_dataset, init_objectness_prior, load_imagenet_backbone
from .predictor import YOLOv2Detector

# %% loader
EMA_DECAY, EMA_RAMP = 0.99, 100


def make_loader(dataset, batch, training=False, *, generator=None, original_weight=1.0):
    """Training loaders sample with class-aware weights; evaluation loaders read every image once, in order.

    Worker processes decode and augment images in parallel so the GPU is not left waiting.
    """
    sampler = make_train_sampler(dataset, original_weight, generator=generator) if training else None
    workers = min(4, os.cpu_count() or 1)
    return DataLoader(dataset, batch_size=batch, sampler=sampler, collate_fn=collate, num_workers=workers,
                      persistent_workers=workers > 0, pin_memory=torch.cuda.is_available())


# %% schedule
def lr_factor(step, total_steps, warmup_steps, schedule):
    """Linear warmup from 1% of the learning rate, then constant or cosine decay to 5%."""
    if step < warmup_steps:
        return 0.01 + 0.99 * step / warmup_steps
    if schedule == "cosine":
        progress = min((step - warmup_steps) / max(1, total_steps - warmup_steps), 1.0)
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress))
    return 1.0


class ModelEMA:
    """Exponential moving average of the weights, used for validation and checkpoints.

    Smooths the epoch-to-epoch swings in validation AP. The decay ramps up from 0 so early averages are
    not dominated by the random initialization.
    """

    def __init__(self, model, decay=EMA_DECAY, ramp=EMA_RAMP):
        self.module = copy.deepcopy(model).eval()
        self.decay, self.ramp, self.updates = decay, ramp, 0
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        decay = self.decay * (1 - math.exp(-self.updates / self.ramp))
        current = model.state_dict()
        for name, value in self.module.state_dict().items():
            if value.dtype.is_floating_point:
                value.mul_(decay).add_(current[name].detach(), alpha=1 - decay)
            else:
                value.copy_(current[name])


# %% optimization
def make_scaler(device):
    """Mixed-precision gradient scaler, active on CUDA only (torch.amp.GradScaler needs PyTorch 2.3+)."""
    enabled = device.type == "cuda"
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def training_step(model, optimizer, images, targets, device, scaler, objectness_target="iou", anchors_per_object=1):
    """One update: forward in mixed precision on CUDA, loss, backward, gradient clipping at norm 10, step."""
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        raw = model(images.to(device))
    loss, _ = yolo_loss_on_cpu(raw, targets, model.anchors, objectness_target, anchors_per_object)
    if not torch.isfinite(loss):
        raise ValueError("Training produced a nonfinite loss.")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
    scaler.step(optimizer)
    scaler.update()
    return loss.item()


# %% train
def build_model(anchors, config):
    """A YOLOv2 model initialized as the configuration says."""
    model = YOLOv2ResNet18(anchors, stride=config.stride)
    if config.pretrained_backbone:
        print(f"Loaded {load_imagenet_backbone(model)} ImageNet tensors into the ResNet-18 backbone.", flush=True)
    if config.objectness_prior:
        init_objectness_prior(model)
    return model


def train(config, dataset_root, work_dir, report_dir, device, *, final_from=None, protocol_path=None):
    """Train YOLOv2 and save the checkpoint with the best validation AP50 to work_dir/best.pt.

    Development (default): train on the training split, validate every epoch, stop early after
    config.patience epochs without improvement. Final (final_from + protocol_path): retrain from scratch
    on train + validation with the development run's settings for exactly its best epoch count; there is
    no validation data left, so the last epoch is saved.
    """
    seed_everything()
    work_dir, dataset_root = Path(work_dir), Path(dataset_root)
    work_dir.mkdir(parents=True, exist_ok=True)
    final = final_from is not None
    protocol = None
    if final:
        source = torch.load(final_from, map_location="cpu", weights_only=True)
        protocol = load_json(protocol_path)
        selected = Path(protocol["models"][MODEL_NAME]["validation_checkpoint"]).resolve()
        if selected != Path(final_from).resolve():
            raise ValueError("Use the validation checkpoint recorded in the selected protocol.")
        if tuple(source["classes"]) != CLASSES or source.get("stage") != "validation":
            raise ValueError("Final retraining requires a compatible validation checkpoint.")
        config = TrainConfig.from_dict(source["config"])
        anchors, epochs = source["anchors"], source["best_epoch"]
    else:
        epochs = config.epochs
    train_set = ToolDataset(dataset_root, "trainval" if final else "train", augment=True,
                            mosaic_p=config.mosaic, augmentation=config.augmentation)
    if not final:
        anchors = anchors_from_dataset(train_set, config.anchors)
    loader = make_loader(train_set, config.batch, training=True, generator=torch.Generator().manual_seed(SEED),
                         original_weight=config.original_weight)
    print(f"{'Final' if final else 'Development'} training on {len(train_set)} images | {epochs} epochs max | "
          f"{len(loader)} updates/epoch", flush=True)
    print("Settings: " + ", ".join(f"{key}={value}" for key, value in config.to_dict().items()), flush=True)
    if not final:
        ap_config = EvaluationConfig(ap_backend="coco", ap_score_floor=AP_SCORE_FLOOR)
        ground_truth = load_ground_truth(dataset_root, "valid")
        valid = make_loader(ToolDataset(dataset_root, "valid"), config.batch)
    model = build_model(anchors, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    total_steps, warmup_steps = epochs * len(loader), int(config.warmup_epochs * len(loader))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_factor(step, total_steps, warmup_steps, config.lr_schedule))
    ema = ModelEMA(model) if config.ema else None
    scaler = make_scaler(device)
    checkpoint = work_dir / "best.pt"
    best, best_epoch, best_val_loss, history = -1, 0, None, []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for images, targets in loader:
            total += training_step(model, optimizer, images, targets, device, scaler,
                                   config.objectness_target, config.anchors_per_object) * len(targets)
            scheduler.step()
            if ema:
                ema.update(model)
        row = {"epoch": epoch, "train_loss": total / len(loader.dataset), "lr": optimizer.param_groups[0]["lr"]}
        scored = ema.module if ema else model  # validation and checkpoints use the averaged weights
        if not final:
            exports, validation_loss = YOLOv2Detector(scored, device).export(
                valid, ground_truth, [0.50], ap_config.ap_score_floor, checkpoint=checkpoint, with_loss=True,
                objectness_target=config.objectness_target, anchors_per_object=config.anchors_per_object)
            aps, _ = coco_ap(ground_truth, exports[0], ap_config)
            score = sum(ap["0.50"] for ap in aps) / len(aps)
            row.update(val_loss=validation_loss, val_map50=score)
        history.append(row)
        print(f"epoch {epoch:>3}/{epochs} | train_loss {row['train_loss']:.4f}"
              + (f" | val_loss {row['val_loss']:.4f} | val_AP50 {row['val_map50']:.4f}" if not final else ""), flush=True)
        if final or score > best:
            if not final:
                best, best_val_loss = score, validation_loss
            best_epoch = epoch
            torch.save({"state_dict": scored.state_dict(), "anchors": anchors, "classes": CLASSES,
                        "config": config.to_dict(), "best_epoch": epoch,
                        "stage": "final" if final else "validation", "protocol": protocol}, checkpoint)
        if not final and epoch - best_epoch >= config.patience:
            print(f"Early stopping: validation AP50 did not improve for {config.patience} epochs.", flush=True)
            break
    with (work_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    plot_training_history(history, report_dir or work_dir, MODEL_NAME)
    print(f"Checkpoint: {checkpoint.resolve()} (epoch {best_epoch})", flush=True)
    return {"checkpoint": checkpoint, "best_epoch": best_epoch, "epochs_run": len(history),
            "best_map50": None if final else best, "val_loss_at_best": best_val_loss, "history": history}


def read_history(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [{key: float(value) for key, value in row.items() if value != ""} for row in csv.DictReader(handle)]
