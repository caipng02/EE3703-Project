"""Faster R-CNN training, inference and the validation / final-test workflow.

The steps match the YOLOv2 engine: development training on the training split with validation AP50 every epoch;
then the RPN NMS IoU and operating confidence are chosen on validation and frozen in evaluation_protocol.json;
then the model is retrained on train + validation for the best epoch count and scored once on the 50 test originals.
The model and losses are in models/faster_rcnn.py; FasterRCNNConfig's defaults are the selected training recipe.
"""

import csv
import shutil
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch.utils.data import Subset

from ..config import AP_SCORE_FLOOR, CLASSES, SEED, seed_everything
from ..data.dataset import ToolDataset, normalize
from ..data.transforms import letterbox, restore_box
from ..evaluation.benchmark import benchmark_inference, model_size_summary
from ..evaluation.metrics import EvaluationConfig, coco_ap
from ..evaluation.protocol import NMS_GRIDS, run_evaluation
from ..evaluation.records import load_ground_truth, load_json, predictions_from_outputs, save_json
from ..evaluation.report import plot_training_history
from ..models.faster_rcnn import (ANCHOR_SIZES, ASPECT_RATIOS, FEATURE_STRIDE, FINAL_NMS_IOU, INFERENCE_PROPOSALS, ROI_FC_LAYERS, ROI_POOL,
                                  RPN_NMS_IOU, RPN_POSITIVE_IOU, TRAIN_PROPOSALS, FasterRCNN_Modular, _match_boxes_to_gt,
                                  compute_unified_loss_per_img, decode_detections, filter_proposals)
from .trainer import ModelEMA, lr_factor, make_loader
from .workflow import Run


# %% config
FRCNN_NAME = "Faster R-CNN"
# Anchors kept (highest RPN score) before RPN NMS at inference, chosen on validation with the RPN NMS IoU. The
# usual 10 x 600 proposals = 6,000 keeps only 1.7% of the 352,000 anchors, which can cut weakly scored screwdrivers.
PRE_NMS_GRID = (6000, 12000, 24000, 48000)
FRCNN_EMA_DECAY = 0.997  # about 2 epochs of 179 updates, the same averaging horizon as YOLOv2's 0.99 at 45 updates


@dataclass(frozen=True)
class FasterRCNNConfig:
    """Every setting that affects Faster R-CNN training. The defaults are the selected recipe."""

    pretrained_backbone: bool = True   # ImageNet weights for the ResNet-18 backbone
    rpn_positive_iou: float = RPN_POSITIVE_IOU  # anchors above this IoU with an object are RPN foreground
    train_proposals: int = TRAIN_PROPOSALS  # proposals kept per image during training (after RPN NMS)
    train_pre_nms: int = 24000         # highest-scoring anchors that enter RPN NMS during training
    roi_samples: int = 128             # RoIs sampled per image for the RoI head
    roi_foreground: int = 32           # at most this many of them overlap an object (IoU > 0.5)
    roi_hard_negatives: float = 0.75   # share of the background RoIs taken from partial boxes (IoU 0.1-0.5)
    augmentation: str = "full"         # the same training augmentation as YOLOv2
    mosaic: float = 0.0                # probability of a 4-image mosaic per training sample
    original_weight: float = 1.0       # sampling weight of original photos relative to public images
    batch: int = 2                     # images per update (each image goes through the network on its own)
    lr: float = 1e-3                   # SGD, momentum 0.9, for the backbone
    head_lr_factor: float = 4.0        # the neck, RPN and RoI head learn this many times faster
    weight_decay: float = 5e-4
    epochs: int = 60                   # maximum (proposal section 4.2)
    patience: int = 10                 # stop after this many epochs without a better validation AP50
    lr_schedule: str = "cosine"        # "cosine" decays to 5% of lr (as for YOLOv2); "constant"
    warmup_epochs: float = 1.0         # linear warmup from 1% of lr
    ema: bool = True                   # validate and save an exponential moving average of the weights

    def __post_init__(self):
        checks = {"rpn_positive_iou": 0.3 <= self.rpn_positive_iou < 1, "train_proposals": self.train_proposals >= 1,
                  "train_pre_nms": self.train_pre_nms >= self.train_proposals,
                  "roi_samples": self.roi_samples >= 1, "roi_foreground": 1 <= self.roi_foreground <= self.roi_samples,
                  "roi_hard_negatives": 0 <= self.roi_hard_negatives <= 1,
                  "lr_schedule": self.lr_schedule in ("constant", "cosine"), "warmup_epochs": self.warmup_epochs >= 0,
                  "augmentation": self.augmentation in ("full", "light"), "mosaic": 0 <= self.mosaic <= 1,
                  "original_weight": self.original_weight > 0, "batch": self.batch >= 1, "lr": self.lr > 0,
                  "head_lr_factor": self.head_lr_factor > 0, "weight_decay": self.weight_decay >= 0,
                  "epochs": self.epochs >= 1, "patience": self.patience >= 1}
        invalid = [name for name, ok in checks.items() if not ok]
        if invalid:
            raise ValueError(f"Invalid training settings: {', '.join(invalid)}")

    def to_dict(self):
        """The settings as a plain dict (saved in run folders and checkpoints)."""
        return asdict(self)


    @classmethod
    def from_dict(cls, values):
        """Settings saved with a run or checkpoint. Runs saved while the architecture was still a setting must have
        used this architecture (ARCHITECTURE); their other settings are kept."""
        values = dict(values)
        check_architecture(values)
        known = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in known})


# The architecture in models/faster_rcnn.py, as the development code recorded it in configs and checkpoints.
ARCHITECTURE = {"features": "c3c4", "anchor_profile": "half_octave", "aspect_ratios": list(ASPECT_RATIOS),
                "roi_pool": ROI_POOL, "pool_size": ROI_POOL, "roi_fc_layers": ROI_FC_LAYERS, "roi_box_coder": "standard",
                "anchor_scales": list(ANCHOR_SIZES)}


def check_architecture(saved):
    """Refuse settings or checkpoints from a different Faster R-CNN architecture."""
    different = [key for key, value in ARCHITECTURE.items()
                 if key in saved and (list(saved[key]) if isinstance(saved[key], (list, tuple)) else saved[key]) != value]
    if different:
        raise ValueError(f"Saved for a different Faster R-CNN architecture ({', '.join(different)}).")


# %% detector
def load_faster_rcnn(path, device):
    """Model in eval mode plus the checkpoint dict (config, best epoch, stage, protocol)."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if tuple(checkpoint["classes"]) != CLASSES:
        raise ValueError("Checkpoint class order differs from the dataset.")
    check_architecture(checkpoint)
    check_architecture(checkpoint["config"])
    model = FasterRCNN_Modular(len(CLASSES) + 1, pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint


class FasterRCNNDetector:
    """A trained Faster R-CNN with its operating settings (confidence threshold, RPN NMS IoU, pre-NMS anchors).

    Detections use the dataset's class IDs (0 = drill, 1 = scissor, 2 = screwdriver); background is removed.
    """

    name, family = FRCNN_NAME, "faster_rcnn"

    def __init__(self, model, device, *, confidence=0.25, nms_iou=RPN_NMS_IOU, pre_nms=None):
        self.model, self.device = model, torch.device(device)
        self.confidence, self.nms_iou = confidence, nms_iou
        self.pre_nms = pre_nms or 10 * INFERENCE_PROPOSALS  # 6,000 unless validation chose another count

    @classmethod
    def from_checkpoint(cls, path, device, protocol=None):
        """Load a checkpoint; with a validation protocol, use the settings chosen on validation (protocols saved
        before the pre-NMS count was selected use the fixed default)."""
        model, _ = load_faster_rcnn(path, device)
        if protocol is None:
            return cls(model, device)
        settings = protocol["models"][FRCNN_NAME]
        return cls(model, device, confidence=settings["operating_confidence"], nms_iou=settings["nms"]["nms_iou"],
                   pre_nms=settings.get("rpn_pre_nms"))

    @torch.inference_mode()
    def detect_image(self, image, score_floor, nms_ious, pre_nms_values=None):
        """One normalized image (3, 640, 640) -> {(pre-NMS anchors, RPN NMS IoU): detections}. The backbone runs
        once for all settings."""
        size = image.shape[-2:]
        features, anchors, logits, deltas = self.model([image.to(self.device)])
        results = {}
        for pre, iou in [(pre, iou) for pre in (pre_nms_values or [self.pre_nms]) for iou in nms_ious]:
            proposals, objectness = filter_proposals(logits, deltas, anchors[0], size, nms_iou=iou,
                                                     limit=INFERENCE_PROPOSALS, pre_nms=pre)
            roi_cls, roi_reg = self.model.forward_roi(features, [proposals])
            probabilities = roi_cls.softmax(1)
            # Discount RoI predictions attached to proposals with weak RPN support.
            scores = probabilities * objectness[:, None].sqrt()
            output = decode_detections(proposals, scores, roi_reg.reshape(-1, 3, 4), size, score_floor,
                                       limit=100)
            output["labels"] = output["labels"] - 1  # detector IDs 1-3 -> dataset IDs 0-2
            results[pre, iou] = output
        return results

    def detect(self, images, score_floor=None, nms_iou=None):
        """Normalized image batch (B, 3, 640, 640) -> detections, at the operating settings by default."""
        self.model.eval()
        floor = self.confidence if score_floor is None else score_floor
        iou = self.nms_iou if nms_iou is None else nms_iou
        return [self.detect_image(image, floor, [iou])[self.pre_nms, iou] for image in images]

    def predict_image(self, image):
        """A PIL RGB photo of any size -> [{"label", "confidence", "box"}] in the photo's own pixels."""
        canvas, transform = letterbox(image)
        result = self.detect(normalize(np.array(canvas))[None])[0]
        detections = []
        for box, label, score in zip(result["boxes"].tolist(), result["labels"].tolist(), result["scores"].tolist()):
            restored = restore_box(box, transform)
            if restored[2] > restored[0] and restored[3] > restored[1]:
                detections.append({"label": CLASSES[label], "confidence": score, "box": restored})
        return detections

    def export(self, loader, ground_truth, nms_ious, score_floor=AP_SCORE_FLOOR, checkpoint="", pre_nms_values=None):
        """Predictions for a whole split, one export per (pre-NMS anchors, RPN NMS IoU), in the evaluator's format."""
        self.model.eval()
        settings = [(pre, iou) for pre in (pre_nms_values or [self.pre_nms]) for iou in nms_ious]
        targets, collected = [], {setting: [] for setting in settings}
        for images, rows in loader:
            for image in images:
                for setting, output in self.detect_image(image, score_floor, nms_ious, pre_nms_values).items():
                    collected[setting].append(output)
            targets.extend(rows)
        return [predictions_from_outputs(
            targets, collected[pre, iou], ground_truth, self.name, label_mapping=dict(enumerate(range(len(CLASSES)))),
            metadata={"checkpoint": str(checkpoint), "model_family": self.family, "nms_stage": "rpn",
                      "nms_iou": iou, "nms_rule": "class_aware", "final_nms_iou": FINAL_NMS_IOU,
                      "export_score_floor": score_floor, "rpn_pre_nms": pre},
        ) for pre, iou in settings]


@torch.inference_mode()
def pipeline_diagnostics(detector, dataset):
    """Where objects are lost, per class.

    rpn_covered: some proposal overlaps the object at IoU >= 0.5 (of all proposals / of the top 100).
    roi_correct_on_exact_boxes: the RoI head names the right class when given the ground-truth box itself.
    """
    model, device = detector.model.eval(), detector.device
    counts = {name: {"objects": 0, "rpn_covered": 0, "rpn_top100_covered": 0, "roi_correct_on_exact_boxes": 0}
              for name in CLASSES}
    for index in range(len(dataset)):
        image, target = dataset[index]
        features, anchors, logits, deltas = model([image.to(device)])
        proposals, _ = filter_proposals(logits, deltas, anchors[0], image.shape[-2:], nms_iou=detector.nms_iou,
                                        limit=INFERENCE_PROPOSALS, pre_nms=detector.pre_nms)
        boxes = target["boxes"].to(device)
        if not len(boxes):
            continue
        if len(proposals):
            overlap = torchvision.ops.box_iou(boxes, proposals)
            best, best_100 = overlap.amax(1), overlap[:, :100].amax(1)
        else:
            best = best_100 = torch.zeros(len(boxes), device=device)
        exact_logits, _ = model.forward_roi(features, [boxes])
        predicted = exact_logits.argmax(1) - 1
        for label, iou, iou_100, guess in zip(target["labels"].tolist(), best.tolist(), best_100.tolist(),
                                              predicted.tolist()):
            row = counts[CLASSES[label]]
            row["objects"] += 1
            row["rpn_covered"] += iou >= 0.5
            row["rpn_top100_covered"] += iou_100 >= 0.5
            row["roi_correct_on_exact_boxes"] += guess == label
    return counts


# %% train
def frcnn_train_batch(model, images, targets, device, config):
    """One training batch: each image separately; config.roi_samples RoIs per image are sampled
    for the RoI head, at most config.roi_foreground of them overlapping an object.

    Returns the mean loss over the batch and the (foreground, background) RoI counts per image.
    """
    losses, counts = [], []
    for image, target in zip(images, targets):
        image = image.to(device)
        target = {key: target[key].to(device) for key in ("boxes", "labels")}
        features, anchors, logits, deltas = model([image])
        boxes, _ = filter_proposals(logits, deltas, anchors[0], image.shape[-2:],
                                    limit=config.train_proposals, pre_nms=config.train_pre_nms)
        boxes = torch.cat((boxes, target["boxes"]))  # the ground-truth boxes are always candidate RoIs
        overlap, _ = _match_boxes_to_gt(boxes, target["boxes"])
        positive = torch.where(overlap > 0.50)[0]
        negative = torch.where(overlap <= 0.50)[0]
        positive = positive[torch.randperm(len(positive), device=device)[:config.roi_foreground]]
        wanted = config.roi_samples - len(positive)
        if config.roi_hard_negatives:  # partial boxes first (Fast R-CNN's background range 0.1-0.5)
            hard = torch.where((overlap >= 0.10) & (overlap <= 0.50))[0]
            easy = torch.where(overlap < 0.10)[0]
            hard, easy = hard[torch.randperm(len(hard), device=device)], easy[torch.randperm(len(easy), device=device)]
            take = min(len(hard), round(config.roi_hard_negatives * wanted))
            take = max(take, wanted - len(easy))  # too few easy ones: fill with more partial boxes
            negative = torch.cat((hard[:take], easy[:wanted - min(take, len(hard))]))
        else:
            negative = negative[torch.randperm(len(negative), device=device)[:wanted]]
        sampled = boxes[torch.cat((positive, negative))]
        roi_cls, roi_reg = model.forward_roi(features, [sampled])
        loss = compute_unified_loss_per_img(logits, deltas, roi_cls, roi_reg, anchors[0], sampled, target,
                                            rpn_positive_iou=config.rpn_positive_iou)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite Faster R-CNN loss.")
        losses.append(loss)
        counts.append((len(positive), len(negative)))
    return torch.stack(losses).mean(), counts


def build_faster_rcnn(config):
    """A Faster R-CNN with ImageNet backbone weights (if config.pretrained_backbone); prints its anchor setup."""
    model = FasterRCNN_Modular(len(CLASSES) + 1, pretrained=config.pretrained_backbone)
    print(f"Features: stride {FEATURE_STRIDE} | RoIAlign {ROI_POOL}x{ROI_POOL} | anchor sizes: {ANCHOR_SIZES} | "
          f"aspect ratios: {ASPECT_RATIOS} | anchors per location: "
          f"{model.anchor_generator.num_anchors_per_location()[0]}", flush=True)
    return model


def frcnn_optimizer(model, config):
    """SGD with momentum; the parts trained from scratch (neck, RPN, RoI head) get head_lr_factor x the backbone's lr."""
    head_lr = config.head_lr_factor * config.lr
    parameters = ([{"params": part.parameters()} for part in model.pretrained_parts()]
                  + [{"params": part.parameters(), "lr": head_lr} for part in model.new_parts()])
    return torch.optim.SGD(parameters, lr=config.lr, momentum=0.9, weight_decay=config.weight_decay)


def train_faster_rcnn(config, dataset_root, work_dir, report_dir, device, *, final_from=None, protocol_path=None):
    """Train Faster R-CNN and save the checkpoint with the best validation AP50 to work_dir/best.pt.

    Development (default): training split, validation AP50 every epoch (RPN NMS 0.70), early stopping.
    Final (final_from + protocol_path): retrain from ImageNet on train + validation for the development run's
    best epoch count; the last epoch is saved.
    """
    seed_everything()
    work_dir, dataset_root = Path(work_dir), Path(dataset_root)
    work_dir.mkdir(parents=True, exist_ok=True)
    final, protocol = final_from is not None, None
    if final:
        source = torch.load(final_from, map_location="cpu", weights_only=True)
        protocol = load_json(protocol_path)
        if Path(protocol["models"][FRCNN_NAME]["validation_checkpoint"]).resolve() != Path(final_from).resolve():
            raise ValueError("Use the validation checkpoint recorded in the selected protocol.")
        if tuple(source["classes"]) != CLASSES or source.get("stage") != "validation":
            raise ValueError("Final retraining requires a compatible validation checkpoint.")
        config, epochs = FasterRCNNConfig.from_dict(source["config"]), source["best_epoch"]
    else:
        epochs = config.epochs
    train_set = ToolDataset(dataset_root, "trainval" if final else "train", augment=True, mosaic_p=config.mosaic,
                            augmentation=config.augmentation)
    loader = make_loader(train_set, config.batch, training=True, generator=torch.Generator().manual_seed(SEED),
                         original_weight=config.original_weight)
    print(f"{'Final' if final else 'Development'} training on {len(train_set)} images | {epochs} epochs max | "
          f"{len(loader)} updates/epoch", flush=True)
    print("Settings: " + ", ".join(f"{key}={value}" for key, value in config.to_dict().items()), flush=True)
    if not final:
        ap_config = EvaluationConfig(ap_backend="coco", ap_score_floor=AP_SCORE_FLOOR)
        ground_truth = load_ground_truth(dataset_root, "valid")
        valid = make_loader(ToolDataset(dataset_root, "valid"), config.batch)
    model = build_faster_rcnn(config).to(device)
    optimizer = frcnn_optimizer(model, config)
    ema = ModelEMA(model, decay=FRCNN_EMA_DECAY) if config.ema else None
    total_steps, warmup_steps = epochs * len(loader), int(config.warmup_epochs * len(loader))
    scheduler = torch.optim.lr_scheduler.LambdaLR(  # scales every parameter group's own lr
        optimizer, lambda step: lr_factor(step, total_steps, warmup_steps, config.lr_schedule))
    checkpoint = work_dir / "best.pt"
    best, best_epoch, history = -1.0, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        # each forward sees one image, so keep the pretrained BatchNorm statistics fixed
        for layer in (module for part in model.pretrained_parts() for module in part.modules()):
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
        total, foreground, background = 0.0, 0, 0
        for images, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss, counts = frcnn_train_batch(model, images, targets, device, config)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            if ema:
                ema.update(model)
            total += loss.item() * len(targets)
            foreground += sum(p for p, _ in counts)
            background += sum(n for _, n in counts)
        row = {"epoch": epoch, "train_loss": total / len(loader.dataset), "lr": optimizer.param_groups[0]["lr"],
               "sampled_foreground": foreground, "sampled_background": background}
        scored = ema.module if ema else model  # validation and checkpoints use the averaged weights
        if not final:
            exports = FasterRCNNDetector(scored, device).export(valid, ground_truth, [RPN_NMS_IOU],
                                                               ap_config.ap_score_floor, checkpoint=checkpoint)
            aps, _ = coco_ap(ground_truth, exports[0], ap_config)
            score = sum(ap["0.50"] for ap in aps) / len(aps)
            row["val_map50"] = score
        history.append(row)
        print(f"epoch {epoch:>3}/{epochs} | train_loss {row['train_loss']:.4f} | RoIs fg/bg {foreground}/{background}"
              + (f" | val_AP50 {row['val_map50']:.4f}" if not final else ""), flush=True)
        if final or score > best:
            if not final:
                best = score
            best_epoch = epoch
            torch.save({"state_dict": scored.state_dict(), **ARCHITECTURE,
                        "classes": CLASSES, "config": config.to_dict(), "best_epoch": epoch,
                        "stage": "final" if final else "validation", "protocol": protocol}, checkpoint)
        if not final and epoch - best_epoch >= config.patience:
            print(f"Early stopping: validation AP50 did not improve for {config.patience} epochs.", flush=True)
            break
    with (work_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    plot_training_history(history, report_dir or work_dir, FRCNN_NAME)
    print(f"Checkpoint: {checkpoint.resolve()} (epoch {best_epoch})", flush=True)
    return {"checkpoint": checkpoint, "best_epoch": best_epoch, "epochs_run": len(history),
            "best_map50": None if final else best, "history": history}


# %% workflow
class FasterRCNNRun(Run):
    """Paths of one Faster R-CNN run (the same layout as a YOLOv2 run)."""

    model_name = FRCNN_NAME


def frcnn_software_check(dataset_root, out_dir, device, batch=2):
    """One training update and one scoring pass on the 8 contract images. Scores are not detector results."""
    seed_everything()
    dataset_root = Path(dataset_root)
    ground_truth = load_ground_truth(dataset_root, "contract")
    names = {image["image_id"] for image in ground_truth["images"]}
    training = ToolDataset(dataset_root, "train")
    loader = make_loader(Subset(training, [i for i, name in enumerate(training.names) if name in names]), batch)
    images, targets = next(iter(loader))
    config = FasterRCNNConfig()
    model = build_faster_rcnn(config).to(device)
    optimizer = frcnn_optimizer(model, config)
    loss, counts = frcnn_train_batch(model, images, targets, device, config)
    loss.backward()
    optimizer.step()
    print(f"Training batch {tuple(images.shape)} | one-update loss={loss.item():.4f} | RoIs (fg, bg) per image {counts}")
    predictions = FasterRCNNDetector(model, device).export(loader, ground_truth, [RPN_NMS_IOU], AP_SCORE_FLOOR,
                                                           checkpoint="inspection")
    for row in predictions[0]["images"]:  # keep the display readable while the model is untrained
        for field in ("boxes", "labels", "scores"):
            row[field] = row[field][:5]
    predictions[0]["metadata"]["prediction_kind"] = "software_fixture"
    run_evaluation(ground_truth, predictions, out_dir,
                   EvaluationConfig(ap_backend="coco", ap_score_floor=AP_SCORE_FLOOR, operating_confidence=AP_SCORE_FLOOR),
                   split="contract", image_root=dataset_root / "contract" / "images", evaluation_purpose="software_check")


def frcnn_develop(config, dataset_root, run, device):
    """Development training in a new run folder; returns the checkpoint with the best validation AP50."""
    run.ensure_fresh("train")
    save_json(run.root / "config.json", config.to_dict())
    return train_faster_rcnn(config, dataset_root, run.work("train"), run.report("train"), device)["checkpoint"]


def frcnn_evaluate_split(checkpoint, dataset_root, split, work_dir, report_dir, device, *, protocol=None,
                         benchmark=False):
    """Score a checkpoint on validation (choosing pre-NMS anchors, RPN NMS and confidence) or on the test (frozen).

    On validation, every pre-NMS count in PRE_NMS_GRID is exported at every RPN NMS IoU; the count whose best
    RPN NMS IoU has the highest AP50 is kept (smaller count on a tie), then the protocol chooses NMS and confidence
    among its exports as before. The count is added to evaluation_protocol.json and used for the test.
    """
    final = split == "test"
    dataset_root, checkpoint = Path(dataset_root), Path(checkpoint)
    model, saved = load_faster_rcnn(checkpoint, device)
    if final and (saved.get("stage") != "final" or saved.get("protocol") != protocol):
        raise ValueError("The test requires the final checkpoint retrained with this frozen protocol.")
    if not final and saved.get("stage") == "final":
        raise ValueError("Select settings on validation with the development checkpoint.")
    frozen = protocol["models"][FRCNN_NAME] if final else None
    detector = FasterRCNNDetector(model, device, pre_nms=frozen.get("rpn_pre_nms") if final else None)
    floor = protocol["ap_score_floor"] if final else AP_SCORE_FLOOR
    candidates = [frozen["nms"]["nms_iou"]] if final else NMS_GRIDS["faster_rcnn"]
    pre_values = [detector.pre_nms] if final else list(PRE_NMS_GRID)
    ground_truth = load_ground_truth(dataset_root, split, allow_test=final)
    image_root = dataset_root / split / "images"
    loader = make_loader(ToolDataset(dataset_root, split, allow_test=final), 2)
    predictions = detector.export(loader, ground_truth, candidates, floor, checkpoint=checkpoint.resolve(),
                                  pre_nms_values=pre_values)
    pre_trials = []
    if not final:
        ap_config = EvaluationConfig(ap_backend="coco", ap_score_floor=floor)
        for prediction in predictions:
            aps, _ = coco_ap(ground_truth, prediction, ap_config)
            pre_trials.append({"rpn_pre_nms": prediction["metadata"]["rpn_pre_nms"],
                               "nms_iou": prediction["metadata"]["nms_iou"],
                               "map50": sum(ap["0.50"] for ap in aps) / len(aps),
                               "class_ap50": {name: ap["0.50"] for name, ap in zip(CLASSES, aps)}})
        best = max(pre_values, key=lambda pre: (max(row["map50"] for row in pre_trials if row["rpn_pre_nms"] == pre),
                                                -pre))
        predictions = [p for p in predictions if p["metadata"]["rpn_pre_nms"] == best]
        detector.pre_nms = best
        print("Pre-NMS anchor trials (AP50 at RPN NMS " + " / ".join(map(str, candidates)) + "): " + ", ".join(
            f"{pre}: " + " / ".join(f"{row['map50']:.4f}" for row in pre_trials if row["rpn_pre_nms"] == pre)
            for pre in pre_values) + f" -> {best}", flush=True)
    size = model_size_summary(model, checkpoint)
    for prediction in predictions:
        prediction["metadata"]["model_size"] = size
    if benchmark:
        samples = []
        for image in ground_truth["images"]:  # decoded before timing; preprocessing is timed
            with Image.open(image_root / image["file_name"]) as opened:
                samples.append(np.array(opened.convert("RGB")))
        settings = protocol["models"][FRCNN_NAME]
        predictions[0]["metadata"]["benchmark"] = benchmark_inference(
            lambda image: detector.detect(normalize(image)[None], settings["operating_confidence"], candidates[0]),
            samples, device=str(device), precision="float32",
            synchronize=(lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else None)
    result = run_evaluation(ground_truth, predictions, work_dir, EvaluationConfig(ap_score_floor=floor),
                            split=split, image_root=image_root, project_protocol=True, select_nms=not final,
                            frozen_protocol=protocol, results_dir=report_dir)[0]
    if not final:  # freeze the selected count with the other settings
        protocol_path = Path(work_dir) / "evaluation_protocol.json"
        written = load_json(protocol_path)
        written["models"][FRCNN_NAME].update(rpn_pre_nms=detector.pre_nms, pre_nms_trials=pre_trials)
        save_json(protocol_path, written)
        result["pre_nms_trials"] = pre_trials
    return result


def frcnn_validate(checkpoint, dataset_root, run, device):
    """Choose pre-NMS anchors, RPN NMS IoU and operating confidence on validation; writes the test's protocol."""
    run.ensure_fresh("validation")
    result = frcnn_evaluate_split(checkpoint, dataset_root, "valid", run.work("validation"), run.report("validation"),
                                  device)
    save_json(run.report("validation") / "result.json", result)
    return result


def frcnn_final_test(run, dataset_root, device):
    """Retrain on train + validation with the frozen settings, then evaluate once on the 50 test originals.

    A finished test is never repeated; an interrupted one (no result saved yet) is cleared and restarted.
    """
    if run.result("test") is not None:
        raise FileExistsError(f"{run.root} already has a final test result; it is never repeated.")
    for stage in ("final_train", "test"):
        for folder in (run.work(stage), run.report(stage)):
            if folder.exists():
                print(f"Clearing an interrupted final test: {folder}", flush=True)
                shutil.rmtree(folder)
    summary = train_faster_rcnn(None, dataset_root, run.work("final_train"), run.report("final_train"), device,
                                final_from=run.development_checkpoint(), protocol_path=run.protocol_path)
    result = frcnn_evaluate_split(summary["checkpoint"], dataset_root, "test", run.work("test"), run.report("test"),
                                  device, protocol=load_json(run.protocol_path), benchmark=True)
    save_json(run.report("test") / "result.json", result)
    return result
