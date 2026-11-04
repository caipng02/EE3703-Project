"""Faster R-CNN training, inference and the validation / final-test workflow.

The steps match the YOLOv2 engine: development training on the training split with validation AP50 every epoch;
then the RPN NMS IoU and operating confidence are chosen on validation and frozen in evaluation_protocol.json;
then the model is retrained on train + validation for the best epoch count and scored once on the 50 test originals.
The model, losses and training recipe are Person 2's (models/faster_rcnn.py).
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
from ..models.faster_rcnn import (ANCHOR_PROFILES, ASPECT_RATIOS, FINAL_NMS_IOU, INFERENCE_PROPOSALS, RPN_NMS_IOU,
                                  RPN_POSITIVE_IOU, TRAIN_PROPOSALS, FasterRCNN_Modular, _match_boxes_to_gt, compute_unified_loss_per_img,
                                  decode_detections, filter_proposals)
from .trainer import lr_factor, make_loader
from .workflow import Run


# %% config
FRCNN_NAME = "Faster R-CNN"


@dataclass(frozen=True)
class FasterRCNNConfig:
    """Every setting that affects Faster R-CNN training. Defaults are Person 2's measured development run."""

    pretrained_backbone: bool = True   # ImageNet weights for the ResNet-18 backbone
    anchor_profile: str = "dense"      # "dense": stride-16 features, anchors 32-512; "original": stride 32, 128-512
    aspect_ratios: tuple = ASPECT_RATIOS  # anchor height / width; 0.25 and 4 add shapes for long, thin tools
    rpn_positive_iou: float = RPN_POSITIVE_IOU  # anchors above this IoU with an object are RPN foreground
    train_proposals: int = TRAIN_PROPOSALS  # proposals kept per image during training (dense profile)
    roi_samples: int = 32              # RoIs sampled per image for the RoI head
    roi_foreground: int = 8            # at most this many of them overlap an object
    augmentation: str = "full"         # the same training augmentation as YOLOv2
    mosaic: float = 0.0
    original_weight: float = 1.0
    batch: int = 2                     # images per update (each image goes through the network on its own)
    lr: float = 5e-4                   # SGD, momentum 0.9, for the backbone
    head_lr_factor: float = 4.0        # RPN and RoI head learn this many times faster (dense profile)
    weight_decay: float = 5e-4
    epochs: int = 10
    patience: int = 10                 # stop after this many epochs without a better validation AP50
    lr_schedule: str = "constant"      # "cosine" decays to 5% of lr (as for YOLOv2)
    warmup_epochs: float = 0.0         # linear warmup from 1% of lr

    def __post_init__(self):
        object.__setattr__(self, "aspect_ratios", tuple(float(ratio) for ratio in self.aspect_ratios))
        checks = {"aspect_ratios": len(self.aspect_ratios) > 0 and all(r > 0 for r in self.aspect_ratios),
                  "rpn_positive_iou": 0.3 <= self.rpn_positive_iou < 1, "train_proposals": self.train_proposals >= 1,
                  "roi_samples": self.roi_samples >= 1, "roi_foreground": 1 <= self.roi_foreground <= self.roi_samples,
                  "lr_schedule": self.lr_schedule in ("constant", "cosine"), "warmup_epochs": self.warmup_epochs >= 0,"anchor_profile": self.anchor_profile in ANCHOR_PROFILES,
                  "augmentation": self.augmentation in ("full", "light"), "mosaic": 0 <= self.mosaic <= 1,
                  "original_weight": self.original_weight > 0, "batch": self.batch >= 1, "lr": self.lr > 0,
                  "head_lr_factor": self.head_lr_factor > 0, "weight_decay": self.weight_decay >= 0,
                  "epochs": self.epochs >= 1, "patience": self.patience >= 1}
        invalid = [name for name, ok in checks.items() if not ok]
        if invalid:
            raise ValueError(f"Invalid training settings: {', '.join(invalid)}")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        known = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in dict(values).items() if key in known})


# %% detector
def load_faster_rcnn(path, device):
    """Model in eval mode plus the checkpoint dict (anchor scales, config, stage, protocol)."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if tuple(checkpoint["classes"]) != CLASSES:
        raise ValueError("Checkpoint class order differs from the dataset.")
    model = FasterRCNN_Modular(len(CLASSES) + 1, pretrained=False, anchor_scales=tuple(checkpoint["anchor_scales"]),
                               aspect_ratios=tuple(checkpoint.get("aspect_ratios", ASPECT_RATIOS)))
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint


class FasterRCNNDetector:
    """A trained Faster R-CNN with its operating settings (confidence threshold and RPN NMS IoU).

    Detections use the dataset's class IDs (0 = drill, 1 = scissor, 2 = screwdriver); background is removed.
    """

    name, family = FRCNN_NAME, "faster_rcnn"

    def __init__(self, model, device, *, confidence=0.25, nms_iou=RPN_NMS_IOU):
        self.model, self.device = model, torch.device(device)
        self.confidence, self.nms_iou = confidence, nms_iou

    @classmethod
    def from_checkpoint(cls, path, device, protocol=None):
        """Load a checkpoint; with a validation protocol, use the confidence and RPN NMS IoU chosen on validation."""
        model, _ = load_faster_rcnn(path, device)
        if protocol is None:
            return cls(model, device)
        settings = protocol["models"][FRCNN_NAME]
        return cls(model, device, confidence=settings["operating_confidence"], nms_iou=settings["nms"]["nms_iou"])

    @torch.inference_mode()
    def detect_image(self, image, score_floor, nms_ious):
        """One normalized image (3, 640, 640) -> {RPN NMS IoU: detections}. The backbone runs once for all IoUs."""
        size = image.shape[-2:]
        features, anchors, logits, deltas = self.model([image.to(self.device)])
        results = {}
        for iou in nms_ious:
            proposals, objectness = filter_proposals(logits, deltas, anchors[0], size, nms_iou=iou,
                                                     limit=INFERENCE_PROPOSALS if self.model.dense else 100)
            roi_cls, roi_reg = self.model.forward_roi(features, [proposals])
            probabilities = roi_cls.softmax(1)
            # Discount RoI predictions attached to proposals with weak RPN support.
            scores = probabilities * objectness[:, None].sqrt() if self.model.dense else probabilities
            output = decode_detections(proposals, scores, roi_reg.reshape(-1, 3, 4), size, score_floor,
                                       limit=100 if self.model.dense else 20)
            output["labels"] = output["labels"] - 1  # detector IDs 1-3 -> dataset IDs 0-2
            results[iou] = output
        return results

    def detect(self, images, score_floor=None, nms_iou=None):
        """Normalized image batch (B, 3, 640, 640) -> detections, at the operating settings by default."""
        self.model.eval()
        floor = self.confidence if score_floor is None else score_floor
        iou = self.nms_iou if nms_iou is None else nms_iou
        return [self.detect_image(image, floor, [iou])[iou] for image in images]

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

    def export(self, loader, ground_truth, nms_ious, score_floor=AP_SCORE_FLOOR, checkpoint=""):
        """Predictions for a whole split, one export per RPN NMS IoU, in the evaluator's format."""
        self.model.eval()
        targets, collected = [], {iou: [] for iou in nms_ious}
        for images, rows in loader:
            for image in images:
                for iou, output in self.detect_image(image, score_floor, nms_ious).items():
                    collected[iou].append(output)
            targets.extend(rows)
        return [predictions_from_outputs(
            targets, collected[iou], ground_truth, self.name, label_mapping=dict(enumerate(range(len(CLASSES)))),
            metadata={"checkpoint": str(checkpoint), "model_family": self.family, "nms_stage": "rpn",
                      "nms_iou": iou, "nms_rule": "class_aware", "final_nms_iou": FINAL_NMS_IOU,
                      "export_score_floor": score_floor},
        ) for iou in nms_ious]


@torch.inference_mode()
def pipeline_diagnostics(detector, dataset):
    """Where objects are lost, per class (Person 2's diagnostics).

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
                                        limit=INFERENCE_PROPOSALS if model.dense else 100)
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
    """One batch of Person 2's training: each image separately; config.roi_samples RoIs per image are sampled
    for the RoI head, at most config.roi_foreground of them overlapping an object.

    Returns the mean loss over the batch and the (foreground, background) RoI counts per image.
    """
    losses, counts = [], []
    for image, target in zip(images, targets):
        image = image.to(device)
        target = {key: target[key].to(device) for key in ("boxes", "labels")}
        features, anchors, logits, deltas = model([image])
        boxes, _ = filter_proposals(logits, deltas, anchors[0], image.shape[-2:],
                                    limit=config.train_proposals if model.dense else 100)
        boxes = torch.cat((boxes, target["boxes"]))  # the ground-truth boxes are always candidate RoIs
        overlap, _ = _match_boxes_to_gt(boxes, target["boxes"])
        positive = torch.where(overlap > 0.50)[0]
        negative = torch.where(overlap <= 0.50)[0]
        positive = positive[torch.randperm(len(positive), device=device)[:config.roi_foreground]]
        negative = negative[torch.randperm(len(negative), device=device)[:config.roi_samples - len(positive)]]
        sampled = boxes[torch.cat((positive, negative))]
        roi_cls, roi_reg = model.forward_roi(features, [sampled])
        loss = compute_unified_loss_per_img(logits, deltas, roi_cls, roi_reg, anchors[0], sampled, target,
                                            balanced_rpn=model.dense, rpn_positive_iou=config.rpn_positive_iou)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite Faster R-CNN loss.")
        losses.append(loss)
        counts.append((len(positive), len(negative)))
    return torch.stack(losses).mean(), counts


def build_faster_rcnn(config):
    model = FasterRCNN_Modular(len(CLASSES) + 1, pretrained=config.pretrained_backbone,
                               anchor_scales=ANCHOR_PROFILES[config.anchor_profile], aspect_ratios=config.aspect_ratios)
    print(f"Anchor scales: {ANCHOR_PROFILES[config.anchor_profile]} | aspect ratios: {config.aspect_ratios} | "
          f"anchors per location: "
          f"{model.anchor_generator.num_anchors_per_location()[0]}", flush=True)
    return model


def frcnn_optimizer(model, config):
    """SGD with momentum; with the dense profile the RPN and RoI head get head_lr_factor x the backbone's lr."""
    parameters = model.parameters()
    if model.dense:
        head_lr = config.head_lr_factor * config.lr
        parameters = [{"params": model.backbone.parameters()},
                      {"params": model.rpn.parameters(), "lr": head_lr},
                      {"params": model.stage2.parameters(), "lr": head_lr}]
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
    total_steps, warmup_steps = epochs * len(loader), int(config.warmup_epochs * len(loader))
    scheduler = torch.optim.lr_scheduler.LambdaLR(  # scales every parameter group's own lr
        optimizer, lambda step: lr_factor(step, total_steps, warmup_steps, config.lr_schedule))
    checkpoint = work_dir / "best.pt"
    best, best_epoch, history = -1.0, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        if model.dense:  # each forward sees one image, so keep the pretrained BatchNorm statistics fixed
            for layer in model.backbone.modules():
                if isinstance(layer, nn.BatchNorm2d):
                    layer.eval()
        total, foreground, background = 0.0, 0, 0
        for images, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss, counts = frcnn_train_batch(model, images, targets, device, config)
            loss.backward()
            if model.dense:
                nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            total += loss.item() * len(targets)
            foreground += sum(p for p, _ in counts)
            background += sum(n for _, n in counts)
        row = {"epoch": epoch, "train_loss": total / len(loader.dataset), "lr": optimizer.param_groups[0]["lr"],
               "sampled_foreground": foreground, "sampled_background": background}
        if not final:
            exports = FasterRCNNDetector(model, device).export(valid, ground_truth, [RPN_NMS_IOU],
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
            torch.save({"state_dict": model.state_dict(), "anchor_scales": ANCHOR_PROFILES[config.anchor_profile],
                        "aspect_ratios": config.aspect_ratios,
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
    run.ensure_fresh("train")
    save_json(run.root / "config.json", config.to_dict())
    return train_faster_rcnn(config, dataset_root, run.work("train"), run.report("train"), device)["checkpoint"]


def frcnn_evaluate_split(checkpoint, dataset_root, split, work_dir, report_dir, device, *, protocol=None,
                         benchmark=False):
    """Score a checkpoint on validation (choosing RPN NMS and confidence) or on the test (frozen protocol)."""
    final = split == "test"
    dataset_root, checkpoint = Path(dataset_root), Path(checkpoint)
    model, saved = load_faster_rcnn(checkpoint, device)
    if final and (saved.get("stage") != "final" or saved.get("protocol") != protocol):
        raise ValueError("The test requires the final checkpoint retrained with this frozen protocol.")
    if not final and saved.get("stage") == "final":
        raise ValueError("Select settings on validation with the development checkpoint.")
    detector = FasterRCNNDetector(model, device)
    floor = protocol["ap_score_floor"] if final else AP_SCORE_FLOOR
    candidates = [protocol["models"][FRCNN_NAME]["nms"]["nms_iou"]] if final else NMS_GRIDS["faster_rcnn"]
    ground_truth = load_ground_truth(dataset_root, split, allow_test=final)
    image_root = dataset_root / split / "images"
    loader = make_loader(ToolDataset(dataset_root, split, allow_test=final), 2)
    predictions = detector.export(loader, ground_truth, candidates, floor, checkpoint=checkpoint.resolve())
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
    return run_evaluation(ground_truth, predictions, work_dir, EvaluationConfig(ap_score_floor=floor),
                          split=split, image_root=image_root, project_protocol=True, select_nms=not final,
                          frozen_protocol=protocol, results_dir=report_dir)[0]


def frcnn_validate(checkpoint, dataset_root, run, device):
    """Choose RPN NMS IoU and operating confidence on validation; writes the protocol the test will use."""
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
