"""Person 4 detection evaluation: one file, Python 3.10+, standard library only.

QUICK START (run in the directory containing this file):
    python person4_evaluation.py --help

Real-data example (all paths may instead be absolute):
    python person4_evaluation.py --ground-truth ground_truth.json \
        --predictions faster_rcnn.json yolo.json --split validation --output results

PROJECT SCORING: pass --project-protocol for COCO AP (requires pycocotools),
validation-selected confidence, and a protocol file for final testing.

PERSON 1 HANDOVER (data branch):
    python person4_evaluation.py --dataset-root PATH/TO/dataset --prepare-only
    python person4_evaluation.py --dataset-root PATH/TO/dataset \
        --predictions faster_rcnn.json yolo.json
The handover loader uses filenames as IDs, manifest origin/source metadata,
640x640 saved canvases, and class IDs 0 drill / 1 scissor / 2 screwdriver.
It reads COCO xywh/category IDs by name or YOLO normalized labels. No image
normalization, letterboxing, augmentation, or NMS is repeated here.

INTERNAL INPUT CONTRACT (adapters translate teammate formats at the boundary):
  Ground truth JSON:
    {"data_kind": "real", "classes": ["drill", "scissor", "screwdriver"],
     "images": [{"image_id": "001", "width": 640, "height": 640,
                 "boxes": [[10, 20, 100, 200]], "labels": [0],
                 "source": "original", "tags": ["occlusion"],
                 "file_name": "001.jpg"}]}
  Prediction JSON, one per model:
    {"data_kind": "real", "model_name": "Faster R-CNN",
     "metadata": {"checkpoint": "REPLACE", "device": "REPLACE"},
     "images": [{"image_id": "001", "boxes": [[12, 22, 99, 198]],
                 "labels": [0], "scores": [0.92]}]}

Every image MUST have a prediction record, even if boxes/labels/scores are [].
Boxes are finite xyxy pixel coordinates, continuous areas (no '+1'), within
the declared image dimensions. Both inputs use the SAME coordinate system.
There is no background class in this evaluation mapping. image_id is a string.
width/height are the coordinate canvas, not necessarily the network input size.

AP METHOD: per-class, score-ranked, one-to-one greedy matching per image,
101-point interpolated precision envelope. mAP averages classes with GT;
classes without GT have AP=null, not zero. AP50:95 averages ten IoUs.
This is a documented custom evaluator, NOT the complete official COCO evaluator:
there are no crowd/ignore flags, area ranges, or per-image maxDets rules.
Cross-check against the group's chosen reference evaluator before final reporting.

PLACEHOLDER sections are marked 'STANDARDIZE'. Sample outputs are SYNTHETIC.
No detector training, NMS, detector-specific inference/decoder, or GUI is implemented here.
The core accepts postprocessed predictions; do not run NMS a second time here.
Importing this file does not run any experiment or write any output.
"""

from __future__ import annotations

import argparse
import base64
import csv
from contextlib import redirect_stdout
import hashlib
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from html import escape
from io import StringIO
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable, Mapping, Sequence


# Class order confirmed by Person 1's data.yaml, classes.txt and data notebook.
PROJECT_CONFIDENCE_GRID = (0.20, 0.30, 0.40)
PROJECT_NMS_GRIDS = {"faster_rcnn": (0.50, 0.70), "yolov2": (0.45, 0.50, 0.60)}
PROJECT_TIMING_SCOPE = ("Decoded image in memory through preprocessing, device transfers, forward pass, "
                        "box decoding and NMS to final CPU boxes and scores")
DEFAULT_CLASS_NAMES = ("drill", "scissor", "screwdriver")


@dataclass(frozen=True)
class EvaluationConfig:
    operating_iou: float = 0.50
    operating_confidence: float = 0.50
    # AP uses ALL supplied predictions by default, independently of operating_confidence.
    ap_score_floor: float = 0.0
    ap_ious: tuple[float, ...] = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))
    localization_error_min_iou: float = 0.10
    ap_backend: str = "custom"

    def __post_init__(self) -> None:
        if self.ap_backend not in ("custom", "coco"):
            raise ValueError("ap_backend must be custom or coco.")
        for name in ("operating_iou", "operating_confidence", "ap_score_floor",
                     "localization_error_min_iou"):
            value = getattr(self, name)
            if not _finite_number(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be a finite number in [0, 1].")
        if self.operating_iou <= 0:
            raise ValueError("operating_iou must be greater than zero.")
        if not 0 < self.localization_error_min_iou < self.operating_iou:
            raise ValueError("localization_error_min_iou must be between 0 and operating_iou.")
        if len(set(self.ap_ious)) != len(self.ap_ious) or not self.ap_ious:
            raise ValueError("ap_ious must be nonempty and unique.")
        if any(not _finite_number(x) or not 0 < x <= 1 for x in self.ap_ious):
            raise ValueError("AP IoU thresholds must be in (0, 1].")


def _finite_number(value: Any) -> bool:
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def _mean(values: Sequence[float]) -> float | None:
    return statistics.mean(values) if values else None


def _ratio(numerator: int | float, denominator: int | float) -> float:
    # Explicit zero-denominator convention for P/R/F1. AP has separate null handling.
    return numerator / denominator if denominator else 0.0


def save_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_json(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8-sig") as stream:
        result = json.load(stream)
    if not isinstance(result, dict):
        raise ValueError(f"{path}: JSON root must be an object.")
    return result


def write_csv(path: str | Path, rows: list[dict], fields: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Continuous xyxy IoU; inputs are expected to have passed validation."""
    intersection = max(0.0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0])) * max(
        0.0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def validate_ground_truth(data: dict) -> dict[str, dict]:
    """Reject malformed data rather than silently dropping images or labels."""
    if data.get("data_kind") not in ("real", "synthetic"):
        raise ValueError("Ground truth must declare data_kind='real' or 'synthetic'.")
    classes = data.get("classes")
    if (not isinstance(classes, list) or not classes
            or any(not isinstance(c, str) or not c.strip() for c in classes)
            or len(set(classes)) != len(classes)):
        raise ValueError("classes must be a nonempty list of unique nonempty strings.")
    images = data.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError("Ground truth images must be a nonempty list.")
    indexed = {}
    for record in images:
        if not isinstance(record, dict):
            raise ValueError("Each ground-truth image must be an object.")
        image_id = record.get("image_id")
        if not isinstance(image_id, str) or not image_id or image_id in indexed:
            raise ValueError("Ground-truth image_id values must be unique nonempty strings.")
        for key in ("width", "height"):
            if not _finite_number(record.get(key)) or record[key] <= 0:
                raise ValueError(f"{image_id}: {key} must be positive and finite.")
        if "source" in record and (not isinstance(record["source"], str) or not record["source"]):
            raise ValueError(f"{image_id}: source must be a nonempty string.")
        tags = record.get("tags", [])
        if not isinstance(tags, list) or any(not isinstance(t, str) or not t for t in tags):
            raise ValueError(f"{image_id}: tags must be a list of nonempty strings.")
        _validate_instances(record, record, len(classes), prediction=False)
        indexed[image_id] = record
    return indexed


def _validate_instances(record: dict, image: dict, class_count: int, prediction: bool) -> None:
    image_id = record["image_id"]
    boxes, labels = record.get("boxes"), record.get("labels")
    if not isinstance(boxes, list) or not isinstance(labels, list) or len(boxes) != len(labels):
        raise ValueError(f"{image_id}: boxes and labels must be equal-length lists.")
    for box in boxes:
        if not isinstance(box, list) or len(box) != 4 or not all(_finite_number(x) for x in box):
            raise ValueError(f"{image_id}: every box must contain four finite numbers.")
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= image["width"] and 0 <= y1 < y2 <= image["height"]):
            raise ValueError(f"{image_id}: invalid/out-of-bounds xyxy box {box}.")
    if any(not isinstance(label, int) or isinstance(label, bool)
           or not 0 <= label < class_count for label in labels):
        raise ValueError(f"{image_id}: labels must be integers in [0, {class_count - 1}].")
    if prediction:
        scores = record.get("scores")
        if (not isinstance(scores, list) or len(scores) != len(boxes)
                or any(not _finite_number(s) or not 0 <= s <= 1 for s in scores)):
            raise ValueError(f"{image_id}: scores must match boxes and lie in [0, 1].")


def validate_predictions(data: dict, ground_truth: dict) -> dict[str, dict]:
    gt = validate_ground_truth(ground_truth)
    if data.get("data_kind") != ground_truth["data_kind"]:
        raise ValueError("Prediction and ground-truth data_kind must agree.")
    if "classes" in data and data["classes"] != ground_truth["classes"]:
        raise ValueError("Prediction class order differs from ground truth.")
    if not isinstance(data.get("model_name"), str) or not data["model_name"].strip():
        raise ValueError("Prediction JSON must declare a nonempty model_name.")
    if not isinstance(data.get("metadata", {}), dict):
        raise ValueError("Prediction metadata must be an object.")
    if not isinstance(data.get("images"), list):
        raise ValueError("Prediction images must be a list.")
    indexed = {}
    for record in data["images"]:
        if not isinstance(record, dict):
            raise ValueError("Each prediction image must be an object.")
        image_id = record.get("image_id")
        if not isinstance(image_id, str) or image_id not in gt or image_id in indexed:
            raise ValueError(f"Unknown or duplicate prediction image_id: {image_id!r}.")
        _validate_instances(record, gt[image_id], len(ground_truth["classes"]), prediction=True)
        indexed[image_id] = record
    missing = set(gt) - set(indexed)
    if missing:
        raise ValueError(f"Missing predictions for {sorted(missing)}. Supply explicit empty lists for no detections.")
    return indexed


def _handover_box(box: list[float], name: str, clips: list[dict]) -> list[float]:
    """Clip only <=0.001 pixel export rounding; keep core validation strict."""
    if len(box) != 4 or not all(_finite_number(v) for v in box):
        raise ValueError(f"{name}: box must have four finite coordinates.")
    outside = max(0.0, -min(box), max(box) - 640)
    if outside > 0.001:
        raise ValueError(f"{name}: box outside 640x640 by {outside:g} pixels.")
    clipped = [min(640.0, max(0.0, v)) for v in box]
    if clipped != box:
        clips.append({"file_name": name, "before": box, "after": clipped})
    if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
        raise ValueError(f"{name}: box has nonpositive area.")
    return clipped


def load_person1_ground_truth(dataset_root: str | Path, split: str = "valid", *,
                              annotation_format: str = "coco", allow_test: bool = False) -> dict:
    """Read the data-branch handover without changing its files or preprocessing.

    COCO and YOLO are cross-checked on EVERY imported image (0.001 px tolerance).
    Default COCO retains annotation precision; choose yolo to reproduce ToolDataset's
    six-decimal decoding. 'contract' contains train images, not a fourth split.
    Test access uses the same explicit allow_test convention as ToolDataset.
    """
    if split not in ("train", "valid", "test", "contract"):
        raise ValueError("Person 1 split must be train, valid, test, or contract.")
    if split == "test" and not allow_test:
        raise ValueError("Test is locked. Use allow_test=True / --allow-test only after settings are fixed.")
    if annotation_format not in ("coco", "yolo"):
        raise ValueError("annotation_format must be coco or yolo.")
    root = Path(dataset_root).resolve()
    folder = root / split
    classes = list(DEFAULT_CLASS_NAMES)
    if (root / "classes.txt").read_text(encoding="utf-8-sig").split() != classes:
        raise ValueError("classes.txt differs from Person 1's drill/scissor/screwdriver order.")
    with (root / "manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    manifest = {}
    for row in rows:
        name = row["file_name"]
        if not name or Path(name).name != name or "\\" in name or name in manifest:
            raise ValueError(f"Invalid or duplicate manifest filename: {name!r}.")
        if row["split"] not in ("train", "valid", "test") or row["origin"] not in ("original", "public"):
            raise ValueError(f"{name}: invalid manifest split/origin.")
        if row["split"] == "test" and row["origin"] != "original":
            raise ValueError("Test manifest must contain originals only.")
        manifest[name] = row
    # Structural evidence only: this cannot establish absence of near-duplicates.
    groups = {}
    for row in rows:
        group = row.get("capture_group")
        if group:
            if group in groups and groups[group] != row["split"]:
                raise ValueError(f"Capture group crosses splits: {group}.")
            groups[group] = row["split"]
    coco_path = folder / "annotations.coco.json"
    coco = load_json(coco_path)
    categories = coco["categories"]
    if (len(categories) != 3 or
            {c["id"]: c["name"] for c in categories} != dict(enumerate(classes, 1))):
        raise ValueError("COCO categories must match Person 1's IDs 1..3 and class names.")
    images, by_id, names = [], {}, set()
    hashes = {}
    for path in (root / "classes.txt", root / "manifest.csv", coco_path):
        hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for im in coco["images"]:
        name, coco_id = im["file_name"], im["id"]
        if not isinstance(coco_id, int) or isinstance(coco_id, bool) or coco_id in by_id or name in names:
            raise ValueError("COCO image IDs and filenames must be unique.")
        if name not in manifest or manifest[name]["split"] != ("train" if split == "contract" else split):
            raise ValueError(f"{name}: COCO/manifest split mismatch.")
        if (im["width"], im["height"]) != (640, 640):
            raise ValueError(f"{name}: expected saved 640x640 canvas.")
        if not (folder / "images" / name).is_file():
            raise ValueError(f"{name}: saved image missing.")
        row = manifest[name]
        record = {"image_id": name, "file_name": name, "coco_image_id": coco_id,
                  "width": 640, "height": 640, "boxes": [], "labels": [],
                  "origin": row["origin"], "source": row["source"],
                  "capture_group": row.get("capture_group", ""), "tags": []}
        images.append(record)
        by_id[coco_id] = record
        names.add(name)
    if split != "contract" and names != {r["file_name"] for r in rows if r["split"] == split}:
        raise ValueError("COCO image list differs from the manifest split.")
    clips, annotation_ids = [], set()
    for ann in coco["annotations"]:
        if ann["id"] in annotation_ids or ann["image_id"] not in by_id:
            raise ValueError("Duplicate annotation ID or unknown annotation image.")
        annotation_ids.add(ann["id"])
        if ann.get("iscrowd", 0) or ann.get("ignore", 0):
            raise ValueError("Crowd/ignore annotations need a different evaluation protocol.")
        if ann["category_id"] not in (1, 2, 3):
            raise ValueError("Unknown COCO category ID.")
        bbox = ann["bbox"]
        if len(bbox) != 4 or not all(_finite_number(v) for v in bbox) or bbox[2] <= 0 or bbox[3] <= 0:
            raise ValueError("Invalid COCO xywh box.")
        x, y, w, h = bbox
        record = by_id[ann["image_id"]]
        record["boxes"].append(_handover_box([x, y, x+w, y+h], record["file_name"], clips))
        record["labels"].append(ann["category_id"] - 1)
    max_delta = 0.0
    for record in images:
        name = record["file_name"]
        label_path = folder / "labels" / (Path(name).stem + ".txt")
        hashes[str(label_path.relative_to(root))] = hashlib.sha256(label_path.read_bytes()).hexdigest()
        yolo_boxes, yolo_labels = [], []
        for line in label_path.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"{name}: expected class cx cy w h in YOLO label.")
            label = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
            if label not in (0, 1, 2) or not all(_finite_number(v) and 0 <= v <= 1 for v in (cx, cy, w, h)) or w <= 0 or h <= 0:
                raise ValueError(f"{name}: invalid YOLO class/normalized box.")
            yolo_boxes.append(_handover_box([(cx-w/2)*640, (cy-h/2)*640,
                                             (cx+w/2)*640, (cy+h/2)*640], name, clips))
            yolo_labels.append(label)
        remaining = list(zip(yolo_labels, yolo_boxes))
        if len(remaining) != len(record["boxes"]):
            raise ValueError(f"{name}: COCO/YOLO box counts differ.")
        for label, box in zip(record["labels"], record["boxes"]):
            matches = [(max(abs(a-b) for a, b in zip(box, other)), i)
                       for i, (other_label, other) in enumerate(remaining) if label == other_label]
            delta, index = min(matches, default=(float("inf"), -1))
            if delta > 0.001:
                raise ValueError(f"{name}: COCO/YOLO labels or coordinates differ beyond rounding.")
            max_delta = max(max_delta, delta)
            remaining.pop(index)
        for label, class_name in enumerate(classes):
            if record["labels"].count(label) != int(manifest[name][class_name]):
                raise ValueError(f"{name}: manifest class count differs from annotations.")
        if annotation_format == "yolo":
            record.update(boxes=yolo_boxes, labels=yolo_labels)
    gt = {"data_kind": "real", "classes": classes, "images": images,
          "metadata": {"dataset_split": split, "annotation_format": annotation_format,
                       "coordinate_canvas": "saved 640x640 letterboxed pixels",
                       "input_sha256": hashes, "rounding_clips": clips,
                       "coco_yolo_max_delta_pixels": max_delta,
                       "image_root": str(folder / "images")}}
    validate_ground_truth(gt)
    return gt


def _plain_list(value: Any) -> list:
    """Accept Python lists or detached CPU copies of numpy/PyTorch values."""
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        raise ValueError("Expected a list, numpy array, or tensor.")
    return value


def predictions_from_outputs(targets: Sequence[dict], outputs: Sequence[dict],
                             ground_truth: dict, model_name: str, *,
                             label_mapping: Mapping[int, int], metadata: dict | None = None) -> dict:
    """Export decoded/post-NMS xyxy outputs paired with Person 1's targets.

    Accumulate ALL batches before calling: missing images are rejected. Pair in the
    actual loader order, using target['file_name'], never batch positions as IDs.
    Mapping is explicit: {0:0,1:1,2:2}, or {1:0,2:1,3:2} if background is reserved.
    Remove background upstream. This does not decode raw YOLO grids or run NMS.
    """
    if len(targets) != len(outputs):
        raise ValueError("One output dictionary is required per target/image.")
    records = []
    for target, output in zip(targets, outputs):
        labels = _plain_list(output["labels"])
        if any(not isinstance(v, int) or isinstance(v, bool) for v in labels):
            raise ValueError("Model labels must be integers.")
        records.append({"image_id": target["file_name"], "boxes": _plain_list(output["boxes"]),
                        "labels": remap_labels(labels, label_mapping),
                        "scores": _plain_list(output["scores"])})
    result = {"data_kind": ground_truth["data_kind"], "model_name": model_name,
              "classes": ground_truth["classes"], "metadata": metadata or {}, "images": records}
    validate_predictions(result, ground_truth)
    return result


# STANDARDIZE 2: model-specific decoding, NMS and checkpoint loading remain upstream.
def remap_labels(labels: Sequence[int], mapping: Mapping[int, int]) -> list[int]:
    """Example: Faster R-CNN foreground {1:0, 2:1, 3:2}; background must be removed first."""
    try:
        return [mapping[label] for label in labels]
    except KeyError as exc:
        raise ValueError(f"No canonical class mapping for label {exc.args[0]}.") from exc


def undo_letterbox_boxes(boxes: Sequence[Sequence[float]], scale: float,
                         pad_x: float, pad_y: float, width: float,
                         height: float) -> list[list[float]]:
    """Map boxes back to original pixels using ACTUAL preprocessing scale and left/top padding.

    Caller must filter zero-area boxes and corresponding labels/scores after clipping.
    This helper assumes isotropic letterboxing. It does not infer padding/rounding.
    """
    if (not all(_finite_number(x) for x in (scale, pad_x, pad_y, width, height))
            or scale <= 0 or width <= 0 or height <= 0):
        raise ValueError("Letterbox scale/dimensions must be positive and finite; padding must be finite.")
    return [[min(width, max(0.0, (b[0] - pad_x) / scale)),
             min(height, max(0.0, (b[1] - pad_y) / scale)),
             min(width, max(0.0, (b[2] - pad_x) / scale)),
             min(height, max(0.0, (b[3] - pad_y) / scale))] for b in boxes]


def match_class(gt: dict[str, dict], predictions: dict[str, dict], class_id: int,
                iou_threshold: float, score_floor: float) -> dict:
    """Score-ranked, same-class, one-to-one matching. Ties: image_id then input index."""
    candidates = []
    support = 0
    for image_id, image in gt.items():
        support += image["labels"].count(class_id)
        pred = predictions[image_id]
        for index, (label, score) in enumerate(zip(pred["labels"], pred["scores"])):
            if label == class_id and score >= score_floor:
                candidates.append((image_id, index, score))
    candidates.sort(key=lambda item: (-item[2], item[0], item[1]))
    used: dict[str, set[int]] = {image_id: set() for image_id in gt}
    rows = []
    for image_id, pred_index, score in candidates:
        image = gt[image_id]
        box = predictions[image_id]["boxes"][pred_index]
        eligible = [(box_iou(box, gt_box), gt_index)
                    for gt_index, (gt_box, label) in enumerate(zip(image["boxes"], image["labels"]))
                    if label == class_id and gt_index not in used[image_id]]
        best_iou, best_index = max(eligible, key=lambda item: (item[0], -item[1]), default=(0.0, -1))
        is_tp = best_index >= 0 and best_iou >= iou_threshold
        if is_tp:
            used[image_id].add(best_index)
        rows.append({"image_id": image_id, "prediction_index": pred_index,
                     "class_id": class_id, "score": score, "tp": int(is_tp),
                     "fp": int(not is_tp), "ground_truth_index": best_index if is_tp else None,
                     "matched_iou": best_iou if is_tp else None})
    return {"support": support, "rows": rows, "used_ground_truth": used}


def precision_recall_curve(matching: dict) -> dict:
    tp = fp = 0
    precision, recall, scores = [], [], []
    for row in matching["rows"]:
        tp += row["tp"]
        fp += row["fp"]
        precision.append(_ratio(tp, tp + fp))
        recall.append(_ratio(tp, matching["support"]))
        scores.append(row["score"])
    return {"precision": precision, "recall": recall, "scores": scores}


def average_precision_101(curve: dict, support: int) -> float | None:
    """Mean interpolated precision at recalls 0.00, 0.01, ..., 1.00."""
    if support == 0:
        return None
    recalls, precisions = curve["recall"], curve["precision"]
    envelope = list(precisions)
    for index in range(len(envelope) - 2, -1, -1):
        envelope[index] = max(envelope[index], envelope[index + 1])
    total = 0.0
    pointer = 0
    for step in range(101):
        target = step / 100
        while pointer < len(recalls) and recalls[pointer] < target:
            pointer += 1
        total += envelope[pointer] if pointer < len(envelope) else 0.0
    return total / 101


def coco_ap(ground_truth: dict, predictions: dict, config: EvaluationConfig) -> tuple[list[dict], dict]:
    """Compute official COCO bbox AP for all areas, with maxDets=100."""
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise ValueError("Project AP requires pycocotools. Install it with python -m pip install pycocotools.") from exc
    gt = validate_ground_truth(ground_truth)
    pred = validate_predictions(predictions, ground_truth)
    categories = [{"id": i + 1, "name": name} for i, name in enumerate(ground_truth["classes"])]
    images, annotations, detections = [], [], []

    def xywh(box: list) -> list:
        x1, y1, x2, y2 = box
        return [x1, y1, x2 - x1, y2 - y1]

    # Stable image IDs keep confidence ties reproducible.
    for image_id, name in enumerate(sorted(gt), 1):
        image = gt[name]
        images.append({"id": image_id, "width": image["width"], "height": image["height"]})
        for box, label in zip(image["boxes"], image["labels"]):
            bbox = xywh(box)
            annotations.append({"id": len(annotations) + 1, "image_id": image_id,
                                "category_id": label + 1, "bbox": bbox,
                                "area": bbox[2] * bbox[3], "iscrowd": 0})
        record = pred[name]
        for box, label, score in zip(record["boxes"], record["labels"], record["scores"]):
            if score >= config.ap_score_floor:
                detections.append({"image_id": image_id, "category_id": label + 1,
                                   "bbox": xywh(box), "score": score})

    with redirect_stdout(StringIO()):
        reference = COCO()
        reference.dataset = {"info": {}, "images": images, "categories": categories,
                             "annotations": annotations}
        reference.createIndex()
        # COCO.loadRes expects at least one detection.
        if detections:
            detected = reference.loadRes(detections)
        else:
            detected = COCO()
            detected.dataset = {**reference.dataset, "annotations": []}
            detected.createIndex()
        evaluator = COCOeval(reference, detected, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()

    # Axes: IoU, recall sample, category, area range, maxDets.
    precision = evaluator.eval["precision"][:, :, :, 0, -1]
    scores = evaluator.eval["scores"][0, :, :, 0, -1]
    aps, curves = [], {}
    for index, name in enumerate(ground_truth["classes"]):
        class_ap = {}
        for iou, row in zip(evaluator.params.iouThrs, precision[:, :, index]):
            valid = row[row >= 0]
            class_ap[f"{iou:.2f}"] = float(valid.mean()) if valid.size else None
        aps.append(class_ap)
        valid = precision[0, :, index] >= 0
        curves[name] = {"precision": precision[0, valid, index].tolist(),
                        "recall": evaluator.params.recThrs[valid].tolist(),
                        "scores": scores[valid, index].tolist()}
    return aps, curves


def select_operating_confidence(ground_truth: dict, predictions: dict) -> tuple[float, list[dict]]:
    """Maximize validation micro-F1 at IoU 0.50. Prefer higher confidence on a tie."""
    gt = validate_ground_truth(ground_truth)
    pred = validate_predictions(predictions, ground_truth)
    trials = []
    for confidence in PROJECT_CONFIDENCE_GRID:
        tp = fp = fn = 0
        for class_id in range(len(ground_truth["classes"])):
            matching = match_class(gt, pred, class_id, 0.50, confidence)
            hits = sum(row["tp"] for row in matching["rows"])
            tp += hits
            fp += len(matching["rows"]) - hits
            fn += matching["support"] - hits
        trials.append({"confidence": confidence, "micro_f1": _ratio(2 * tp, 2 * tp + fp + fn),
                       "tp": tp, "fp": fp, "fn": fn})
    winner = max(trials, key=lambda row: (row["micro_f1"], row["confidence"]))
    return winner["confidence"], trials


def select_nms_exports(ground_truth: dict, exports: list[dict], config: EvaluationConfig) -> tuple[list[dict], dict]:
    """Rank validation exports produced by the detector at each proposed NMS IoU."""
    groups = {}
    for export in exports:
        groups.setdefault(export["model_name"], []).append(export)
    winners, trials = [], {}
    for name, candidates in groups.items():
        settings = [candidate.get("metadata", {}) for candidate in candidates]
        family = settings[0].get("model_family")
        if family not in PROJECT_NMS_GRIDS:
            raise ValueError(f"{name}: model_family must be faster_rcnn or yolov2.")
        if any(meta.get("model_family") != family for meta in settings):
            raise ValueError(f"{name}: inconsistent model family across NMS candidates.")
        for field in ("checkpoint", "nms_stage", "nms_rule"):
            if not settings[0].get(field) or any(meta.get(field) != settings[0][field] for meta in settings):
                raise ValueError(f"{name}: keep {field} fixed across NMS candidates and record it.")
        for field in ("export_score_floor", "final_nms_iou"):
            if field == "final_nms_iou" and settings[0]["nms_stage"] == "final":
                continue
            if any(meta.get(field) != settings[0].get(field) for meta in settings):
                raise ValueError(f"{name}: keep {field} fixed across NMS candidates.")
        floor = settings[0].get("export_score_floor")
        if not _finite_number(floor) or not 0 <= floor <= config.ap_score_floor:
            raise ValueError(f"{name}: record the shared export_score_floor at or below the AP floor.")
        if family == "yolov2" and settings[0]["nms_stage"] != "final":
            raise ValueError(f"{name}: YOLOv2 sweeps final detection NMS.")
        thresholds = [meta.get("nms_iou") for meta in settings]
        grid = PROJECT_NMS_GRIDS[family]
        if len(thresholds) != len(grid) or set(thresholds) != set(grid):
            raise ValueError(f"{name}: supply one export for each NMS IoU in {grid}.")
        rows = []
        for candidate, threshold in zip(candidates, thresholds):
            aps, _ = coco_ap(ground_truth, candidate, config)
            ap50 = _mean([row["0.50"] for row in aps if row["0.50"] is not None])
            ap5095 = _mean([value for row in aps for value in row.values() if value is not None])
            if ap50 is None:
                raise ValueError("NMS selection requires validation ground truth with objects.")
            rows.append({"nms_iou": threshold, "map50": ap50, "map_iou_average": ap5095})
        winner = max(range(len(rows)), key=lambda i: (rows[i]["map50"], rows[i]["map_iou_average"],
                                                       -rows[i]["nms_iou"]))
        winners.append(candidates[winner])
        trials[name] = rows
    return winners, trials


def evaluate_model(ground_truth: dict, predictions: dict,
                   config: EvaluationConfig | None = None) -> dict:
    """Evaluate already postprocessed detections. Never tunes settings or calls a detector."""
    config = config or EvaluationConfig()
    gt = validate_ground_truth(ground_truth)
    pred = validate_predictions(predictions, ground_truth)
    per_class, all_operating_rows, all_ious = [], [], []
    official_aps, curves = coco_ap(ground_truth, predictions, config) if config.ap_backend == "coco" else (None, {})
    for class_id, name in enumerate(ground_truth["classes"]):
        if official_aps is not None:
            aps = official_aps[class_id]
        else:
            aps = {}
            for threshold in sorted(set(config.ap_ious) | {0.50}):
                matching = match_class(gt, pred, class_id, threshold, config.ap_score_floor)
                curve = precision_recall_curve(matching)
                aps[f"{threshold:.2f}"] = average_precision_101(curve, matching["support"])
                if threshold == 0.50:
                    curves[name] = curve
        operating = match_class(gt, pred, class_id, config.operating_iou, config.operating_confidence)
        rows = operating["rows"]
        tp, fp = sum(r["tp"] for r in rows), sum(r["fp"] for r in rows)
        fn = operating["support"] - tp
        precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
        ious = [r["matched_iou"] for r in rows if r["tp"]]
        all_ious.extend(ious)
        all_operating_rows.extend(rows)
        per_class.append({"class_id": class_id, "class_name": name,
                          "ground_truth_instances": operating["support"],
                          "predictions_at_operating_threshold": tp + fp,
                          "tp": tp, "fp": fp, "fn": fn,
                          "precision": precision, "recall": recall,
                          "f1": _ratio(2 * precision * recall, precision + recall),
                          "ap50": aps["0.50"],
                          "ap_iou_average": _mean([aps[f"{t:.2f}"] for t in config.ap_ious
                                                   if aps[f"{t:.2f}"] is not None]),
                          "ap_by_iou": aps, "mean_matched_iou": _mean(ious)})
    tp = sum(c["tp"] for c in per_class)
    fp = sum(c["fp"] for c in per_class)
    fn = sum(c["fn"] for c in per_class)
    p, r = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    standard_grid = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))
    result = {
        "model_name": predictions["model_name"], "data_kind": ground_truth["data_kind"],
        "model_metadata": predictions.get("metadata", {}), "config": asdict(config),
        "metric_method": ("pycocotools COCO bbox AP, all areas, maxDets=100" if official_aps is not None
                          else "Custom 101-point interpolated AP"),
        "iou_average_is_standard_50_95": tuple(sorted(config.ap_ious)) == standard_grid,
        "image_count": len(gt), "ground_truth_instances": tp + fn,
        "supplied_predictions": sum(len(x["boxes"]) for x in pred.values()),
        "evaluated_ap_predictions": sum(sum(s >= config.ap_score_floor for s in x["scores"]) for x in pred.values()),
        "classes_with_ground_truth": [c["class_name"] for c in per_class if c["ground_truth_instances"]],
        "map50": _mean([c["ap50"] for c in per_class if c["ap50"] is not None]),
        "map_iou_average": _mean([c["ap_iou_average"] for c in per_class if c["ap_iou_average"] is not None]),
        "precision_micro": p, "recall_micro": r, "f1_micro": _ratio(2 * p * r, p + r),
        "tp": tp, "fp": fp, "fn": fn, "mean_matched_iou": _mean(all_ious),
        "per_class": per_class, "pr_curves_at_iou_0.50": curves,
        "operating_matches": all_operating_rows,
    }
    result["error_records"] = analyse_errors(ground_truth, predictions, result, config)
    return result


def analyse_errors(ground_truth: dict, predictions: dict, result: dict,
                   config: EvaluationConfig) -> list[dict]:
    """Mutually exclusive descriptive FP labels, NOT experimentally established causes.

    Priority: same-class overlap >= operating IoU -> duplicate;
    other-class overlap >= operating IoU -> wrong_class;
    same-class overlap >= diagnostic minimum -> localization; otherwise background.
    Wrong-class detections also leave a FN for their unmatched GT object.
    """
    gt = {x["image_id"]: x for x in ground_truth["images"]}
    pred = {x["image_id"]: x for x in predictions["images"]}
    matched = set()
    records = []
    for row in result["operating_matches"]:
        image_id, index = row["image_id"], row["prediction_index"]
        image = gt[image_id]
        if row["tp"]:
            matched.add((image_id, row["ground_truth_index"]))
            continue
        box = pred[image_id]["boxes"][index]
        overlaps = [(box_iou(box, b), i) for i, b in enumerate(image["boxes"])]
        same = [x for x in overlaps if image["labels"][x[1]] == row["class_id"]]
        other = [x for x in overlaps if image["labels"][x[1]] != row["class_id"]]
        best_same = max(same, default=(0.0, -1))
        best_other = max(other, default=(0.0, -1))
        if best_same[0] >= config.operating_iou:
            category, evidence = "duplicate", best_same
        elif best_other[0] >= config.operating_iou:
            category, evidence = "wrong_class", best_other
        elif best_same[0] >= config.localization_error_min_iou:
            category, evidence = "localization", best_same
        else:
            category, evidence = "background_or_low_overlap", max(overlaps, default=(0.0, -1))
        records.append({"image_id": image_id, "kind": "false_positive", "category": category,
                        "prediction_index": index, "predicted_class": ground_truth["classes"][row["class_id"]],
                        "score": row["score"], "related_ground_truth_index": evidence[1] if evidence[1] >= 0 else None,
                        "overlap_iou": evidence[0], "source": image.get("source", "unknown"),
                        "tags": ";".join(image.get("tags", []))})
    for image_id, image in gt.items():
        for index, label in enumerate(image["labels"]):
            if (image_id, index) not in matched:
                records.append({"image_id": image_id, "kind": "false_negative", "category": "missed_object",
                                "ground_truth_index": index, "ground_truth_class": ground_truth["classes"][label],
                                "source": image.get("source", "unknown"), "tags": ";".join(image.get("tags", []))})
    return records


def evaluate_subsets(ground_truth: dict, predictions: dict, config: EvaluationConfig) -> dict:
    """Report Person 1's origin separately from detailed source and optional tags."""
    groups: dict[str, list[str]] = {}
    for image in ground_truth["images"]:
        if image.get("origin"):
            groups.setdefault("origin:" + image["origin"], []).append(image["image_id"])
        if image.get("source"):
            groups.setdefault("source:" + image["source"], []).append(image["image_id"])
        for tag in set(image.get("tags", [])):
            groups.setdefault("tag:" + tag, []).append(image["image_id"])
    results = {}
    for name, ids in sorted(groups.items()):
        selected = set(ids)
        subset_gt = {**ground_truth, "images": [x for x in ground_truth["images"] if x["image_id"] in selected]}
        subset_pred = {**predictions, "images": [x for x in predictions["images"] if x["image_id"] in selected]}
        evaluation = evaluate_model(subset_gt, subset_pred, config)
        results[name] = {key: evaluation[key] for key in (
            "image_count", "ground_truth_instances", "classes_with_ground_truth", "map50",
            "map_iou_average", "precision_micro", "recall_micro", "f1_micro", "tp", "fp", "fn", "per_class")}
    return results


def benchmark_inference(predict_one: Callable[[Any], Any], samples: Sequence[Any], *,
                        warmup: int = 5, repeats: int = 3,
                        synchronize: Callable[[], Any] | None = None,
                        device: str, timed_scope: str = PROJECT_TIMING_SCOPE,
                        precision: str | None = None) -> dict:
    """Batch-one wall-clock timing. Adapter must use eval/inference mode itself.

    STANDARDIZE 3: call with synchronize=torch.cuda.synchronize for CUDA.
    Supply preloaded samples unless image loading is intentionally part of scope.
    State whether preprocessing, device copies, postprocessing/NMS are included.
    The returned predictions are not evaluated or saved by this helper.
    Do not claim FPS here is batched/asynchronous throughput.
    """
    if (len(samples) == 0 or not isinstance(warmup, int) or isinstance(warmup, bool)
            or not isinstance(repeats, int) or isinstance(repeats, bool)
            or warmup < 0 or repeats < 1 or not device.strip() or not timed_scope.strip()):
        raise ValueError("Need samples, nonnegative warmup, positive repeats, device and timing scope.")
    if "cuda" in device.lower() and synchronize is None:
        raise ValueError("CUDA timing requires a synchronize callback.")
    sync = synchronize or (lambda: None)
    for index in range(warmup):
        predict_one(samples[index % len(samples)])
    sync()
    times_ms = []
    for _ in range(repeats):
        for sample in samples:
            sync()
            start = time.perf_counter()
            predict_one(sample)
            sync()
            times_ms.append((time.perf_counter() - start) * 1000)
    mean_ms = statistics.mean(times_ms)
    ordered = sorted(times_ms)
    return {"device": device, "precision": precision, "timed_scope": timed_scope, "batch_size": 1,
            "warmup_calls": warmup, "measured_calls": len(times_ms),
            "mean_latency_ms": mean_ms, "median_latency_ms": statistics.median(times_ms),
            "p95_latency_ms_nearest_rank": ordered[math.ceil(0.95 * len(ordered)) - 1],
            "sequential_fps": 1000 / mean_ms if mean_ms else None,
            "latencies_ms": times_ms}


def model_size_summary(model: Any = None, checkpoint_path: str | Path | None = None) -> dict:
    """Optional duck-typed PyTorch model; no torch import is needed by this file."""
    summary = {}
    if model is not None:
        parameters = list(model.parameters())
        summary.update(parameter_count=sum(p.numel() for p in parameters),
                       trainable_parameter_count=sum(p.numel() for p in parameters if p.requires_grad),
                       parameter_storage_bytes=sum(p.numel() * p.element_size() for p in parameters))
    if checkpoint_path is not None:
        summary["checkpoint_file_bytes"] = Path(checkpoint_path).stat().st_size
    return summary


# SVG figures are generated directly so plotting needs no external packages.
def _svg_start(width: int, height: int, title: str) -> list[str]:
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            '<style>text{font-family:Arial,sans-serif;font-size:13px} .title{font-size:17px;font-weight:bold}</style>',
            f'<text x="24" y="28" class="title">{escape(title)}</text>']


def _save_svg(path: str | Path, elements: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(elements + ["</svg>"]), encoding="utf-8")


def plot_curves(series: dict[str, list[tuple[float, float]]], path: str | Path,
                title: str, x_label: str, y_label: str, *, unit_axes: bool = False) -> None:
    """Simple labelled line chart. Missing history values should be omitted, not fabricated."""
    width, height = 1000, 570
    left, top, plot_w, plot_h = 75, 65, 570, 410
    elements = _svg_start(width, height, title)
    points = [point for values in series.values() for point in values]
    x_max = 1.0 if unit_axes else max([p[0] for p in points] + [1.0])
    y_max = 1.0 if unit_axes else max([p[1] for p in points] + [0.01]) * 1.05
    for step in range(6):
        fraction = step / 5
        x, y = left + fraction * plot_w, top + plot_h * (1 - fraction)
        elements.extend([f'<path d="M{x},{top} V{top + plot_h} M{left},{y} H{left + plot_w}" stroke="#e4e7eb" fill="none"/>',
                         f'<text x="{x}" y="{top + plot_h + 24}" text-anchor="middle">{fraction * x_max:.2g}</text>',
                         f'<text x="{left - 10}" y="{y + 4}" text-anchor="end">{fraction * y_max:.2g}</text>'])
    colors = ("#1764ab", "#cc4d16", "#228644", "#864ba8", "#946f00", "#007e87")
    for index, (name, values) in enumerate(series.items()):
        color = colors[index % len(colors)]
        points_string = " ".join(f"{left + x / x_max * plot_w:.2f},{top + (1 - y / y_max) * plot_h:.2f}" for x, y in values)
        elements.append(f'<polyline points="{points_string}" fill="none" stroke="{color}" stroke-width="2"/>')
        legend_y = 80 + index * 24
        elements.extend([f'<path d="M680,{legend_y} h24" stroke="{color}" stroke-width="3"/>',
                         f'<text x="713" y="{legend_y + 4}">{escape(name)}</text>'])
    elements.extend([f'<text x="360" y="530" text-anchor="middle">{escape(x_label)}</text>',
                     f'<text transform="translate(20,280) rotate(-90)" text-anchor="middle">{escape(y_label)}</text>'])
    _save_svg(path, elements)


def plot_training_history(history: Sequence[dict], output_dir: str | Path,
                          model_name: str = "MODEL_NAME_PLACEHOLDER") -> None:
    """STANDARDIZE 4: rows use epoch/train_loss/val_loss/val_map50; losses are optional.

    Loss magnitudes across different detector architectures are not directly comparable.
    History is supplied by teammates, not invented by this evaluator.
    """
    for row in history:
        if not _finite_number(row.get("epoch")) or row["epoch"] < 0:
            raise ValueError("History rows require a finite nonnegative epoch.")
        for key in ("train_loss", "val_loss", "val_map50"):
            if row.get(key) is not None:
                if not _finite_number(row[key]) or row[key] < 0 or (key == "val_map50" and row[key] > 1):
                    raise ValueError(f"Invalid history value for {key}.")
    if len({row["epoch"] for row in history}) != len(history):
        raise ValueError("Duplicate epochs in training history.")
    ordered = sorted(history, key=lambda row: row["epoch"])
    loss_series = {key: [(row["epoch"], row[key]) for row in ordered if row.get(key) is not None]
                   for key in ("train_loss", "val_loss")}
    loss_series = {key: values for key, values in loss_series.items() if values}
    if loss_series:
        plot_curves(loss_series, Path(output_dir) / "training_loss.svg", f"{model_name}: loss", "Epoch", "Loss")
    maps = [(row["epoch"], row["val_map50"]) for row in ordered if row.get("val_map50") is not None]
    if maps:
        plot_curves({"validation mAP@0.5": maps}, Path(output_dir) / "validation_map.svg",
                    f"{model_name}: validation mAP@0.5", "Epoch", "mAP@0.5")


def plot_ap_comparison(results: list[dict], classes: list[str], path: str | Path, title: str) -> None:
    elements = _svg_start(1000, 130 + 75 * len(classes) * len(results), title)
    row = 0
    for name in classes:
        for model in results:
            value = next(c["ap50"] for c in model["per_class"] if c["class_name"] == name)
            y = 65 + 65 * row
            row += 1
            elements.append(f'<text x="24" y="{y}">{escape(model["model_name"])} / {escape(name)}</text>')
            if value is not None:
                elements.extend([f'<rect x="360" y="{y - 15}" width="{value * 500}" height="22" fill="#1764ab"/>',
                                 f'<text x="{370 + value * 500}" y="{y}">{value:.3f}</text>'])
            else:
                elements.append(f'<text x="360" y="{y}">N/A: no ground truth for this class</text>')
    elements.append('<text x="360" y="45">AP@0.5 (bar range: 0 to 1)</text>')
    _save_svg(path, elements)


def save_prediction_gallery(ground_truth: dict, predictions_list: list[dict],
                            output_dir: str | Path, config: EvaluationConfig,
                            image_root: str | Path | None = None, *,
                            model_confidences: Mapping[str, float] | None = None) -> None:
    """One SVG per image: GT and every model on identical image/coordinate canvas.

    Without image_root: coordinate-only diagrams, explicitly labelled as such.
    With image_root: embed real PNG/JPEG/GIF/WebP images using relative file_name.
    No EXIF orientation correction is applied: exported pixels must match annotations.
    """
    gt = validate_ground_truth(ground_truth)
    predictions = [validate_predictions(p, ground_truth) for p in predictions_list]
    root = Path(image_root).resolve() if image_root is not None else None
    for number, (image_id, image) in enumerate(gt.items(), 1):
        background = None
        if root is not None:
            if not isinstance(image.get("file_name"), str) or not image["file_name"]:
                raise ValueError(f"{image_id}: image_root requires file_name.")
            path = (root / image["file_name"]).resolve()
            if not path.is_relative_to(root):
                raise ValueError("file_name must stay within image_root.")
            mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                    ".gif": "image/gif", ".webp": "image/webp"}.get(path.suffix.lower())
            if mime is None:
                raise ValueError(f"Unsupported image extension: {path.suffix}")
            background = f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")
        panel_w = 420
        scale = min(380 / image["width"], 380 / image["height"])
        canvas_w, canvas_h = image["width"] * scale, image["height"] * scale
        synthetic = ground_truth["data_kind"] == "synthetic"
        title = ("SYNTHETIC TEST DATA - " if synthetic else "") + image_id
        if any(p.get("metadata", {}).get("prediction_kind") == "software_fixture" for p in predictions_list):
            title = "SOFTWARE CHECK - NOT DETECTOR RESULTS: " + image_id
        elements = _svg_start(panel_w * (len(predictions) + 1), 510, title)
        subtitle = "Embedded image; verify pixel orientation and canvas size" if background else "Coordinate-only diagram: no photograph supplied"
        elements.append(f'<text x="24" y="50">{subtitle}</text>')
        panels = [("Ground truth", image, "#147b34", False)] + [
            (p["model_name"], indexed[image_id], "#b0381d", True)
            for p, indexed in zip(predictions_list, predictions)]
        for panel_index, (name, record, color, is_prediction) in enumerate(panels):
            confidence = (model_confidences or {}).get(name, config.operating_confidence)
            x0, y0 = 20 + panel_index * panel_w, 100
            elements.append(f'<text x="{x0}" y="80">{escape(name)}</text>')
            elements.append(f'<rect x="{x0}" y="{y0}" width="{canvas_w}" height="{canvas_h}" fill="#f4f5f7" stroke="#999"/>')
            if background:
                elements.append(f'<image x="{x0}" y="{y0}" width="{canvas_w}" height="{canvas_h}" preserveAspectRatio="none" href="{background}"/>')
            for index, (box, label) in enumerate(zip(record["boxes"], record["labels"])):
                if is_prediction and record["scores"][index] < confidence:
                    continue
                x1, y1, x2, y2 = box
                x, y = x0 + x1 * scale, y0 + y1 * scale
                text_label = ground_truth["classes"][label]
                if is_prediction:
                    text_label += f' {record["scores"][index]:.2f}'
                elements.append(f'<rect x="{x}" y="{y}" width="{(x2 - x1) * scale}" height="{(y2 - y1) * scale}" fill="none" stroke="{color}" stroke-width="2"/>')
                elements.append(f'<text x="{x}" y="{max(y0 + 14, y - 4)}" fill="{color}" stroke="white" stroke-width="3" paint-order="stroke">{escape(text_label)}</text>')
            if is_prediction:
                elements.append(f'<text x="{x0}" y="495">Display confidence threshold: {confidence:g}</text>')
        _save_svg(Path(output_dir) / f"image_{number:03d}.svg", elements)


def run_evaluation(ground_truth: dict, predictions_list: list[dict], output_dir: str | Path,
                   config: EvaluationConfig | None = None, *, split: str = "validation",
                   image_root: str | Path | None = None, include_subsets: bool = True,
                   evaluation_purpose: str = "model_evaluation",
                   project_protocol: bool = False, select_nms: bool = False,
                   frozen_protocol: dict | None = None) -> list[dict]:
    """Integration entry point: evaluate, save tables/figures, and produce an HTML index."""
    config = config or EvaluationConfig(ap_score_floor=0.001 if project_protocol else 0.0)
    validate_ground_truth(ground_truth)
    if split not in ("train", "contract", "valid", "validation", "test"):
        raise ValueError("Invalid declared evaluation split.")
    if evaluation_purpose not in ("model_evaluation", "software_check"):
        raise ValueError("Invalid evaluation purpose.")
    imported_split = ground_truth.get("metadata", {}).get("dataset_split")
    if imported_split and imported_split != ("valid" if split == "validation" else split):
        raise ValueError("Declared split differs from imported handover split.")
    if evaluation_purpose == "software_check" and split == "test":
        raise ValueError("Software checks must use development data.")
    if not predictions_list:
        raise ValueError("At least one prediction set is required.")
    names = [p.get("model_name") for p in predictions_list]
    for pred in predictions_list:
        validate_predictions(pred, ground_truth)
        if pred.get("metadata", {}).get("prediction_kind") == "software_fixture" and evaluation_purpose != "software_check":
            raise ValueError("Fabricated software fixtures require evaluation_purpose='software_check'.")
    if len(set(names)) != len(names) and not select_nms:
        raise ValueError("Model names must be unique within a comparison.")
    protocol = None
    if project_protocol:
        if ground_truth["data_kind"] != "real" or evaluation_purpose != "model_evaluation":
            raise ValueError("The project protocol requires real evaluation inputs.")
        if split not in ("valid", "validation", "test"):
            raise ValueError("Use validation or test for the project protocol.")
        if not include_subsets:
            raise ValueError("The project protocol includes origin breakdowns.")
        if split == "test" and (select_nms or frozen_protocol is None):
            raise ValueError("Final test requires a frozen validation protocol and cannot select NMS.")
        if split != "test" and frozen_protocol is not None:
            raise ValueError("Use a frozen protocol only for final testing.")
        config = EvaluationConfig(ap_backend="coco", ap_score_floor=config.ap_score_floor)
        protocol = {"version": 1, "selected_on": "validation", "ap_backend": "coco",
                    "ap_score_floor": config.ap_score_floor, "matching_iou": 0.50,
                    "models": {}, "nms_trials": {}}
        if split == "test":
            if len(ground_truth["images"]) != 50 or any(im.get("origin") != "original" for im in ground_truth["images"]):
                raise ValueError("Final test requires exactly 50 images marked as original.")
            if (frozen_protocol.get("version"), frozen_protocol.get("selected_on"),
                frozen_protocol.get("ap_backend"), frozen_protocol.get("matching_iou")) != (1, "validation", "coco", 0.50):
                raise ValueError("Invalid frozen validation protocol.")
            config = replace(config, ap_score_floor=frozen_protocol["ap_score_floor"])
            if set(frozen_protocol["models"]) != set(names):
                raise ValueError("Test model names must match the frozen validation protocol.")
            protocol = frozen_protocol
        elif select_nms:
            predictions_list, protocol["nms_trials"] = select_nms_exports(ground_truth, predictions_list, config)

        final_rules = {pred.get("metadata", {}).get("nms_rule") for pred in predictions_list}
        if final_rules != {"class_aware"}:
            raise ValueError("Record metadata.nms_rule='class_aware' for final detection NMS.")
        for pred in predictions_list:
            name = pred["model_name"]
            meta = pred.get("metadata", {})
            nms = {key: meta.get(key) for key in ("nms_iou", "nms_stage", "nms_rule", "final_nms_iou")}
            if (not _finite_number(nms["nms_iou"]) or not 0 < nms["nms_iou"] <= 1
                    or nms["nms_stage"] not in ("rpn", "final")):
                raise ValueError(f"{name}: record nms_iou and nms_stage ('rpn' or 'final').")
            if nms["nms_stage"] == "final":
                if nms["final_nms_iou"] is not None and nms["final_nms_iou"] != nms["nms_iou"]:
                    raise ValueError(f"{name}: conflicting final NMS thresholds.")
                nms["final_nms_iou"] = nms["nms_iou"]
            if not _finite_number(nms["final_nms_iou"]) or not 0 < nms["final_nms_iou"] <= 1:
                raise ValueError(f"{name}: record final_nms_iou when sweeping RPN NMS.")
            floor = meta.get("export_score_floor")
            if not _finite_number(floor) or not 0 <= floor <= config.ap_score_floor:
                raise ValueError(f"{name}: record export_score_floor at or below the AP floor.")
            if split == "test":
                saved = protocol["models"][name]
                if saved["nms"] != nms or saved["operating_confidence"] not in PROJECT_CONFIDENCE_GRID:
                    raise ValueError(f"{name}: test settings differ from the frozen validation protocol.")
            else:
                confidence, trials = select_operating_confidence(ground_truth, pred)
                protocol["models"][name] = {"operating_confidence": confidence,
                                            "confidence_trials": trials, "nms": nms,
                                            "validation_checkpoint": meta.get("checkpoint")}
    elif select_nms or frozen_protocol is not None:
        raise ValueError("NMS selection and frozen settings require the project protocol.")
    model_confidences = {name: settings["operating_confidence"]
                         for name, settings in protocol["models"].items()} if protocol else {}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    synthetic = ground_truth["data_kind"] == "synthetic"
    banner = "SYNTHETIC TEST DATA - NOT PROJECT RESULTS" if synthetic else f"REAL DATA - declared split: {split}"
    if evaluation_purpose == "software_check":
        banner = "SOFTWARE CHECK - NOT DETECTOR RESULTS"
    summary_rows, subset_rows = [], []
    manifest = {"generated_utc": datetime.now(timezone.utc).isoformat(), "data_kind": ground_truth["data_kind"],
                "evaluation_purpose": evaluation_purpose, "ground_truth_metadata": ground_truth.get("metadata", {}),
                "declared_split": split, "config": asdict(config), "classes": ground_truth["classes"],
                "image_ids": [x["image_id"] for x in ground_truth["images"]],
                "models": [], "note": "Split integrity, checkpoint provenance and training leakage require external verification."}
    for index, predictions in enumerate(predictions_list, 1):
        model_config = replace(config, operating_confidence=model_confidences.get(
            predictions["model_name"], config.operating_confidence))
        result = evaluate_model(ground_truth, predictions, model_config)
        result["evaluation_purpose"] = evaluation_purpose
        result["declared_split"] = split
        result["subsets"] = evaluate_subsets(ground_truth, predictions, model_config) if include_subsets else {}
        results.append(result)
        model_dir = output / f"model_{index:02d}"
        save_json(model_dir / "metrics.json", result)
        write_csv(model_dir / "per_class.csv", result["per_class"], [
            "class_id", "class_name", "ground_truth_instances", "predictions_at_operating_threshold",
            "tp", "fp", "fn", "precision", "recall", "f1", "ap50", "ap_iou_average", "mean_matched_iou"])
        write_csv(model_dir / "errors.csv", result["error_records"], [
            "image_id", "kind", "category", "prediction_index", "predicted_class", "score",
            "related_ground_truth_index", "overlap_iou", "ground_truth_index", "ground_truth_class", "source", "tags"])
        for name, curve in result["pr_curves_at_iou_0.50"].items():
            class_index = ground_truth["classes"].index(name)
            write_csv(model_dir / f"pr_class_{class_index}.csv", [dict(zip(("score", "precision", "recall"), values))
                      for values in zip(curve["scores"], curve["precision"], curve["recall"])], ["score", "precision", "recall"])
        plot_curves({name: list(zip(curve["recall"], curve["precision"]))
                     for name, curve in result["pr_curves_at_iou_0.50"].items()}, model_dir / "pr_curves.svg",
                    f'{banner}: {predictions["model_name"]}', "Recall", "Precision", unit_axes=True)
        summary = {key: result[key] for key in ("model_name", "image_count", "ground_truth_instances", "map50",
                   "map_iou_average", "precision_micro", "recall_micro", "f1_micro", "tp", "fp", "fn", "mean_matched_iou")}
        # STANDARDIZE 6: attach measured integration metadata in these optional keys.
        # Blank cells mean NOT MEASURED; no timings or sizes are fabricated.
        metadata = predictions.get("metadata", {})
        benchmark = metadata.get("benchmark", {})
        size = metadata.get("model_size", {})
        if not isinstance(benchmark, dict) or not isinstance(size, dict):
            raise ValueError("metadata.benchmark and metadata.model_size must be objects.")
        summary.update(data_kind=ground_truth["data_kind"], evaluation_purpose=evaluation_purpose, split=split,
                       operating_confidence=model_config.operating_confidence, ap_backend=config.ap_backend,
                       mean_latency_ms=benchmark.get("mean_latency_ms"),
                       sequential_fps=benchmark.get("sequential_fps"),
                       device=benchmark.get("device"), precision=benchmark.get("precision"), timed_scope=benchmark.get("timed_scope"),
                       parameter_count=size.get("parameter_count"),
                       checkpoint_file_bytes=size.get("checkpoint_file_bytes"))
        summary_rows.append(summary)
        for subset_name, subset in result["subsets"].items():
            subset_rows.append({"model_name": result["model_name"], "subset": subset_name,
                               **{k: subset[k] for k in ("image_count", "ground_truth_instances", "map50", "map_iou_average", "precision_micro", "recall_micro")},
                               "classes_with_ground_truth": ";".join(subset["classes_with_ground_truth"])})
        manifest["models"].append({"model_name": result["model_name"], "directory": model_dir.name,
                                   "metadata": predictions.get("metadata", {})})
    write_csv(output / "comparison.csv", summary_rows, list(summary_rows[0]))
    write_csv(output / "subsets.csv", subset_rows, ["model_name", "subset", "image_count", "ground_truth_instances",
              "classes_with_ground_truth", "map50", "map_iou_average", "precision_micro", "recall_micro"])
    if protocol:
        manifest["project_protocol"] = protocol
        save_json(output / "evaluation_protocol.json", protocol)
    save_json(output / "run_manifest.json", manifest)
    # Snapshot exact inputs so plots/tables can be regenerated without another inference run.
    save_json(output / "ground_truth_snapshot.json", ground_truth)
    for index, predictions in enumerate(predictions_list, 1):
        save_json(output / f"predictions_{index:02d}_snapshot.json", predictions)
    plot_ap_comparison(results, ground_truth["classes"], output / "ap_comparison.svg", banner)
    save_prediction_gallery(ground_truth, predictions_list, output / "gallery", config, image_root,
                            model_confidences=model_confidences)
    links = ["comparison.csv", "subsets.csv", "run_manifest.json", "ground_truth_snapshot.json"]
    if protocol:
        links.append("evaluation_protocol.json")
    body = ["<!doctype html><html><head><meta charset='utf-8'><title>Person 4 evaluation</title>",
            "<style>body{font:16px Arial;max-width:1300px;margin:30px auto;padding:20px}img{max-width:100%}h1{color:#923414}</style></head><body>",
            f"<h1>{escape(banner)}</h1>",
            ("<p>Real training annotations with fabricated detections. This checks the software, not detector accuracy.</p>"
             if evaluation_purpose == "software_check" else ""),
            f'<p>{escape(results[0]["metric_method"])}. IoU grid: 0.50:0.05:0.95.</p>',
            f"<p>Operating IoU: {config.operating_iou}; AP score floor: {config.ap_score_floor}.</p>",
            "<p>AP uses supplied score-ranked predictions, independently of the operating confidence. Earlier model filtering can still truncate AP.</p>",
            "<p>AP is N/A for classes without ground truth; they are excluded from mAP. P/R/F1 use zero for zero denominators.</p>",
            "<p>Source/tag subsets may overlap and have different supported classes. Check counts before comparing their mAP.</p>",
            "<ul>" + "".join(f'<li><a href="{link}">{link}</a></li>' for link in links) + "</ul>",
            '<h2>Per-class AP comparison</h2><img src="ap_comparison.svg" alt="Per-class AP">']
    for index, result in enumerate(results, 1):
        body.append(f'<h2>{escape(result["model_name"])}</h2>'
                    f'<p>Operating confidence: {result["config"]["operating_confidence"]:g}</p><p><a href="model_{index:02d}/metrics.json">Metrics JSON</a> | '
                    f'<a href="model_{index:02d}/per_class.csv">Per-class CSV</a> | <a href="model_{index:02d}/errors.csv">Error records</a></p>'
                    f'<img src="model_{index:02d}/pr_curves.svg" alt="Precision recall curves">')
    body.append("<h2>Same-image comparisons</h2><p>Coordinate-only diagrams unless image_root was supplied. Diagnostic error categories are descriptions, not proven failure causes.</p>")
    for index, image in enumerate(ground_truth["images"], 1):
        body.append(f'<h3>{escape(image["image_id"])}</h3><img src="gallery/image_{index:03d}.svg" alt="Ground truth and model predictions">')
    (output / "index.html").write_text("\n".join(body + ["</body></html>"]), encoding="utf-8")
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--ground-truth", type=Path, help="Ground-truth JSON in the documented common format.")
    mode.add_argument("--dataset-root", type=Path, help="Person 1's dataset folder containing manifest.csv, train/, valid/, contract/.")
    parser.add_argument("--dataset-split", choices=("train", "valid", "contract", "test"), default="valid")
    parser.add_argument("--annotation-format", choices=("coco", "yolo"), default="coco", help="Both are checked; select which precision to evaluate.")
    parser.add_argument("--allow-test", action="store_true", help="Unlock final test import after settings are fixed, like Person 1's allow_test.")
    parser.add_argument("--prepare-only", action="store_true", help="Import/validate handover and save ground_truth.json without detector predictions.")
    parser.add_argument("--project-protocol", action="store_true", help="Use COCO AP and validation-selected operating confidence.")
    parser.add_argument("--select-nms", action="store_true", help="Compare validation exports for every proposed NMS candidate.")
    parser.add_argument("--frozen-protocol", type=Path, help="Validation evaluation_protocol.json required for project test.")
    parser.add_argument("--predictions", nargs="+", type=Path, help="One prediction JSON path per model.")
    parser.add_argument("--output", type=Path, help="Output directory; existing named report files are replaced.")
    parser.add_argument("--split", choices=("validation", "test"), default="validation", help="Record intended split; does not verify provenance.")
    parser.add_argument("--confidence", type=float, default=0.5, help="Operating threshold for P/R/F1, errors and gallery, not AP.")
    parser.add_argument("--iou", type=float, default=0.5, help="Operating IoU for P/R/F1 and errors; AP grid stays 0.50:0.05:0.95.")
    parser.add_argument("--ap-score-floor", type=float, default=None, help="Minimum supplied score used for AP; record any upstream filtering too.")
    parser.add_argument("--image-root", type=Path, help="Optional folder containing original image files for overlays.")
    parser.add_argument("--no-subsets", action="store_true", help="Skip source/tag subset evaluation.")
    args = parser.parse_args(argv)
    if (args.prepare_only or args.allow_test or args.dataset_split != "valid" or args.annotation_format != "coco") and not args.dataset_root:
        parser.error("Handover options require --dataset-root.")
    if args.prepare_only and args.predictions:
        parser.error("--prepare-only cannot consume model predictions.")
    if (args.select_nms or args.frozen_protocol) and not args.project_protocol:
        parser.error("--select-nms and --frozen-protocol require --project-protocol.")
    if args.project_protocol and args.prepare_only:
        parser.error("--project-protocol requires real model predictions.")
    if args.project_protocol and (args.confidence != 0.5 or args.iou != 0.5):
        parser.error("Project operating confidence comes from validation and matching IoU is fixed at 0.50.")
    if args.frozen_protocol and args.ap_score_floor is not None:
        parser.error("Final test reads the AP floor from the frozen protocol.")
    try:
        base = Path(__file__).resolve().parent
        floor = args.ap_score_floor if args.ap_score_floor is not None else (0.001 if args.project_protocol else 0.0)
        config = EvaluationConfig(operating_iou=args.iou, operating_confidence=args.confidence, ap_score_floor=floor)
        if args.dataset_root:
            gt = load_person1_ground_truth(args.dataset_root, args.dataset_split,
                                           annotation_format=args.annotation_format, allow_test=args.allow_test)
            declared_split = args.dataset_split
            image_root = args.image_root or args.dataset_root / args.dataset_split / "images"
        else:
            gt = load_json(args.ground_truth)
            declared_split, image_root = args.split, args.image_root
        if args.prepare_only:
            output = args.output or base / "person4_prepared" / args.dataset_split
            save_json(output / "ground_truth.json", gt)
            print(f"Prepared {len(gt['images'])} real images; no detector evaluated.")
            print(f"Ground truth: {(output / 'ground_truth.json').resolve()}")
            return 0
        if not args.predictions:
            parser.error("Evaluation requires --predictions; use --prepare-only to import handover data alone.")
        preds = [load_json(p) for p in args.predictions]
        output = args.output or base / "person4_evaluation_outputs"
        if args.dataset_root and (args.no_subsets or args.split != "validation"):
            parser.error("Handover evaluation requires origin/source subsets; use --dataset-split to select its split.")
        results = run_evaluation(gt, preds, output, config, split=declared_split,
                                 image_root=image_root, include_subsets=not args.no_subsets,
                                 project_protocol=args.project_protocol, select_nms=args.select_nms,
                                 frozen_protocol=load_json(args.frozen_protocol) if args.frozen_protocol else None)
        print(f'{gt["data_kind"].upper()} DATA; declared split: {declared_split}')
        for result in results:
            map_text = f'{result["map50"]:.6f}' if result["map50"] is not None else "N/A"
            print(f'{result["model_name"]}: mAP@0.5={map_text}; TP={result["tp"]}, FP={result["fp"]}, FN={result["fn"]}')
        print(f'Report: {(output / "index.html").resolve()}')
        return 0
    except (ValueError, OSError, TypeError, KeyError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
