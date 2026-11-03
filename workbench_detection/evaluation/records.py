"""Ground truth and prediction records for evaluation, and the checks that keep them consistent.

Ground truth: {"data_kind": "real", "classes": [...], "images": [{"image_id", "width", "height",
"boxes": [[x1, y1, x2, y2], ...] (pixels), "labels": [class IDs], "origin", "source", ...}]}.
Predictions: the same images with "scores", plus "model_name" and "metadata" (checkpoint, NMS settings).
Every image needs a prediction record, with empty lists when nothing is detected.
"""

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..config import CLASSES


# %% helpers
def finite_number(value: Any) -> bool:
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def mean_or_none(values: Sequence[float]) -> float | None:
    return statistics.mean(values) if values else None


def ratio(numerator: int | float, denominator: int | float) -> float:
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


# %% validation


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
            if not finite_number(record.get(key)) or record[key] <= 0:
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
        if not isinstance(box, list) or len(box) != 4 or not all(finite_number(x) for x in box):
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
                or any(not finite_number(s) or not 0 <= s <= 1 for s in scores)):
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


# %% ground-truth


def _canvas_box(box: list[float], name: str, clips: list[dict]) -> list[float]:
    """Clip only <=0.001 pixel export rounding; keep core validation strict."""
    if len(box) != 4 or not all(finite_number(v) for v in box):
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


def load_ground_truth(dataset_root: str | Path, split: str = "valid", *,
                              annotation_format: str = "coco", allow_test: bool = False) -> dict:
    """Load one split's ground truth from the dataset folder, cross-checking COCO and YOLO labels.

    COCO and YOLO are cross-checked on EVERY imported image (0.001 px tolerance).
    Default COCO retains annotation precision; choose yolo to reproduce ToolDataset's
    six-decimal decoding. 'contract' contains train images, not a fourth split.
    Test access uses the same explicit allow_test convention as ToolDataset.
    """
    if split not in ("train", "valid", "test", "contract"):
        raise ValueError("Split must be train, valid, test, or contract.")
    if split == "test" and not allow_test:
        raise ValueError("Test is locked. Use allow_test=True / --allow-test only after settings are fixed.")
    if annotation_format not in ("coco", "yolo"):
        raise ValueError("annotation_format must be coco or yolo.")
    root = Path(dataset_root).resolve()
    folder = root / split
    classes = list(CLASSES)
    if (root / "classes.txt").read_text(encoding="utf-8-sig").split() != classes:
        raise ValueError("classes.txt differs from the drill/scissor/screwdriver order.")
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
        raise ValueError("COCO categories must use IDs 1..3 with the project class names.")
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
        if len(bbox) != 4 or not all(finite_number(v) for v in bbox) or bbox[2] <= 0 or bbox[3] <= 0:
            raise ValueError("Invalid COCO xywh box.")
        x, y, w, h = bbox
        record = by_id[ann["image_id"]]
        record["boxes"].append(_canvas_box([x, y, x+w, y+h], record["file_name"], clips))
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
            if label not in (0, 1, 2) or not all(finite_number(v) and 0 <= v <= 1 for v in (cx, cy, w, h)) or w <= 0 or h <= 0:
                raise ValueError(f"{name}: invalid YOLO class/normalized box.")
            yolo_boxes.append(_canvas_box([(cx-w/2)*640, (cy-h/2)*640,
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


# %% export


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
    """Export decoded, post-NMS pixel xyxy outputs paired with the dataset targets.

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


# model-specific decoding, NMS and checkpoint loading remain upstream.
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
    if (not all(finite_number(x) for x in (scale, pad_x, pad_y, width, height))
            or scale <= 0 or width <= 0 or height <= 0):
        raise ValueError("Letterbox scale/dimensions must be positive and finite; padding must be finite.")
    return [[min(width, max(0.0, (b[0] - pad_x) / scale)),
             min(height, max(0.0, (b[1] - pad_y) / scale)),
             min(width, max(0.0, (b[2] - pad_x) / scale)),
             min(height, max(0.0, (b[3] - pad_y) / scale))] for b in boxes]
