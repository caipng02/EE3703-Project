"""Detection metrics: COCO AP, precision/recall/F1 at an operating point, error categories and subsets."""

from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from io import StringIO

from ..boxes import pair_iou
from .records import finite_number, mean_or_none, ratio, validate_ground_truth, validate_predictions


# %% config


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
            if not finite_number(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be a finite number in [0, 1].")
        if self.operating_iou <= 0:
            raise ValueError("operating_iou must be greater than zero.")
        if not 0 < self.localization_error_min_iou < self.operating_iou:
            raise ValueError("localization_error_min_iou must be between 0 and operating_iou.")
        if len(set(self.ap_ious)) != len(self.ap_ious) or not self.ap_ious:
            raise ValueError("ap_ious must be nonempty and unique.")
        if any(not finite_number(x) or not 0 < x <= 1 for x in self.ap_ious):
            raise ValueError("AP IoU thresholds must be in (0, 1].")


# %% matching


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
        eligible = [(pair_iou(box, gt_box), gt_index)
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
        precision.append(ratio(tp, tp + fp))
        recall.append(ratio(tp, matching["support"]))
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


# %% coco


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


# %% evaluate


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
        precision, recall = ratio(tp, tp + fp), ratio(tp, tp + fn)
        ious = [r["matched_iou"] for r in rows if r["tp"]]
        all_ious.extend(ious)
        all_operating_rows.extend(rows)
        per_class.append({"class_id": class_id, "class_name": name,
                          "ground_truth_instances": operating["support"],
                          "predictions_at_operating_threshold": tp + fp,
                          "tp": tp, "fp": fp, "fn": fn,
                          "precision": precision, "recall": recall,
                          "f1": ratio(2 * precision * recall, precision + recall),
                          "ap50": aps["0.50"],
                          "ap_iou_average": mean_or_none([aps[f"{t:.2f}"] for t in config.ap_ious
                                                   if aps[f"{t:.2f}"] is not None]),
                          "ap_by_iou": aps, "mean_matched_iou": mean_or_none(ious)})
    tp = sum(c["tp"] for c in per_class)
    fp = sum(c["fp"] for c in per_class)
    fn = sum(c["fn"] for c in per_class)
    p, r = ratio(tp, tp + fp), ratio(tp, tp + fn)
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
        "map50": mean_or_none([c["ap50"] for c in per_class if c["ap50"] is not None]),
        "map_iou_average": mean_or_none([c["ap_iou_average"] for c in per_class if c["ap_iou_average"] is not None]),
        "precision_micro": p, "recall_micro": r, "f1_micro": ratio(2 * p * r, p + r),
        "tp": tp, "fp": fp, "fn": fn, "mean_matched_iou": mean_or_none(all_ious),
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
        overlaps = [(pair_iou(box, b), i) for i, b in enumerate(image["boxes"])]
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


def evaluate_subsets(ground_truth: dict, predictions: dict, config: EvaluationConfig, *,
                     origins_only: bool = False) -> dict:
    """Metrics for each image origin (original/public), and optionally each source and tag."""
    groups: dict[str, list[str]] = {}
    for image in ground_truth["images"]:
        if image.get("origin"):
            groups.setdefault("origin:" + image["origin"], []).append(image["image_id"])
        if origins_only:
            continue
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
