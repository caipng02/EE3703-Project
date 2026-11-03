"""The evaluation protocol: settings chosen on validation, frozen, then applied once to the test split.

On validation the detector exports one prediction set per candidate NMS IoU; the best by AP50 is kept and
its operating confidence (for precision, recall and F1) is chosen from the proposal's grid by F1. The
resulting evaluation_protocol.json is the only way to score the test split.
"""

from contextlib import redirect_stdout
from dataclasses import asdict, replace
from datetime import datetime, timezone
from html import escape
from io import StringIO
from pathlib import Path

from .metrics import EvaluationConfig, coco_ap, evaluate_model, evaluate_subsets, match_class
from .records import finite_number, mean_or_none, ratio, save_json, validate_ground_truth, validate_predictions, write_csv
from .report import plot_ap_comparison, plot_curves, print_metrics, save_prediction_gallery

# %% grids
CONFIDENCE_GRID = (0.20, 0.30, 0.40)                                # proposal section 4.2
NMS_GRIDS = {"faster_rcnn": (0.50, 0.70), "yolov2": (0.45, 0.50, 0.60)}


# %% selection


def select_operating_confidence(ground_truth: dict, predictions: dict) -> tuple[float, list[dict]]:
    """Maximize validation micro-F1 at IoU 0.50. Prefer higher confidence on a tie."""
    gt = validate_ground_truth(ground_truth)
    pred = validate_predictions(predictions, ground_truth)
    trials = []
    for confidence in CONFIDENCE_GRID:
        tp = fp = fn = 0
        for class_id in range(len(ground_truth["classes"])):
            matching = match_class(gt, pred, class_id, 0.50, confidence)
            hits = sum(row["tp"] for row in matching["rows"])
            tp += hits
            fp += len(matching["rows"]) - hits
            fn += matching["support"] - hits
        trials.append({"confidence": confidence, "micro_f1": ratio(2 * tp, 2 * tp + fp + fn),
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
        if family not in NMS_GRIDS:
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
        if not finite_number(floor) or not 0 <= floor <= config.ap_score_floor:
            raise ValueError(f"{name}: record the shared export_score_floor at or below the AP floor.")
        if family == "yolov2" and settings[0]["nms_stage"] != "final":
            raise ValueError(f"{name}: YOLOv2 sweeps final detection NMS.")
        thresholds = [meta.get("nms_iou") for meta in settings]
        grid = NMS_GRIDS[family]
        if len(thresholds) != len(grid) or set(thresholds) != set(grid):
            raise ValueError(f"{name}: supply one export for each NMS IoU in {grid}.")
        rows = []
        for candidate, threshold in zip(candidates, thresholds):
            aps, _ = coco_ap(ground_truth, candidate, config)
            ap50 = mean_or_none([row["0.50"] for row in aps if row["0.50"] is not None])
            ap5095 = mean_or_none([value for row in aps for value in row.values() if value is not None])
            if ap50 is None:
                raise ValueError("NMS selection requires validation ground truth with objects.")
            rows.append({"nms_iou": threshold, "map50": ap50, "map_iou_average": ap5095})
        winner = max(range(len(rows)), key=lambda i: (rows[i]["map50"], rows[i]["map_iou_average"],
                                                       -rows[i]["nms_iou"]))
        winners.append(candidates[winner])
        trials[name] = rows
    return winners, trials


# %% run


def run_evaluation(ground_truth: dict, predictions_list: list[dict], output_dir: str | Path,
                   config: EvaluationConfig | None = None, *, split: str = "validation",
                   image_root: str | Path | None = None, include_subsets: bool = True,
                   evaluation_purpose: str = "model_evaluation",
                   project_protocol: bool = False, select_nms: bool = False,
                   frozen_protocol: dict | None = None, detailed: bool = False,
                   results_dir: str | Path | None = None) -> list[dict]:
    """Write report metrics and figures separately from protocol and diagnostic files."""
    config = config or EvaluationConfig(ap_score_floor=0.001 if project_protocol else 0.0)
    validate_ground_truth(ground_truth)
    if split not in ("train", "contract", "valid", "validation", "test"):
        raise ValueError("Invalid declared evaluation split.")
    if evaluation_purpose not in ("model_evaluation", "software_check"):
        raise ValueError("Invalid evaluation purpose.")
    imported_split = ground_truth.get("metadata", {}).get("dataset_split")
    if imported_split and imported_split != ("valid" if split == "validation" else split):
        raise ValueError("Declared split differs from the loaded ground-truth split.")
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
            if (not finite_number(nms["nms_iou"]) or not 0 < nms["nms_iou"] <= 1
                    or nms["nms_stage"] not in ("rpn", "final")):
                raise ValueError(f"{name}: record nms_iou and nms_stage ('rpn' or 'final').")
            if nms["nms_stage"] == "final":
                if nms["final_nms_iou"] is not None and nms["final_nms_iou"] != nms["nms_iou"]:
                    raise ValueError(f"{name}: conflicting final NMS thresholds.")
                nms["final_nms_iou"] = nms["nms_iou"]
            if not finite_number(nms["final_nms_iou"]) or not 0 < nms["final_nms_iou"] <= 1:
                raise ValueError(f"{name}: record final_nms_iou when sweeping RPN NMS.")
            floor = meta.get("export_score_floor")
            if not finite_number(floor) or not 0 <= floor <= config.ap_score_floor:
                raise ValueError(f"{name}: record export_score_floor at or below the AP floor.")
            if split == "test":
                saved = protocol["models"][name]
                if saved["nms"] != nms or saved["operating_confidence"] not in CONFIDENCE_GRID:
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
    if results_dir is not None:
        work, report_path = output.resolve(), Path(results_dir).resolve()
        if work.is_relative_to(report_path) or report_path.is_relative_to(work):
            raise ValueError("Report and diagnostic directories must be separate.")
    output.mkdir(parents=True, exist_ok=True)
    results = []
    synthetic = ground_truth["data_kind"] == "synthetic"
    banner = "SYNTHETIC TEST DATA - NOT PROJECT RESULTS" if synthetic else f"REAL DATA - declared split: {split}"
    if evaluation_purpose == "software_check":
        banner = "SOFTWARE CHECK - NOT DETECTOR RESULTS"
    if not detailed or results_dir is not None:
        report = Path(results_dir) if results_dir is not None else output
        report.mkdir(parents=True, exist_ok=True)
        print(banner)
        summaries = [banner]
        good, bad = [], []
        for index, predictions in enumerate(predictions_list, 1):
            model_config = replace(config, operating_confidence=model_confidences.get(
                predictions["model_name"], config.operating_confidence))
            result = evaluate_model(ground_truth, predictions, model_config)
            result["declared_split"] = split
            result["subsets"] = evaluate_subsets(ground_truth, predictions, model_config, origins_only=True) if include_subsets else {}
            results.append(result)
            with redirect_stdout(StringIO()) as summary:
                print_metrics(result)
            rendered = summary.getvalue().strip()
            print(f"\n{rendered}")
            summaries.append(rendered)
            curves = result["pr_curves_at_iou_0.50"]
            plot_curves({name: list(zip(curve["recall"], curve["precision"])) for name, curve in curves.items()},
                        report / f"model_{index:02d}_precision_recall.svg",
                        f"{banner} | {result['model_name']} | IoU 0.50",
                        "Recall", "Precision", unit_axes=True)
            bad.extend(row["image_id"] for row in result["error_records"])
            good.extend(row["image_id"] for row in result["operating_matches"] if row["tp"])
        # Select at most one clean success and two distinct examples with errors.
        errors = list(dict.fromkeys(bad))
        error_ids = set(errors)
        clean = next((image_id for image_id in good if image_id not in error_ids), None)
        examples = ([clean] if clean else []) + errors[:2]
        if not examples:
            examples = [ground_truth["images"][0]["image_id"]]
        selected = set(examples)
        if protocol:
            save_json(output / "evaluation_protocol.json", protocol)
            for name, trials in protocol["nms_trials"].items():
                print(f"{name} NMS trials: " + ", ".join(
                    f"{row['nms_iou']:.2f}: AP50={row['map50']:.4f}" for row in trials))
        save_prediction_gallery(
            {**ground_truth, "images": [im for im in ground_truth["images"] if im["image_id"] in selected]},
            [{**p, "images": [im for im in p["images"] if im["image_id"] in selected]} for p in predictions_list],
            report / "examples", config, image_root, model_confidences=model_confidences)
        (report / "metrics.txt").write_text("\n\n".join(summaries) + "\n", encoding="utf-8")
        print(f"Report metrics and figures: {report.resolve()}")
        if not detailed:
            return results
        results.clear()

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
        # attach measured integration metadata in these optional keys.
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
    body = ["<!doctype html><html><head><meta charset='utf-8'><title>Evaluation report</title>",
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
