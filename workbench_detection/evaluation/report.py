"""Figures and printed summaries for evaluation results. SVG is written directly, so no plotting library is needed."""

import base64
from html import escape
from pathlib import Path
from typing import Mapping, Sequence

from .metrics import EvaluationConfig
from .records import finite_number, validate_ground_truth, validate_predictions


# %% svg


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
    """Rows use epoch/train_loss/val_loss/val_map50; losses are optional.

    Loss magnitudes across different detector architectures are not directly comparable.
    History is supplied by teammates, not invented by this evaluator.
    """
    for row in history:
        if not finite_number(row.get("epoch")) or row["epoch"] < 0:
            raise ValueError("History rows require a finite nonnegative epoch.")
        for key in ("train_loss", "val_loss", "val_map50"):
            if row.get(key) is not None:
                if not finite_number(row[key]) or row[key] < 0 or (key == "val_map50" and row[key] > 1):
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
    """Grouped bar chart of AP50 per class for one or more models, saved to path."""
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


# %% summary


def print_metrics(result: dict) -> None:
    """Print one model's overall, per-class and per-subset metrics as a short text summary."""
    def value(number):
        return "N/A" if number is None else f"{number:.4f}"

    config = result["config"]
    print(f"\n{result['model_name']} | {result['declared_split']} | {result['image_count']} images")
    print(f"AP50={value(result['map50'])}  AP50:95={value(result['map_iou_average'])}")
    print(f"Operating confidence={config['operating_confidence']:.3f}  matching IoU={config['operating_iou']:.2f}")
    print(f"Precision={value(result['precision_micro'])}  Recall={value(result['recall_micro'])}  "
          f"F1={value(result['f1_micro'])}  matched IoU={value(result['mean_matched_iou'])}")
    print(f"TP={result['tp']}  FP={result['fp']}  FN={result['fn']}")
    for row in result["per_class"]:
        print(f"  {row['class_name']}: AP50={value(row['ap50'])}  AP50:95={value(row['ap_iou_average'])}  "
              f"objects={row['ground_truth_instances']}")
    for name, row in result["subsets"].items():
        if name.startswith("origin:"):
            print(f"  {name}: {row['image_count']} images, {row['ground_truth_instances']} objects, "
                  f"AP50={value(row['map50'])}, AP50:95={value(row['map_iou_average'])}")
    metadata = result["model_metadata"]
    nms = metadata.get("nms_iou")
    if nms is not None:
        print(f"NMS IoU={nms:.2f}")
    timing = metadata.get("benchmark", {})
    size = metadata.get("model_size", {})
    if timing:
        print(f"Latency={value(timing.get('mean_latency_ms'))} ms/image  "
              f"FPS={value(timing.get('sequential_fps'))}  device={timing.get('device')}")
    if size:
        print(f"Parameters={size.get('parameter_count')}  checkpoint bytes={size.get('checkpoint_file_bytes')}")
