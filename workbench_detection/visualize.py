"""Figures for the notebook: data, augmentation, anchors, training curves, predictions and errors.

Boxes are drawn in pixel xyxy on the 640 x 640 canvas: ground truth in green, predictions in class colours.
"""

from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .config import CLASSES, IMAGE_SIZE

# %% drawing
CLASS_COLOURS = {"drill": "#1f77b4", "scissor": "#d62728", "screwdriver": "#ff7f0e"}
TRUTH_COLOUR = "#2ca02c"


def _axes_grid(count, columns, size=3.2):
    import matplotlib.pyplot as plt

    rows = max(1, -(-count // columns))
    figure, axes = plt.subplots(rows, columns, figsize=(size * columns, size * rows), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    return figure, axes.flat


def draw_boxes(axis, image, boxes=(), labels=(), scores=None, colour=None, title=None):
    """Show an image with xyxy boxes; labels are class IDs, scores optional."""
    from matplotlib.patches import Rectangle

    axis.imshow(image)
    for index, (box, label) in enumerate(zip(boxes, labels)):
        name = CLASSES[int(label)]
        x1, y1, x2, y2 = [float(v) for v in box]
        edge = colour or CLASS_COLOURS[name]
        axis.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=edge, linewidth=2))
        text = name if scores is None else f"{name} {float(scores[index]):.2f}"
        axis.text(x1, max(y1 - 4, 10), text, color="white", fontsize=8, bbox=dict(facecolor=edge, pad=1, linewidth=0))
    if title:
        axis.set_title(title, fontsize=9)
    axis.axis("off")


def draw_detections(image, detections):
    """PIL image with GUI-format detections drawn on it (used by the predict command)."""
    output = image.copy()
    pen = ImageDraw.Draw(output)
    width = max(2, round(max(image.size) / 300))
    for detection in detections:
        colour = CLASS_COLOURS[detection["label"]]
        pen.rectangle(detection["box"], outline=colour, width=width)
        pen.text((detection["box"][0] + width, detection["box"][1] + width),
                 f"{detection['label']} {detection['confidence']:.2f}", fill=colour)
    return output


# %% data-figures
def split_summary(rows):
    """Images, originals, public and boxes per class for each split, from the manifest rows."""
    table = []
    for split in ("train", "valid", "test"):
        part = [row for row in rows if row["split"] == split]
        table.append({"split": split, "images": len(part),
                      "original": sum(row["origin"] == "original" for row in part),
                      "public": sum(row["origin"] == "public" for row in part),
                      **{name: sum(int(row[name]) for row in part) for name in CLASSES}})
    return table


def plot_class_balance(table):
    """Bar chart of boxes per class in each split (from class_balance)."""
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 3.2))
    width = 0.25
    for offset, name in enumerate(CLASSES):
        axis.bar(np.arange(len(table)) + (offset - 1) * width, [row[name] for row in table], width,
                 label=name, color=CLASS_COLOURS[name])
    axis.set_xticks(range(len(table)), [row["split"] for row in table])
    axis.set_ylabel("boxes")
    axis.set_title("Boxes per class and split")
    axis.legend()
    plt.show()


def show_samples(dataset, indices, columns=4):
    """Training images with their ground-truth boxes."""
    import matplotlib.pyplot as plt

    _, axes = _axes_grid(len(indices), columns)
    for axis, index in zip(axes, indices):
        image, boxes, labels = dataset.raw(index)
        draw_boxes(axis, image, boxes, labels, title=f"{dataset.names[index]} ({dataset.origins[index]})")
    plt.tight_layout()
    plt.show()


def show_augmentations(dataset, index, count=8, columns=4, seed=0):
    """The same training image drawn several times through the training augmentation."""
    import matplotlib.pyplot as plt
    import torch

    from .data.dataset import denormalize

    torch.manual_seed(seed)
    _, axes = _axes_grid(count, columns)
    for number, axis in enumerate(axes):
        if number >= count:
            break
        tensor, target = dataset[index]
        draw_boxes(axis, denormalize(tensor), target["boxes"], target["labels"], title=f"draw {number + 1}")
    plt.tight_layout()
    plt.show()


def plot_box_counts(before, after):
    """Boxes per class in one epoch without and with class-aware sampling."""
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6, 3))
    positions = np.arange(len(CLASSES))
    axis.bar(positions - 0.2, before, 0.4, label="uniform sampling", color="#bbbbbb")
    axis.bar(positions + 0.2, after, 0.4, label="class-aware sampling", color="#1f77b4")
    axis.set_xticks(positions, CLASSES)
    axis.set_ylabel("expected boxes per epoch")
    axis.legend()
    plt.show()


# %% model-figures
def plot_anchors(dataset, anchors):
    """Training box sizes with the k-means anchor shapes on top."""
    import matplotlib.pyplot as plt

    sizes = np.concatenate([boxes[:, 2:] - boxes[:, :2] for boxes in dataset.boxes]) / IMAGE_SIZE
    labels = np.concatenate(dataset.labels)
    figure, axis = plt.subplots(figsize=(5.5, 4.5))
    for class_id, name in enumerate(CLASSES):
        chosen = labels == class_id
        axis.scatter(sizes[chosen, 0], sizes[chosen, 1], s=10, alpha=0.5, color=CLASS_COLOURS[name], label=name)
    anchors = np.asarray(anchors)
    axis.scatter(anchors[:, 0], anchors[:, 1], s=180, marker="X", color="black", label="anchors")
    axis.set_xlabel("box width (fraction of image)")
    axis.set_ylabel("box height (fraction of image)")
    axis.set_title(f"Training box sizes and the {len(anchors)} anchors")
    axis.legend()
    plt.show()


def show_assignment(dataset, index, anchors, grid=IMAGE_SIZE // 32):
    """Each object's responsible grid cell and the anchor shape assigned to it (best width/height IoU)."""
    import matplotlib.pyplot as plt
    import torch
    from matplotlib.patches import Rectangle

    from .boxes import wh_iou

    image, boxes, labels = dataset.raw(index)
    figure, axis = plt.subplots(figsize=(5.5, 5.5))
    draw_boxes(axis, image, boxes, labels, colour=TRUTH_COLOUR, title="green: object | dashed: responsible cell and anchor")
    cell = IMAGE_SIZE / grid
    for x in range(grid + 1):
        axis.axvline(x * cell, color="white", linewidth=0.3)
        axis.axhline(x * cell, color="white", linewidth=0.3)
    for box in boxes:
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        size = torch.tensor([[(box[2] - box[0]) / IMAGE_SIZE, (box[3] - box[1]) / IMAGE_SIZE]])
        slot = int(wh_iou(size, torch.as_tensor(anchors)).argmax())
        aw, ah = float(anchors[slot][0]) * IMAGE_SIZE, float(anchors[slot][1]) * IMAGE_SIZE
        ix, iy = int(cx // cell), int(cy // cell)
        axis.add_patch(Rectangle((ix * cell, iy * cell), cell, cell, color="yellow", alpha=0.6))
        axis.add_patch(Rectangle((cx - aw / 2, cy - ah / 2), aw, ah, fill=False, edgecolor="yellow",
                                 linestyle="--", linewidth=1.5))
        axis.text(cx, cy, f"anchor {slot}", color="black", fontsize=8, ha="center",
                  bbox=dict(facecolor="yellow", pad=1, linewidth=0))
    plt.show()


# %% result-figures
def plot_history(history, title="Training"):
    """Loss curves and validation AP50 per epoch (loss only for a final retrain, which has no validation data)."""
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    if "val_map50" not in history[0]:
        figure, left = plt.subplots(figsize=(5.5, 3.6))
        left.plot(epochs, [row["train_loss"] for row in history], label="train")
        left.set(title=f"{title}: loss", xlabel="epoch", yscale="log")
        left.legend()
        plt.tight_layout()
        plt.show()
        return
    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 3.6))
    left.plot(epochs, [row["train_loss"] for row in history], label="train")
    if "val_loss" in history[0]:
        left.plot(epochs, [row["val_loss"] for row in history], label="validation")
    if "val_map50" in history[0]:
        right.plot(epochs, [row["val_map50"] for row in history], color="#2ca02c")
        best = max(history, key=lambda row: row["val_map50"])
        right.scatter([best["epoch"]], [best["val_map50"]], color="black", zorder=3,
                      label=f"best {best['val_map50']:.3f} (epoch {int(best['epoch'])})")
        right.legend()
    left.set(title=f"{title}: loss", xlabel="epoch", yscale="log")
    left.legend()
    right.set(title=f"{title}: validation AP50", xlabel="epoch", ylim=(0, 1))
    plt.tight_layout()
    plt.show()


def plot_pr_curves(result):
    """Precision-recall curve of each class at IoU 0.50, with its AP50 in the legend."""
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(5, 4))
    for name, curve in result["pr_curves_at_iou_0.50"].items():
        row = next(r for r in result["per_class"] if r["class_name"] == name)
        ap = "n/a" if row["ap50"] is None else f"{row['ap50']:.3f}"
        axis.plot(curve["recall"], curve["precision"], color=CLASS_COLOURS[name], label=f"{name} (AP50 {ap})")
    axis.set(xlabel="recall", ylabel="precision", xlim=(0, 1), ylim=(0, 1.02), title="Precision-recall at IoU 0.50")
    axis.legend()
    plt.show()


def plot_error_breakdown(result):
    """Counts of each error type at the operating confidence."""
    import matplotlib.pyplot as plt

    counts = Counter(row["category"] for row in result["error_records"])
    names = ["missed_object", "background_or_low_overlap", "localization", "wrong_class", "duplicate"]
    figure, axis = plt.subplots(figsize=(7, 3))
    axis.barh(names, [counts.get(name, 0) for name in names], color="#888888")
    axis.set_xlabel("count")
    axis.set_title(f"Errors at confidence {result['config']['operating_confidence']:g} "
                   f"(TP {result['tp']}, FP {result['fp']}, FN {result['fn']})")
    plt.show()
    return counts


def show_predictions(detector, dataset_root, split, ground_truth, image_ids, columns=3):
    """Ground truth (green) and the detector's predictions at its operating settings, image by image."""
    import matplotlib.pyplot as plt

    from .data.dataset import normalize

    records = {image["image_id"]: image for image in ground_truth["images"]}
    axes = list(_axes_grid(2 * len(image_ids), 2 * min(columns, len(image_ids)))[1])
    for number, image_id in enumerate(image_ids):
        image = np.array(Image.open(Path(dataset_root) / split / "images" / image_id).convert("RGB"))
        detection = detector.detect(normalize(image)[None])[0]
        truth = records[image_id]
        draw_boxes(axes[2 * number], image, truth["boxes"], truth["labels"], colour=TRUTH_COLOUR,
                   title=f"{image_id}: ground truth")
        draw_boxes(axes[2 * number + 1], image, detection["boxes"].tolist(), detection["labels"].tolist(),
                   detection["scores"].tolist(), title=f"prediction (conf >= {detector.confidence:g})")
    plt.tight_layout()
    plt.show()


def pick_examples(result, kinds=("missed_object", "background_or_low_overlap", "localization", "wrong_class"), per_kind=1):
    """Image IDs of clean successes and of each error type, for side-by-side display."""
    errors = {}
    for row in result["error_records"]:
        errors.setdefault(row["category"], []).append(row["image_id"])
    failed = {row["image_id"] for row in result["error_records"]}
    successes = list(dict.fromkeys(row["image_id"] for row in result["operating_matches"]
                                   if row["tp"] and row["image_id"] not in failed))
    chosen = {"success": successes[:2]}
    for kind in kinds:
        chosen[kind] = list(dict.fromkeys(errors.get(kind, [])))[:per_kind]
    return chosen
