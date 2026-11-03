"""Box formats, IoU and non-maximum suppression, shared by the model, training and evaluation.

Formats: xyxy = corners (x1, y1, x2, y2); cxcywh = centre, width, height. Model-side boxes are
normalized to [0, 1]; evaluation boxes are pixels on the 640 x 640 canvas.
"""

import torch

from .config import IMAGE_SIZE


# %% box-ops
def xywh_to_xyxy(boxes):
    """Centre-format boxes (..., 4) to corner format."""
    return torch.cat((boxes[..., :2] - boxes[..., 2:] / 2, boxes[..., :2] + boxes[..., 2:] / 2), -1)


def pixels_to_cxcywh(boxes):
    """Pixel xyxy boxes on the 640 canvas to normalized centre format, the YOLO target encoding."""
    return torch.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2,
                        boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]], dim=1) / IMAGE_SIZE


def box_iou(a, b):
    """Pairwise IoU between xyxy tensors a (N, 4) and b (M, 4) -> (N, M)."""
    inter = (torch.minimum(a[:, None, 2:], b[None, :, 2:]) - torch.maximum(a[:, None, :2], b[None, :, :2])).clamp(min=0).prod(-1)
    area_a = (a[:, 2:] - a[:, :2]).clamp(min=0).prod(-1)
    area_b = (b[:, 2:] - b[:, :2]).clamp(min=0).prod(-1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def wh_iou(a, b):
    """IoU of width/height pairs aligned at a common centre: (N, 2) x (M, 2) -> (N, M). Used for anchors."""
    inter = torch.minimum(a[:, None], b[None]).prod(-1)
    return inter / (a.prod(-1)[:, None] + b.prod(-1)[None] - inter + 1e-9)


def pair_iou(box_a, box_b):
    """IoU of two pixel xyxy boxes given as plain sequences (exact float arithmetic, for scoring)."""
    intersection = max(0.0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0])) * max(
        0.0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


# %% nms
def nms(boxes, scores, threshold=0.5):
    """Greedy non-maximum suppression: keep the highest score, drop boxes overlapping it by more than threshold."""
    order = scores.argsort(descending=True)
    keep = []
    while len(order):
        best = order[0]
        keep.append(best)
        if len(order) == 1:
            break
        rest = order[1:]
        order = rest[box_iou(boxes[best:best + 1], boxes[rest])[0] <= threshold]
    return torch.stack(keep) if keep else torch.empty(0, dtype=torch.long, device=boxes.device)
