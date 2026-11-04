"""The YOLOv2 training loss: anchor assignment, target encoding and the four loss terms."""

import torch
import torch.nn.functional as F

from ..boxes import box_iou, pixels_to_cxcywh, wh_iou, xywh_to_xyxy
from .yolov2 import decode

# %% loss
LOCATION_WEIGHT = 5.0     # box centre and size
OBJECT_WEIGHT = 5.0       # objectness of anchors responsible for an object (raised from 1 in round 2.2)
BACKGROUND_WEIGHT = 0.5   # objectness of anchors with no object
IGNORE_IOU = 0.6          # background anchors whose prediction already overlaps an object this much are not penalized


def encode_targets(targets):
    """Dataset targets (pixel xyxy) -> per-image tensors of [class, cx, cy, w, h], normalized."""
    return [torch.cat((row["labels"].float()[:, None], pixels_to_cxcywh(row["boxes"])), dim=1) for row in targets]


def yolo_loss(raw, targets, anchors, ignore_iou=IGNORE_IOU, objectness_target="iou", anchors_per_object=1):
    """YOLOv2 loss for raw predictions (B, A, H, W, 5 + C) and encoded targets.

    Assignment: each object goes to the grid cell containing its centre and to the free anchor whose
    shape best matches it (larger objects choose first). With anchors_per_object=k, the k best-matching free
    anchors in that cell all learn the object, so thin objects that fit no single anchor well get more
    positive examples; NMS removes the duplicate boxes at inference. Terms, summed then divided by the batch size:
      location   BCE on the x, y offsets within the cell + MSE on log(w, h / anchor)
      object     MSE between the objectness of assigned anchors and their predicted box's IoU
                 (objectness_target="one": and 1, so small objects, whose IoU is naturally lower,
                 are not pushed towards low confidence)
      background squared objectness of unassigned anchors (except ignored ones)
      class      cross-entropy on assigned anchors
    """
    b, a, h, w, _ = raw.shape
    pos = torch.zeros(raw.shape[:-1], device=raw.device, dtype=torch.bool)
    target = torch.zeros_like(raw)
    assigned_boxes = torch.zeros_like(raw[..., :4])
    collisions = 0
    for bi, gt in enumerate(targets):
        for row in gt[torch.argsort(gt[:, 3] * gt[:, 4], descending=True)]:
            cls, x, y, bw, bh = row.tolist()
            ix, iy = min(int(x * w), w - 1), min(int(y * h), h - 1)
            ranking = wh_iou(row[None, 3:5].to(raw.device), anchors).argsort(1, descending=True)[0]
            slots = [int(j) for j in ranking if not pos[bi, j, iy, ix]][:anchors_per_object]
            if not slots:  # every anchor in this cell is already taken
                collisions += 1
                continue
            for slot in slots:
                pos[bi, slot, iy, ix] = True
                target[bi, slot, iy, ix, :2] = raw.new_tensor([x * w - ix, y * h - iy])
                target[bi, slot, iy, ix, 2:4] = torch.log(row[3:5].to(raw.device) / anchors[slot] + 1e-9)
                target[bi, slot, iy, ix, 5] = cls
                assigned_boxes[bi, slot, iy, ix] = row[1:5].to(raw.device)
    predicted = decode(raw, anchors)
    negative = ~pos
    with torch.no_grad():
        for bi, gt in enumerate(targets):
            if len(gt):
                overlap = box_iou(xywh_to_xyxy(predicted[bi].reshape(-1, 4)),
                                  xywh_to_xyxy(gt[:, 1:5].to(raw.device))).max(1).values
                negative[bi] &= overlap.reshape(a, h, w) < ignore_iou
    zero = raw.sum() * 0
    if pos.any():
        location = (F.binary_cross_entropy_with_logits(raw[..., :2][pos], target[..., :2][pos], reduction="sum")
                    + F.mse_loss(raw[..., 2:4][pos], target[..., 2:4][pos], reduction="sum"))
        pp, tt = xywh_to_xyxy(predicted[pos]), xywh_to_xyxy(assigned_boxes[pos])
        inter = (torch.minimum(pp[:, 2:], tt[:, 2:]) - torch.maximum(pp[:, :2], tt[:, :2])).clamp(min=0).prod(1)
        iou = (inter / ((pp[:, 2:] - pp[:, :2]).prod(1) + (tt[:, 2:] - tt[:, :2]).prod(1) - inter + 1e-9)).detach()
        goal = iou if objectness_target == "iou" else torch.ones_like(iou)
        objectness = F.mse_loss(raw[..., 4][pos].sigmoid(), goal, reduction="sum")
        classification = F.cross_entropy(raw[..., 5:][pos], target[..., 5][pos].long(), reduction="sum")
    else:
        location = objectness = classification = zero
    background = raw[..., 4][negative].sigmoid().square().sum()
    loss = (LOCATION_WEIGHT * location + OBJECT_WEIGHT * objectness + BACKGROUND_WEIGHT * background
            + classification) / b
    details = {"loss": float(loss.detach()), "assigned": int(pos.sum()), "collisions": collisions,
               "location": float(LOCATION_WEIGHT * location.detach() / b),
               "object": float(OBJECT_WEIGHT * objectness.detach() / b),
               "background": float(BACKGROUND_WEIGHT * background.detach() / b),
               "class": float(classification.detach() / b)}
    return loss, details


def yolo_loss_on_cpu(raw, targets, anchors, objectness_target="iou", anchors_per_object=1):
    """yolo_loss with the target assignment on the CPU; the result returns to raw's device.

    The assignment loops object by object, and on a GPU every step would wait for the device. Moving the
    small raw tensor is cheap, gradients still flow back to the model, and the value is unchanged.
    """
    loss, details = yolo_loss(raw.float().cpu(), encode_targets(targets), anchors.cpu(),
                              objectness_target=objectness_target, anchors_per_object=anchors_per_object)
    return loss.to(raw.device), details
