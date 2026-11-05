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


def yolo_loss(raw, targets, anchors, ignore_iou=IGNORE_IOU):
    """YOLOv2 loss for raw predictions (B, A, H, W, 5 + C) and encoded targets.

    Assignment: each object goes to the grid cell containing its centre and to the free anchor whose
    shape best matches it (larger objects choose first). Terms, summed then divided by the batch size:
      location   BCE on the x, y offsets within the cell + MSE on log(w, h / anchor)
      object     MSE between the objectness of assigned anchors and their predicted box's IoU
      background squared objectness of unassigned anchors (except ignored ones)
      class      cross-entropy on assigned anchors
    """
    batch_size, num_anchors, grid_h, grid_w, _ = raw.shape
    # Per (image, anchor, cell): is it responsible for an object, its regression/class target, and that object's box
    positive = torch.zeros(raw.shape[:-1], device=raw.device, dtype=torch.bool)
    target = torch.zeros_like(raw)
    assigned_boxes = torch.zeros_like(raw[..., :4])
    collisions = 0  # objects dropped because every anchor of their cell was already taken

    # 1. Assign each object to one anchor in the cell that contains its centre
    for image_index, objects in enumerate(targets):
        largest_first = objects[torch.argsort(objects[:, 3] * objects[:, 4], descending=True)]
        for row in largest_first:
            cls, x, y, box_w, box_h = row.tolist()  # normalized centre and size
            cell_x, cell_y = min(int(x * grid_w), grid_w - 1), min(int(y * grid_h), grid_h - 1)
            # Anchors ranked by how well their shape matches the box (IoU of width/height only)
            ranking = wh_iou(row[None, 3:5].to(raw.device), anchors).argsort(1, descending=True)[0]
            slot = next((int(j) for j in ranking if not positive[image_index, j, cell_y, cell_x]), None)
            if slot is None:  # every anchor in this cell is already taken
                collisions += 1
                continue
            positive[image_index, slot, cell_y, cell_x] = True
            # Targets: offset of the centre within the cell, log of the size relative to the anchor, class
            target[image_index, slot, cell_y, cell_x, :2] = raw.new_tensor([x * grid_w - cell_x, y * grid_h - cell_y])
            target[image_index, slot, cell_y, cell_x, 2:4] = torch.log(row[3:5].to(raw.device) / anchors[slot] + 1e-9)
            target[image_index, slot, cell_y, cell_x, 5] = cls
            assigned_boxes[image_index, slot, cell_y, cell_x] = row[1:5].to(raw.device)

    # 2. Background anchors: all unassigned ones, except those whose predicted box already overlaps an object
    #    by ignore_iou or more (YOLOv2 does not punish a good box just because another anchor owns the object)
    predicted = decode(raw, anchors)
    background_mask = ~positive
    with torch.no_grad():
        for image_index, objects in enumerate(targets):
            if len(objects):
                overlap = box_iou(xywh_to_xyxy(predicted[image_index].reshape(-1, 4)),
                                  xywh_to_xyxy(objects[:, 1:5].to(raw.device))).max(1).values
                background_mask[image_index] &= overlap.reshape(num_anchors, grid_h, grid_w) < ignore_iou

    # 3. The four loss terms
    zero = raw.sum() * 0
    if positive.any():
        location = (F.binary_cross_entropy_with_logits(raw[..., :2][positive], target[..., :2][positive],
                                                       reduction="sum")
                    + F.mse_loss(raw[..., 2:4][positive], target[..., 2:4][positive], reduction="sum"))
        # IoU between each assigned anchor's predicted box and its object: the objectness target
        pred_xyxy, true_xyxy = xywh_to_xyxy(predicted[positive]), xywh_to_xyxy(assigned_boxes[positive])
        inter = (torch.minimum(pred_xyxy[:, 2:], true_xyxy[:, 2:])
                 - torch.maximum(pred_xyxy[:, :2], true_xyxy[:, :2])).clamp(min=0).prod(1)
        union = (pred_xyxy[:, 2:] - pred_xyxy[:, :2]).prod(1) + (true_xyxy[:, 2:] - true_xyxy[:, :2]).prod(1) - inter
        iou = (inter / (union + 1e-9)).detach()
        objectness = F.mse_loss(raw[..., 4][positive].sigmoid(), iou, reduction="sum")
        classification = F.cross_entropy(raw[..., 5:][positive], target[..., 5][positive].long(), reduction="sum")
    else:
        location = objectness = classification = zero
    background = raw[..., 4][background_mask].sigmoid().square().sum()
    loss = (LOCATION_WEIGHT * location + OBJECT_WEIGHT * objectness + BACKGROUND_WEIGHT * background
            + classification) / batch_size
    details = {"loss": float(loss.detach()), "assigned": int(positive.sum()), "collisions": collisions,
               "location": float(LOCATION_WEIGHT * location.detach() / batch_size),
               "object": float(OBJECT_WEIGHT * objectness.detach() / batch_size),
               "background": float(BACKGROUND_WEIGHT * background.detach() / batch_size),
               "class": float(classification.detach() / batch_size)}
    return loss, details


def yolo_loss_on_cpu(raw, targets, anchors):
    """yolo_loss with the target assignment on the CPU; the result returns to raw's device.

    The assignment loops object by object, and on a GPU every step would wait for the device. Moving the
    small raw tensor is cheap, gradients still flow back to the model, and the value is unchanged.
    """
    loss, details = yolo_loss(raw.float().cpu(), encode_targets(targets), anchors.cpu())
    return loss.to(raw.device), details
