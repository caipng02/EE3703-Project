"""Faster R-CNN with a ResNet-18 backbone: region proposal network, RoI head, losses and decoding.

Stage 1, the RPN, scores every anchor as object or background and refines it into a proposal; NMS keeps the best
proposals. Stage 2, the RoI head, pools each proposal's features (RoIAlign), then classifies it as background,
drill, scissor or screwdriver and refines its box for that class. Detector class IDs: 0 = background, 1 = drill,
2 = scissor, 3 = screwdriver (dataset IDs plus 1).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.image_list import ImageList

# %% settings
# Anchors: 11 sizes in half-octave steps (x1.41) from 16 to 512 px, so a box between two octave sizes still fits an
# anchor well, and 5 aspect ratios (height / width) for long, thin tools: 55 anchors per feature cell.
ANCHOR_SIZES = (16, 23, 32, 45, 64, 91, 128, 181, 256, 362, 512)
ASPECT_RATIOS = (0.25, 0.5, 1.0, 2.0, 4.0)
FEATURE_STRIDE = 8                  # layer 3 upsampled and added to layer 2: an 80 x 80 map for 640 x 640 images
ROI_POOL = 7                        # RoIAlign output size (7 x 7)
ROI_FC_LAYERS = 2                   # fully connected layers of 1024 in the RoI head
# RoI head box offsets are scaled by these weights and trained with smooth L1 at beta 1/9 (torchvision's Faster
# R-CNN), so errors of a few percent of the box size still give a full-size gradient. The RPN uses raw offsets.
BOX_WEIGHTS = (10.0, 10.0, 5.0, 5.0)
BOX_BETA = 1 / 9
RPN_BOX_LAMBDA = 10.0               # weight of the RPN box loss
RPN_POSITIVE_IOU = 0.60             # an anchor overlapping an object above this IoU is foreground
RPN_NMS_IOU = 0.70
FINAL_NMS_IOU = 0.50
TRAIN_PROPOSALS = 600
INFERENCE_PROPOSALS = 600


# %% model
class RPN(nn.Module):
    """Region proposal network head: for every anchor at every feature cell, an objectness logit and 4 box offsets.

    A shared 3x3 convolution, then two 1x1 convolutions: (B, A, H, W) objectness and (B, 4A, H, W) offsets.
    """

    def __init__(self, in_channels=256, num_anchors=55):
        super().__init__()
        self.rpn_head = nn.Sequential(  # shared 3x3 convolution
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)     # objectness per anchor
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)  # (tx, ty, tw, th) per anchor

    def forward(self, x):
        """Feature map (B, C, H, W) -> (objectness logits, box offsets)."""
        t = self.rpn_head(x)
        return self.cls_logits(t), self.bbox_pred(t)


class Stage2_network(nn.Module):
    """The second stage (RoI head): pools each proposal's features and classifies and refines it.

    RoIAlign crops every proposal from the feature map to a fixed 7 x 7 grid; two fully connected layers of 1024
    feed a 4-way classifier (background + 3 tools) and class-specific box offsets (4 per tool class).
    """

    def __init__(self, in_channels=256, num_classes=4, *, stride=FEATURE_STRIDE, pool_size=ROI_POOL,
                 fc_layers=ROI_FC_LAYERS):
        super().__init__()
        # spatial_scale maps image-pixel boxes onto the stride-8 feature map
        self.roi_align = torchvision.ops.RoIAlign(output_size=(pool_size, pool_size), spatial_scale=1.0 / stride,
                                                  sampling_ratio=2, aligned=True)

        # Fully connected layers of 1024 (two, as in torchvision's Faster R-CNN head)
        layers, width = [], in_channels * pool_size ** 2
        for _ in range(fc_layers):
            layers += [nn.Linear(in_features=width, out_features=1024), nn.LeakyReLU(0.01)]
            width = 1024
        self.fc = nn.Sequential(*layers)

        self.cls_score = nn.Linear(1024, num_classes)            # background + 3 tool classes
        self.bbox_head = nn.Linear(1024, (num_classes - 1) * 4)  # box offsets for each tool class

    def forward(self, features, proposals):
        """Feature map and a list of (N, 4) proposal boxes per image -> (class logits (N, 4), offsets (N, 12))."""
        pooled = self.roi_align(features, proposals)
        x = pooled.flatten(start_dim=1)
        x = self.fc(x)
        return self.cls_score(x), self.bbox_head(x)


class FPNMerge(nn.Module):
    """Layer 3 (stride 16) upsampled and added to a 1x1 projection of layer 2 (stride 8), then a 3x3 smoothing
    convolution: the stride-8 map keeps layer 2's position detail and layer 3's semantics (one FPN level)."""

    def __init__(self, fine_channels=128, coarse_channels=256):
        super().__init__()
        self.lateral = nn.Conv2d(fine_channels, coarse_channels, kernel_size=1)
        self.smooth = nn.Conv2d(coarse_channels, coarse_channels, kernel_size=3, padding=1)

    def forward(self, fine, coarse):
        """Layer-2 map (B, 128, 80, 80) and layer-3 map (B, 256, 40, 40) -> merged map (B, 256, 80, 80)."""
        merged = self.lateral(fine) + F.interpolate(coarse, size=fine.shape[-2:], mode="nearest")
        return self.smooth(merged)


class FasterRCNN_Modular(nn.Module):
    """The complete Faster R-CNN: ResNet-18 backbone, feature merge, anchors, RPN and RoI head.

    forward() runs stage 1 (backbone, anchors, RPN); the caller filters proposals (filter_proposals) and passes
    them to forward_roi() for stage 2. Keeping the stages separate lets training sample RoIs between them.
    """

    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        backbone_model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)

        # ResNet-18 up to layer 2 (stride 8, 128 channels) and layer 3 (stride 16, 256 channels), merged by the neck
        # into one stride-8 map with 256 channels.
        children = list(backbone_model.children())
        self.backbone = nn.Sequential(*children[:6])   # stem + layers 1-2
        self.backbone_top = children[6]                # layer 3
        self.neck = FPNMerge(128, 256)
        channels = 256

        # 55 anchors per feature cell (11 sizes x 5 aspect ratios); the RPN predicts one output set per anchor
        self.anchor_generator = AnchorGenerator(sizes=(ANCHOR_SIZES,), aspect_ratios=(ASPECT_RATIOS,))
        self.rpn = RPN(in_channels=channels, num_anchors=self.anchor_generator.num_anchors_per_location()[0])
        self.stage2 = Stage2_network(in_channels=channels, num_classes=num_classes)

        # Small initial weights for the output layers: objectness and class scores start near uniform
        for layer in (self.rpn.cls_logits, self.rpn.bbox_pred, self.stage2.cls_score, self.stage2.bbox_head):
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.zeros_(layer.bias)

    def pretrained_parts(self):
        """The ResNet-18 layers (ImageNet weights); the neck, RPN and RoI head are trained from scratch."""
        return [self.backbone, self.backbone_top]

    def new_parts(self):
        """The layers trained from scratch (they get a higher learning rate)."""
        return [self.neck, self.rpn, self.stage2]

    def forward(self, images):
        """Stage 1. A list of (3, 640, 640) images -> (features, anchors per image, RPN logits, RPN offsets)."""
        image_sizes = [(img.shape[-2], img.shape[-1]) for img in images]
        image_list = ImageList(torch.stack(images), image_sizes)  # the batch format the anchor generator expects

        # 1. Feature extraction: layers 2 and 3 merged into the stride-8 map
        fine = self.backbone(image_list.tensors)
        features = self.neck(fine, self.backbone_top(fine))

        # 2. Anchors: 55 per cell of the 80 x 80 map, in image pixels
        anchors = self.anchor_generator(image_list, [features])

        # 3. RPN predictions for every anchor
        rpn_cls, rpn_reg = self.rpn(features)
        return features, anchors, rpn_cls, rpn_reg

    def forward_roi(self, features, proposals):
        """Stage 2: class logits and box offsets for the given proposals."""
        return self.stage2(features, proposals)


# %% loss
def calculate_transforms(ref_boxes, gt_boxes, weights=None):
    """Encode target boxes as offsets (t_x, t_y, t_w, t_h) from reference boxes (anchors or proposals).

    Boxes are (xmin, ymin, xmax, ymax). t_x, t_y: centre shift in units of the reference width and height;
    t_w, t_h: log of the size ratio. `weights` scales the four offsets (BOX_WEIGHTS for the RoI head).
    """
    # 1. Reference box sizes and centres
    widths = ref_boxes[:, 2] - ref_boxes[:, 0]
    heights = ref_boxes[:, 3] - ref_boxes[:, 1]
    ctr_x = ref_boxes[:, 0] + 0.5 * widths
    ctr_y = ref_boxes[:, 1] + 0.5 * heights

    # 2. Ground-truth box sizes and centres
    gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1]
    gt_ctr_x = gt_boxes[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_boxes[:, 1] + 0.5 * gt_heights

    # 3. Prevent division by zero
    widths = torch.clamp(widths, min=1.0)
    heights = torch.clamp(heights, min=1.0)
    gt_widths = torch.clamp(gt_widths, min=1.0)
    gt_heights = torch.clamp(gt_heights, min=1.0)

    # 4. Offsets
    t_x = (gt_ctr_x - ctr_x) / widths
    t_y = (gt_ctr_y - ctr_y) / heights
    t_w = torch.log(gt_widths / widths)
    t_h = torch.log(gt_heights / heights)

    target_transforms = torch.stack((t_x, t_y, t_w, t_h), dim=1)
    if weights is not None:
        target_transforms = target_transforms * target_transforms.new_tensor(weights)
    return target_transforms


def _prepare_rpn_outputs(rpn_cls_raw, rpn_reg_raw):
    """Flatten the RPN outputs into one row per anchor, in the anchor generator's order (cell by cell)."""
    rpn_cls_flat = rpn_cls_raw.permute(0, 2, 3, 1).flatten()           # (H * W * A,) objectness logits
    rpn_reg_flat = rpn_reg_raw.permute(0, 2, 3, 1).reshape(-1, 4)      # (H * W * A, 4) offsets
    return rpn_cls_flat, rpn_reg_flat


def _match_boxes_to_gt(query_boxes, gt_boxes):
    """For each query box (anchor or proposal): its highest IoU with a ground-truth box, and that box's index."""
    if gt_boxes.numel() == 0 or query_boxes.numel() == 0:  # no objects (or no boxes): every IoU is 0
        return (torch.zeros(query_boxes.shape[0], device=query_boxes.device),
                torch.zeros(query_boxes.shape[0], dtype=torch.long, device=query_boxes.device))
    iou_matrix = torchvision.ops.box_iou(gt_boxes, query_boxes)  # (objects, queries)
    max_iou, matched_gt_indices = iou_matrix.max(dim=0)
    return max_iou, matched_gt_indices


def _create_rpn_labels(max_iou_rpn, device, anchors_per_img, gt_boxes, positive_iou=RPN_POSITIVE_IOU):
    """Label every anchor: 1 = foreground (IoU > positive_iou), 0 = background (IoU < 0.3), -1 = ignored.

    Each object's best-matching anchor is always foreground, so every object has at least one positive anchor.
    Returns the labels, the masks of labelled and of positive anchors, and the number of labelled anchors.
    """
    rpn_labels = torch.ones_like(max_iou_rpn, dtype=torch.float32, device=device) * -1  # ignore by default
    rpn_labels[max_iou_rpn < 0.3] = 0              # background
    rpn_labels[max_iou_rpn > positive_iou] = 1     # foreground

    # Each object's best-matching anchor is foreground
    if gt_boxes.numel() > 0 and anchors_per_img.numel() > 0:
        iou_matrix = torchvision.ops.box_iou(gt_boxes, anchors_per_img)
        best_anchor_indices = iou_matrix.argmax(dim=1)
        rpn_labels[best_anchor_indices] = 1

    valid_mask_rpn = rpn_labels >= 0                 # labelled anchors (foreground or background)
    pos_mask_rpn = (rpn_labels == 1)
    N_rpn = valid_mask_rpn.sum().float().clamp(min=1.0)
    return rpn_labels, valid_mask_rpn, pos_mask_rpn, N_rpn


def _calculate_rpn_losses(rpn_cls_flat, rpn_reg_flat, rpn_labels, valid_mask_rpn, pos_mask_rpn, N_rpn,
                          anchors_per_img, gt_boxes, matched_gt_indices_rpn, lambda_val, device):
    """RPN loss on 256 sampled anchors: binary cross-entropy on objectness + lambda_val x smooth L1 on the
    offsets of the positive anchors, divided by the number of sampled anchors."""
    # Sample 256 anchors per image, at most half foreground, so background does not dominate
    positive = torch.where(pos_mask_rpn)[0]
    negative = torch.where(rpn_labels == 0)[0]
    positive = positive[torch.randperm(len(positive), device=device)[:128]]
    negative = negative[torch.randperm(len(negative), device=device)[:256 - len(positive)]]
    sampled = torch.cat((positive, negative))
    valid_mask_rpn = sampled
    pos_mask_rpn = positive
    N_rpn = max(len(sampled), 1)

    # Objectness: binary cross-entropy
    if rpn_cls_flat[valid_mask_rpn].numel():
        loss_rpn_cls = F.binary_cross_entropy_with_logits(rpn_cls_flat[valid_mask_rpn], rpn_labels[valid_mask_rpn],
                                                          reduction='sum')
    else:
        loss_rpn_cls = torch.tensor(0.0, device=device)

    # Box offsets of the positive anchors: smooth L1 against the encoded matched object
    if rpn_reg_flat[pos_mask_rpn].numel():
        target_gt_boxes_for_pos_anchors = gt_boxes[matched_gt_indices_rpn[pos_mask_rpn]]
        positive_anchors = anchors_per_img[pos_mask_rpn]
        target_transforms_rpn = calculate_transforms(positive_anchors, target_gt_boxes_for_pos_anchors)
        loss_rpn_reg = F.smooth_l1_loss(rpn_reg_flat[pos_mask_rpn], target_transforms_rpn, reduction='sum')
    else:
        loss_rpn_reg = torch.tensor(0.0, device=device)

    rpn_loss = (loss_rpn_cls + lambda_val * loss_rpn_reg) / N_rpn
    return rpn_loss


def _create_roi_labels(max_iou_roi, proposals_per_img, matched_gt_idx, gt_labels, device):
    """Class target of each sampled RoI: the matched object's class if IoU > 0.5, otherwise background (0).

    Dataset classes 0 = drill, 1 = scissor, 2 = screwdriver become detector classes 1-3 (0 is background).
    Returns the targets, the foreground mask and the number of RoIs.
    """
    N_roi = float(proposals_per_img.size(0))
    if N_roi == 0:
        return (torch.tensor([], dtype=torch.long, device=device), torch.tensor([], dtype=torch.bool, device=device),
                0.0)

    roi_target_labels = torch.zeros(proposals_per_img.size(0), dtype=torch.long, device=device)  # background
    pos_mask_roi = max_iou_roi > 0.5
    if pos_mask_roi.sum() > 0:
        roi_target_labels[pos_mask_roi] = gt_labels[matched_gt_idx[pos_mask_roi]] + 1
    return roi_target_labels, pos_mask_roi, N_roi


def _calculate_roi_losses(roi_cls, roi_reg, roi_target_labels, pos_mask_roi, N_roi, proposals_per_img, gt_boxes,
                          matched_gt_idx, device):
    """RoI head loss: cross-entropy on the class (foreground and background weighted equally) + smooth L1 on the
    true class's box offsets for foreground RoIs, divided by the number of RoIs."""
    if N_roi == 0:
        return torch.tensor(0.0, device=device)

    # Classification: cross-entropy, averaged separately over foreground and background RoIs
    class_losses = F.cross_entropy(roi_cls, roi_target_labels, reduction="none")
    loss_roi_cls = class_losses.mean()
    if pos_mask_roi.any() and (~pos_mask_roi).any():  # foreground and background weighted equally
        loss_roi_cls = 0.5 * (class_losses[pos_mask_roi].mean() + class_losses[~pos_mask_roi].mean())

    # Box refinement: only the offsets predicted for each foreground RoI's true class are trained
    if pos_mask_roi.sum() > 0:
        target_gt_boxes_for_pos_roi = gt_boxes[matched_gt_idx[pos_mask_roi]]
        pred_boxes_for_pos_roi = roi_reg[pos_mask_roi]
        pred_boxes_for_pos_roi = pred_boxes_for_pos_roi.reshape(-1, 3, 4)  # (RoIs, tool classes, 4)
        class_indices = roi_target_labels[pos_mask_roi] - 1
        pred_boxes_for_pos_roi = pred_boxes_for_pos_roi[torch.arange(len(class_indices), device=device),
                                                        class_indices]
        target_transforms_roi = calculate_transforms(proposals_per_img[pos_mask_roi], target_gt_boxes_for_pos_roi,
                                                     BOX_WEIGHTS)
        loss_roi_reg = F.smooth_l1_loss(pred_boxes_for_pos_roi, target_transforms_roi, reduction='sum', beta=BOX_BETA)
    else:
        loss_roi_reg = torch.tensor(0.0, device=device)

    roi_loss = loss_roi_cls + loss_roi_reg / N_roi
    return roi_loss


def compute_unified_loss_per_img(rpn_cls, rpn_reg, roi_cls, roi_reg, anchors_per_img, proposals_per_img, targets, *,
                                 rpn_positive_iou=RPN_POSITIVE_IOU):
    """Total Faster R-CNN loss for one image: RPN loss + RoI head loss.

    rpn_cls, rpn_reg: stage-1 outputs; roi_cls, roi_reg: stage-2 outputs for the sampled RoIs
    (proposals_per_img); targets: the image's ground-truth "boxes" (pixels, xyxy) and "labels".
    """
    device = rpn_cls.device
    gt_boxes = targets["boxes"].to(device)
    gt_labels = targets["labels"].to(device)

    # 1. RPN: label every anchor by its overlap with the objects, then the RPN loss
    rpn_cls_flat, rpn_reg_flat = _prepare_rpn_outputs(rpn_cls, rpn_reg)
    max_iou_rpn, matched_gt_indices_rpn = _match_boxes_to_gt(anchors_per_img, gt_boxes)
    rpn_labels, valid_mask_rpn, pos_mask_rpn, N_rpn = _create_rpn_labels(max_iou_rpn, device, anchors_per_img,
                                                                         gt_boxes, rpn_positive_iou)
    rpn_loss = _calculate_rpn_losses(rpn_cls_flat, rpn_reg_flat, rpn_labels, valid_mask_rpn, pos_mask_rpn, N_rpn,
                                     anchors_per_img, gt_boxes, matched_gt_indices_rpn, RPN_BOX_LAMBDA, device)

    # 2. RoI head: label every sampled RoI by its overlap with the objects, then the RoI loss
    max_iou_roi, matched_gt_idx = _match_boxes_to_gt(proposals_per_img, gt_boxes)
    roi_target_labels, pos_mask_roi, N_roi = _create_roi_labels(max_iou_roi, proposals_per_img, matched_gt_idx,
                                                                gt_labels, device)
    roi_loss = _calculate_roi_losses(roi_cls, roi_reg, roi_target_labels, pos_mask_roi, N_roi, proposals_per_img,
                                     gt_boxes, matched_gt_idx, device)

    total_loss = rpn_loss + roi_loss
    return total_loss


# %% decoding
def decode_boxes(ref_boxes, transforms, image_size=(640, 640), weights=None):
    """Apply offsets (t_x, t_y, t_w, t_h) to reference boxes: the inverse of calculate_transforms.

    Returns (xmin, ymin, xmax, ymax) boxes clipped to the image. tw, th are clamped so exp() stays finite.
    """
    if weights is not None:  # undo the offset scaling of the box coder
        transforms = transforms / transforms.new_tensor(weights)

    # Reference box sizes and centres
    aw = (ref_boxes[:, 2] - ref_boxes[:, 0]).clamp(min=1.0)
    ah = (ref_boxes[:, 3] - ref_boxes[:, 1]).clamp(min=1.0)
    ax = ref_boxes[:, 0] + 0.5 * aw
    ay = ref_boxes[:, 1] + 0.5 * ah

    # Predicted offsets
    tx = transforms[:, 0]
    ty = transforms[:, 1]
    tw = transforms[:, 2].clamp(-4.0, 4.0)
    th = transforms[:, 3].clamp(-4.0, 4.0)

    # New centre and size
    px = ax + aw * tx
    py = ay + ah * ty
    pw = aw * torch.exp(tw)
    ph = ah * torch.exp(th)

    refined_boxes = torch.stack([px - 0.5 * pw, py - 0.5 * ph, px + 0.5 * pw, py + 0.5 * ph], dim=1)

    # Clip to the image
    H, W = image_size
    refined_boxes[:, [0, 2]] = refined_boxes[:, [0, 2]].clamp(0, W)
    refined_boxes[:, [1, 3]] = refined_boxes[:, [1, 3]].clamp(0, H)
    return refined_boxes


def filter_proposals(rpn_cls, rpn_reg, curr_anchors, image_size=(640, 640), *, limit=100, nms_iou=RPN_NMS_IOU,
                     pre_nms=None):
    """The `limit` best proposals after NMS, from the `pre_nms` highest-scoring anchors (default 10 x limit).

    Returns the proposal boxes and their objectness probabilities. No gradients: proposals are inputs to stage 2.
    """
    with torch.no_grad():
        # 1. One score and one offset row per anchor
        scores = rpn_cls[0].permute(1, 2, 0).flatten()
        transforms = rpn_reg[0].permute(1, 2, 0).reshape(-1, 4)

        # 2. Keep the highest-scoring anchors
        pre_nms_topk = min(pre_nms or 10 * limit, scores.numel())
        top_scores, top_idx = scores.topk(pre_nms_topk)
        top_anchors = curr_anchors[top_idx]
        top_transforms = transforms[top_idx]

        # 3. Turn them into boxes and drop invalid ones
        refined_boxes = decode_boxes(top_anchors, top_transforms, image_size)
        valid = (torch.isfinite(refined_boxes).all(dim=1)
                 & (refined_boxes[:, 2] > refined_boxes[:, 0])
                 & (refined_boxes[:, 3] > refined_boxes[:, 1]))
        refined_boxes = refined_boxes[valid]
        top_scores = top_scores[valid]

        # 4. Non-maximum suppression, then the `limit` best
        keep_idx = torchvision.ops.nms(refined_boxes, top_scores, iou_threshold=nms_iou)
        keep_idx = keep_idx[:limit]
        return refined_boxes[keep_idx], top_scores[keep_idx].sigmoid()


def decode_detections(proposals, probabilities, regression, image_size, score_floor, *, limit=100):
    """Final detections from the RoI head: for each tool class, refine the proposals with that class's offsets,
    keep scores >= score_floor and boxes wider and taller than 2 px, and apply class-wise NMS; then the `limit`
    highest scores. Labels are detector classes 1-3."""
    boxes_out, scores_out, labels_out = [], [], []
    for label in range(1, 4):
        scores = probabilities[:, label]
        keep = scores >= score_floor
        boxes = decode_boxes(proposals[keep], regression[keep, label - 1], image_size, BOX_WEIGHTS)
        scores = scores[keep]
        valid = torch.isfinite(boxes).all(1) & ((boxes[:, 2:] - boxes[:, :2]) > 2).all(1)
        boxes, scores = boxes[valid], scores[valid]
        chosen = torchvision.ops.nms(boxes, scores, FINAL_NMS_IOU)
        boxes_out.append(boxes[chosen])
        scores_out.append(scores[chosen])
        labels_out.append(torch.full((len(chosen),), label, dtype=torch.long, device=boxes.device))
    scores = torch.cat(scores_out)
    order = scores.argsort(descending=True)[:limit]
    return {"boxes": torch.cat(boxes_out)[order].cpu(), "labels": torch.cat(labels_out)[order].cpu(),
            "scores": scores[order].cpu()}
