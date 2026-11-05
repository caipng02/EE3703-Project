"""Faster R-CNN with a ResNet-18 backbone (Person 2): region proposal network, RoI head, losses and decoding.

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
ANCHOR_PROFILES = {"original": (128, 256, 512), "small": (64, 128, 256, 512),
                   "dense": (32, 64, 128, 256, 512),
                   # half-octave steps (x1.41) from 16 px: the size between two dense sizes fits far better
                   "half_octave": (16, 23, 32, 45, 64, 91, 128, 181, 256, 362, 512)}
# Backbone features the RPN and RoI head use: "c4" = ResNet layer 3 (stride 16, 256 channels); "c5" = layer 4
# (stride 32, 512 channels; Person 2's original model); "c3c4" = layer 3 upsampled and added to layer 2 (stride 8).
FEATURE_STRIDES = {"c3c4": 8, "c4": 16, "c5": 32}
# RoI head box offsets. "plain" (Person 2): raw offsets, smooth L1 with beta 1, weight lambda = 10. "standard"
# (torchvision's Faster R-CNN): offsets scaled by (10, 10, 5, 5) and smooth L1 with beta 1/9, weight 1, so errors of
# a few percent of the box size still give a full-size gradient. The RPN keeps the plain coding in both.
BOX_CODERS = {"plain": {"weights": None, "beta": 1.0, "lambda": 10.0},
              "standard": {"weights": (10.0, 10.0, 5.0, 5.0), "beta": 1 / 9, "lambda": 1.0}}
ASPECT_RATIOS = (0.5, 1.0, 2.0)      # anchor height / width
RPN_POSITIVE_IOU = 0.70             # an anchor overlapping an object above this IoU is foreground
RPN_NMS_IOU = 0.70
FINAL_NMS_IOU = 0.50
TRAIN_PROPOSALS = 300
INFERENCE_PROPOSALS = 600


# %% model
class RPN(nn.Module):

    def __init__(self, in_channels=512, num_anchors=9):
        super().__init__()

        # Intermediate 3x3 convolution
        self.rpn_head = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels,
                kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU()
        )

        # Objectness classification head
        self.cls_logits = nn.Conv2d(
            in_channels, num_anchors,
            kernel_size=1, stride=1
        )

        # Bounding box regression head
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors * 4,
            kernel_size=1, stride=1
        )

    def forward(self, x):
        t = self.rpn_head(x)
        return self.cls_logits(t), self.bbox_pred(t)

class Stage2_network(nn.Module):

    def __init__(self, in_channels=512, num_classes=4, *, stride=32, pool_size=7, fc_layers=1):
        super().__init__()

        # Region of Interest (RoI) Align
        self.roi_align = torchvision.ops.RoIAlign(
            output_size=(pool_size, pool_size),
            spatial_scale=1.0 / stride,
            sampling_ratio=2,
            aligned=True
        )

        # Fully connected layers (Person 2: one; torchvision's Faster R-CNN head: two)
        layers, width = [], in_channels * pool_size ** 2
        for _ in range(fc_layers):
            layers += [nn.Linear(in_features=width, out_features=1024),
                       nn.LeakyReLU(0.01) if stride < 32 else nn.ReLU()]
            width = 1024
        self.fc = nn.Sequential(*layers)

        # Classification head
        self.cls_score = nn.Linear(
            1024, num_classes
        )

        # Bounding box regression head
        self.bbox_head = nn.Linear(
            1024, (num_classes - 1) * 4
        )

    def forward(self, features, proposals):

        pooled = self.roi_align(features, proposals)

        x = pooled.flatten(start_dim=1)

        x = self.fc(x)

        return self.cls_score(x), self.bbox_head(x)

def default_features(anchor_scales):
    """Features of checkpoints saved before `features` existed: the dense anchors meant layer 3, others layer 4."""
    return "c4" if tuple(anchor_scales) == ANCHOR_PROFILES["dense"] else "c5"


class FPNMerge(nn.Module):
    """Layer 3 (stride 16) upsampled and added to a 1x1 projection of layer 2 (stride 8), then a 3x3 smoothing
    convolution: the stride-8 map keeps layer 2's position detail and layer 3's semantics (one FPN level)."""

    def __init__(self, fine_channels=128, coarse_channels=256):
        super().__init__()
        self.lateral = nn.Conv2d(fine_channels, coarse_channels, kernel_size=1)
        self.smooth = nn.Conv2d(coarse_channels, coarse_channels, kernel_size=3, padding=1)

    def forward(self, fine, coarse):
        merged = self.lateral(fine) + F.interpolate(coarse, size=fine.shape[-2:], mode="nearest")
        return self.smooth(merged)


class FasterRCNN_Modular(nn.Module):

    def __init__(self, num_classes, pretrained=True, *, anchor_scales=ANCHOR_PROFILES["original"],
                 aspect_ratios=ASPECT_RATIOS, features=None, pool_size=None, roi_fc_layers=1, roi_box_coder="plain"):
        super().__init__()
        self.features = features or default_features(anchor_scales)
        if self.features not in FEATURE_STRIDES:
            raise ValueError(f"Unknown features {self.features!r}; use one of {sorted(FEATURE_STRIDES)}.")
        # "dense" = Person 2's improved path (finer features, more proposals, balanced losses); c5 is the original.
        self.dense = self.features != "c5"
        stride = FEATURE_STRIDES[self.features]
        self.pool_size = pool_size or (5 if self.dense else 7)
        self.roi_fc_layers = roi_fc_layers
        if roi_box_coder not in BOX_CODERS:
            raise ValueError(f"Unknown roi_box_coder {roi_box_coder!r}; use one of {sorted(BOX_CODERS)}.")
        self.roi_box_coder = roi_box_coder

        # STEP 2.3. CNN Backbone
        backbone_model = resnet18(
            weights=ResNet18_Weights.DEFAULT if pretrained else None
        )

        # Layer 3's stride-16 map (c4) for small tools; layer 4 (c5) for previously trained stride-32
        # checkpoints; or layers 2 + 3 merged at stride 8 (c3c4). Module names keep old checkpoints loadable.
        children = list(backbone_model.children())
        self.neck = None
        if self.features == "c3c4":
            self.backbone = nn.Sequential(*children[:6])   # stem + layers 1-2: stride 8, 128 channels
            self.backbone_top = children[6]                # layer 3: stride 16, 256 channels
            self.neck = FPNMerge(128, 256)
        else:
            self.backbone = nn.Sequential(*children[:7 if self.features == "c4" else 8])
        channels = 512 if self.features == "c5" else 256

        # STEP 2.4. Anchor Generator
        # Use the same anchor count in the generator and both RPN heads.
        self.anchor_generator = AnchorGenerator(
          sizes=(tuple(anchor_scales),),
          aspect_ratios=(tuple(aspect_ratios),)
        )

        # Assemble RPN and Stage 2 detector
        self.rpn = RPN(
            in_channels=channels,
            num_anchors=self.anchor_generator.num_anchors_per_location()[0]
        )

        self.stage2 = Stage2_network(
            in_channels=channels,
            num_classes=num_classes,
            stride=stride,
            pool_size=self.pool_size,
            fc_layers=roi_fc_layers,
        )
        if self.dense:
            for layer in (self.rpn.cls_logits, self.rpn.bbox_pred,
                          self.stage2.cls_score, self.stage2.bbox_head):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.zeros_(layer.bias)

    def pretrained_parts(self):
        """The ResNet-18 layers (ImageNet weights); the neck, RPN and RoI head are trained from scratch."""
        return [self.backbone] + ([self.backbone_top] if self.neck is not None else [])

    def new_parts(self):
        return ([self.neck] if self.neck is not None else []) + [self.rpn, self.stage2]

    def forward(self, images):

        # STEP 2.5. Stage 1: Backbone + RPN forward pass

        # Prepare ImageList for anchor generation
        image_sizes = [
            (img.shape[-2], img.shape[-1])
            for img in images
        ]

        image_list = ImageList(
            torch.stack(images),
            image_sizes
        )

        # 1. Feature extraction
        features = self.backbone(image_list.tensors)
        if self.neck is not None:
            features = self.neck(features, self.backbone_top(features))

        # 2. Anchor generation
        anchors = self.anchor_generator(
            image_list,
            [features]
        )

        # 3. RPN predictions
        rpn_cls, rpn_reg = self.rpn(features)

        return features, anchors, rpn_cls, rpn_reg

    def forward_roi(self, features, proposals):

        # STEP 2.5. Stage 2: RoI Head forward pass
        return self.stage2(features, proposals)


# %% loss
def calculate_transforms(ref_boxes, gt_boxes, weights=None):
    """
    Converts absolute bounding box coordinates (xmin, ymin, xmax, ymax)
    into target transforms (t_x, t_y, t_w, t_h) relative to
    reference boxes (anchors or proposals); `weights` scales the four offsets (BOX_CODERS).
    """

    # 1. Reference box dimensions and centres
    widths = ref_boxes[:, 2] - ref_boxes[:, 0]
    heights = ref_boxes[:, 3] - ref_boxes[:, 1]
    ctr_x = ref_boxes[:, 0] + 0.5 * widths
    ctr_y = ref_boxes[:, 1] + 0.5 * heights

    # 2. Ground truth box dimensions and centres
    gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1]
    gt_ctr_x = gt_boxes[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_boxes[:, 1] + 0.5 * gt_heights

    # 3. Prevent division by zero
    widths = torch.clamp(widths, min=1.0)
    heights = torch.clamp(heights, min=1.0)
    gt_widths = torch.clamp(gt_widths, min=1.0)
    gt_heights = torch.clamp(gt_heights, min=1.0)

    # 4. Calculate regression targets
    t_x = (gt_ctr_x - ctr_x) / widths
    t_y = (gt_ctr_y - ctr_y) / heights
    t_w = torch.log(gt_widths / widths)
    t_h = torch.log(gt_heights / heights)

    # 5. Combine transforms
    target_transforms = torch.stack(
        (t_x, t_y, t_w, t_h), dim=1
    )
    if weights is not None:
        target_transforms = target_transforms * target_transforms.new_tensor(weights)

    return target_transforms

def _prepare_rpn_outputs(rpn_cls_raw, rpn_reg_raw):
    """
    Flattens and reshapes raw RPN outputs to align them with anchor boxes
    generated by AnchorGenerator.
    """

    # Reshape classification scores
    rpn_cls_flat = rpn_cls_raw.permute(0, 2, 3, 1).flatten()

    # Reshape bounding box regression predictions
    rpn_reg_flat = rpn_reg_raw.permute(0, 2, 3, 1).reshape(-1, 4)

    return rpn_cls_flat, rpn_reg_flat

def _match_boxes_to_gt(query_boxes, gt_boxes):
    """
    Calculates IoU between query boxes and ground truth boxes.
    Returns maximum IoU and matched GT indices.
    """

    # Handle empty boxes
    if gt_boxes.numel() == 0 or query_boxes.numel() == 0:
        return (
            torch.zeros(
                query_boxes.shape[0],
                device=query_boxes.device
            ),
            torch.zeros(
                query_boxes.shape[0],
                dtype=torch.long,
                device=query_boxes.device
            )
        )

    # Calculate IoU matrix
    iou_matrix = torchvision.ops.box_iou(
        gt_boxes, query_boxes
    )

    # Find the best ground truth match for each query box
    max_iou, matched_gt_indices = iou_matrix.max(dim=0)

    return max_iou, matched_gt_indices

def _create_rpn_labels(
    max_iou_rpn, device, anchors_per_img, gt_boxes, positive_iou=RPN_POSITIVE_IOU
):

    # Initialize all anchors to -1 (ignore)
    rpn_labels = torch.ones_like(
        max_iou_rpn,
        dtype=torch.float32,
        device=device
    ) * -1

    # Negative anchors (background)
    rpn_labels[max_iou_rpn < 0.3] = 0

    # Positive anchors (foreground)
    rpn_labels[max_iou_rpn > positive_iou] = 1

    # Ensure best-matching anchors are positive
    if gt_boxes.numel() > 0 and anchors_per_img.numel() > 0:

        iou_matrix = torchvision.ops.box_iou(
            gt_boxes, anchors_per_img
        )

        best_anchor_indices = iou_matrix.argmax(dim=1)

        rpn_labels[best_anchor_indices] = 1

    # Valid anchors: positive or negative
    valid_mask_rpn = rpn_labels >= 0

    # Positive anchors only
    pos_mask_rpn = (rpn_labels == 1)

    # Number of valid anchors
    N_rpn = valid_mask_rpn.sum().float().clamp(min=1.0)

    return rpn_labels, valid_mask_rpn, pos_mask_rpn, N_rpn

def _calculate_rpn_losses(
    rpn_cls_flat, rpn_reg_flat, rpn_labels, valid_mask_rpn,
    pos_mask_rpn, N_rpn, anchors_per_img, gt_boxes,
    matched_gt_indices_rpn, lambda_val, device, balanced=False
):

    if balanced:
        positive = torch.where(pos_mask_rpn)[0]
        negative = torch.where(rpn_labels == 0)[0]
        positive = positive[torch.randperm(len(positive), device=device)[:128]]
        negative = negative[torch.randperm(len(negative), device=device)[:256 - len(positive)]]
        sampled = torch.cat((positive, negative))
        valid_mask_rpn = sampled
        pos_mask_rpn = positive
        N_rpn = max(len(sampled), 1)

    # RPN Classification Loss (Binary Cross-Entropy)
    if rpn_cls_flat[valid_mask_rpn].numel():

        loss_rpn_cls = F.binary_cross_entropy_with_logits(
            rpn_cls_flat[valid_mask_rpn],
            rpn_labels[valid_mask_rpn],
            reduction='sum'
        )

    else:
        loss_rpn_cls = torch.tensor(0.0, device=device)

    # RPN Regression Loss (Smooth L1 Loss)
    if rpn_reg_flat[pos_mask_rpn].numel():

        target_gt_boxes_for_pos_anchors = gt_boxes[
            matched_gt_indices_rpn[pos_mask_rpn]
        ]

        positive_anchors = anchors_per_img[pos_mask_rpn]

        target_transforms_rpn = calculate_transforms(
            positive_anchors,
            target_gt_boxes_for_pos_anchors
        )

        loss_rpn_reg = F.smooth_l1_loss(
            rpn_reg_flat[pos_mask_rpn],
            target_transforms_rpn,
            reduction='sum'
        )

    else:
        loss_rpn_reg = torch.tensor(0.0, device=device)

    # Total RPN loss
    rpn_loss = (
        loss_rpn_cls + lambda_val * loss_rpn_reg
    ) / N_rpn

    return rpn_loss

def _create_roi_labels(
    max_iou_roi, proposals_per_img,
    matched_gt_idx, gt_labels, device
):

    # Number of RoI proposals
    N_roi = float(proposals_per_img.size(0))

    if N_roi == 0:
        return (
            torch.tensor([], dtype=torch.long, device=device),
            torch.tensor([], dtype=torch.bool, device=device),
            0.0
        )

    # Initialise all proposals as background
    roi_target_labels = torch.zeros(
        proposals_per_img.size(0),
        dtype=torch.long,
        device=device
    )

    # Positive proposals have IoU > 0.5
    pos_mask_roi = max_iou_roi > 0.5

    # Assign corresponding ground-truth class
    # Dataset: 0=drill, 1=scissor, 2=screwdriver
    # Detector: 0=background, 1=drill, 2=scissor, 3=screwdriver
    if pos_mask_roi.sum() > 0:
        roi_target_labels[pos_mask_roi] = (
            gt_labels[matched_gt_idx[pos_mask_roi]] + 1
        )

    return roi_target_labels, pos_mask_roi, N_roi

def _calculate_roi_losses(
    roi_cls, roi_reg, roi_target_labels, pos_mask_roi,
    N_roi, proposals_per_img, gt_boxes, matched_gt_idx,
    lambda_val, device, balanced=False, box_coder="plain"
):
    coder = BOX_CODERS[box_coder]
    if box_coder != "plain":
        lambda_val = coder["lambda"]

    # If no proposals, return zero loss
    if N_roi == 0:
        return torch.tensor(0.0, device=device)

    # RoI Classification Loss (Cross-Entropy)
    class_losses = F.cross_entropy(roi_cls, roi_target_labels, reduction="none")
    loss_roi_cls = class_losses.mean()
    if balanced and pos_mask_roi.any() and (~pos_mask_roi).any():
        loss_roi_cls = 0.5 * (class_losses[pos_mask_roi].mean() +
                              class_losses[~pos_mask_roi].mean())

    # RoI Regression Loss (Smooth L1)
    if pos_mask_roi.sum() > 0:

        # Ground truth boxes matched to positive proposals
        target_gt_boxes_for_pos_roi = gt_boxes[
            matched_gt_idx[pos_mask_roi]
        ]

        # Get predicted regression deltas
        pred_boxes_for_pos_roi = roi_reg[pos_mask_roi]

        # Select the four regression values for each tool class
        pred_boxes_for_pos_roi = pred_boxes_for_pos_roi.reshape(
            -1, 3, 4
        )

        class_indices = roi_target_labels[pos_mask_roi] - 1

        pred_boxes_for_pos_roi = pred_boxes_for_pos_roi[
            torch.arange(
                len(class_indices),
                device=device
            ),
            class_indices
        ]

        # Corrected tutorial implementation:
        # Convert ground truth boxes into relative transforms
        target_transforms_roi = calculate_transforms(
            proposals_per_img[pos_mask_roi],
            target_gt_boxes_for_pos_roi,
            coder["weights"]
        )

        loss_roi_reg = F.smooth_l1_loss(
            pred_boxes_for_pos_roi,
            target_transforms_roi,
            reduction='sum',
            beta=coder["beta"]
        )

    else:
        loss_roi_reg = torch.tensor(0.0, device=device)

    # Total RoI loss
    roi_loss = loss_roi_cls + lambda_val * loss_roi_reg / N_roi

    return roi_loss

def compute_unified_loss_per_img(
    rpn_cls, rpn_reg, roi_cls, roi_reg,
    anchors_per_img, proposals_per_img,
    targets, lambda_val=10.0, *, balanced_rpn=False, rpn_positive_iou=RPN_POSITIVE_IOU, roi_box_coder="plain"
):

    device = rpn_cls.device

    gt_boxes = targets["boxes"].to(device)
    gt_labels = targets["labels"].to(device)

    # STEP 3.1. Prepare RPN outputs
    rpn_cls_flat, rpn_reg_flat = _prepare_rpn_outputs(
        rpn_cls, rpn_reg
    )

    # STEP 3.2. Match anchors to ground truth
    max_iou_rpn, matched_gt_indices_rpn = _match_boxes_to_gt(
        anchors_per_img, gt_boxes
    )

    # STEP 3.3. Create RPN labels
    rpn_labels, valid_mask_rpn, pos_mask_rpn, N_rpn = (
    _create_rpn_labels(
        max_iou_rpn,
        device,
        anchors_per_img,
        gt_boxes,
        rpn_positive_iou
      )
    )

    # STEP 3.4. Calculate RPN loss
    rpn_loss = _calculate_rpn_losses(
        rpn_cls_flat,
        rpn_reg_flat,
        rpn_labels,
        valid_mask_rpn,
        pos_mask_rpn,
        N_rpn,
        anchors_per_img,
        gt_boxes,
        matched_gt_indices_rpn,
        lambda_val,
        device, balanced=balanced_rpn
    )

    # STEP 3.5. Match RoIs to ground truth
    max_iou_roi, matched_gt_idx = _match_boxes_to_gt(
        proposals_per_img, gt_boxes
    )

    # Create RoI labels (adapted for 3 tool classes)
    roi_target_labels, pos_mask_roi, N_roi = _create_roi_labels(
        max_iou_roi,
        proposals_per_img,
        matched_gt_idx,
        gt_labels,
        device
    )

    # Calculate RoI loss
    roi_loss = _calculate_roi_losses(
        roi_cls,
        roi_reg,
        roi_target_labels,
        pos_mask_roi,
        N_roi,
        proposals_per_img,
        gt_boxes,
        matched_gt_idx,
        lambda_val,
        device, balanced=balanced_rpn, box_coder=roi_box_coder
    )

    # Unified Faster R-CNN loss
    total_loss = rpn_loss + roi_loss

    return total_loss


# %% decoding
def decode_boxes(ref_boxes, transforms, image_size=(640, 640), weights=None):
    if weights is not None:  # undo the offset scaling of the box coder
        transforms = transforms / transforms.new_tensor(weights)

    # Reference box dimensions and centres
    aw = (ref_boxes[:, 2] - ref_boxes[:, 0]).clamp(min=1.0)
    ah = (ref_boxes[:, 3] - ref_boxes[:, 1]).clamp(min=1.0)

    ax = ref_boxes[:, 0] + 0.5 * aw
    ay = ref_boxes[:, 1] + 0.5 * ah

    # Predicted transforms
    tx = transforms[:, 0]
    ty = transforms[:, 1]
    tw = transforms[:, 2].clamp(-4.0, 4.0)
    th = transforms[:, 3].clamp(-4.0, 4.0)

    # Apply transforms
    px = ax + aw * tx
    py = ay + ah * ty
    pw = aw * torch.exp(tw)
    ph = ah * torch.exp(th)

    # Convert to (xmin, ymin, xmax, ymax)
    refined_boxes = torch.stack([
        px - 0.5 * pw,
        py - 0.5 * ph,
        px + 0.5 * pw,
        py + 0.5 * ph
    ], dim=1)

    # --- STEP 4.3. Clip Boundaries ---
    H, W = image_size

    refined_boxes[:, [0, 2]] = refined_boxes[:, [0, 2]].clamp(0, W)
    refined_boxes[:, [1, 3]] = refined_boxes[:, [1, 3]].clamp(0, H)

    return refined_boxes

def filter_proposals(rpn_cls, rpn_reg, curr_anchors, image_size=(640, 640), *, limit=100, nms_iou=RPN_NMS_IOU,
                     pre_nms=None):
    """The `limit` best proposals after NMS, from the `pre_nms` highest-scoring anchors (default 10 x limit)."""

    with torch.no_grad():

        # Flatten RPN predictions
        scores = rpn_cls[0].permute(1, 2, 0).flatten()
        transforms = rpn_reg[0].permute(1, 2, 0).reshape(-1, 4)

        # STEP 4.1. Select top-scoring anchors
        pre_nms_topk = min(pre_nms or 10 * limit, scores.numel())

        top_scores, top_idx = scores.topk(pre_nms_topk)

        top_anchors = curr_anchors[top_idx]
        top_transforms = transforms[top_idx]

        # STEP 4.2. Refine bounding boxes
        refined_boxes = decode_boxes(
            top_anchors,
            top_transforms,
            image_size
        )

        # Remove invalid boxes
        valid = (
            torch.isfinite(refined_boxes).all(dim=1)
            & (refined_boxes[:, 2] > refined_boxes[:, 0])
            & (refined_boxes[:, 3] > refined_boxes[:, 1])
        )

        refined_boxes = refined_boxes[valid]
        top_scores = top_scores[valid]

        # STEP 4.4. Non-Maximum Suppression (NMS)
        keep_idx = torchvision.ops.nms(
            refined_boxes,
            top_scores,
            iou_threshold=nms_iou
        )

        keep_idx = keep_idx[:limit]
        return refined_boxes[keep_idx], top_scores[keep_idx].sigmoid()


def decode_detections(proposals, probabilities, regression, image_size, score_floor, *, limit=20, box_coder="plain"):
    boxes_out, scores_out, labels_out = [], [], []
    for label in range(1, 4):
        scores = probabilities[:, label]
        keep = scores >= score_floor
        boxes = decode_boxes(proposals[keep], regression[keep, label - 1], image_size, BOX_CODERS[box_coder]["weights"])
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
