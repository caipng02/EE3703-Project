"""Train and evaluate the group's Faster R-CNN on Person 1's dataset."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.image_list import ImageList

import person4_evaluation as scoring
from workbench_data import ToolDataset, collate, make_train_sampler

ANCHOR_PROFILES = {"original": (128, 256, 512), "small": (64, 128, 256, 512),
                   "dense": (32, 64, 128, 256, 512)}
RPN_NMS_IOU = 0.70
FINAL_NMS_IOU = 0.50
TRAIN_PROPOSALS = 300
INFERENCE_PROPOSALS = 600

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

    def __init__(self, in_channels=512, num_classes=4, *, stride=32, pool_size=7):
        super().__init__()

        # Region of Interest (RoI) Align
        self.roi_align = torchvision.ops.RoIAlign(
            output_size=(pool_size, pool_size),
            spatial_scale=1.0 / stride,
            sampling_ratio=2,
            aligned=True
        )

        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(
                in_features=in_channels * pool_size ** 2,
                out_features=1024
            ),
            nn.LeakyReLU(0.01) if stride == 16 else nn.ReLU()
        )

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

class FasterRCNN_Modular(nn.Module):

    def __init__(self, num_classes, pretrained=True, *, anchor_scales=ANCHOR_PROFILES["original"]):
        super().__init__()
        self.dense = tuple(anchor_scales) == ANCHOR_PROFILES["dense"]

        # STEP 2.3. CNN Backbone
        backbone_model = resnet18(
            weights=ResNet18_Weights.DEFAULT if pretrained else None
        )

        # Use layer3's stride-16 map for small tools, retaining the legacy path
        # so previously trained stride-32 checkpoints can still be evaluated.
        self.backbone = nn.Sequential(*list(backbone_model.children())[:7 if self.dense else 8])
        channels = 256 if self.dense else 512

        # STEP 2.4. Anchor Generator
        # Use the same anchor count in the generator and both RPN heads.
        self.anchor_generator = AnchorGenerator(
          sizes=(tuple(anchor_scales),),
          aspect_ratios=((0.5, 1.0, 2.0),)
        )

        # Assemble RPN and Stage 2 detector
        self.rpn = RPN(
            in_channels=channels,
            num_anchors=self.anchor_generator.num_anchors_per_location()[0]
        )

        self.stage2 = Stage2_network(
            in_channels=channels,
            num_classes=num_classes,
            stride=16 if self.dense else 32,
            pool_size=5 if self.dense else 7,
        )
        if self.dense:
            for layer in (self.rpn.cls_logits, self.rpn.bbox_pred,
                          self.stage2.cls_score, self.stage2.bbox_head):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.zeros_(layer.bias)

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

def calculate_transforms(ref_boxes, gt_boxes):
    """
    Converts absolute bounding box coordinates (xmin, ymin, xmax, ymax)
    into target transforms (t_x, t_y, t_w, t_h) relative to
    reference boxes (anchors or proposals).
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
    max_iou_rpn, device, anchors_per_img, gt_boxes
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
    rpn_labels[max_iou_rpn > 0.7] = 1

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
    lambda_val, device, balanced=False
):

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
            target_gt_boxes_for_pos_roi
        )

        loss_roi_reg = F.smooth_l1_loss(
            pred_boxes_for_pos_roi,
            target_transforms_roi,
            reduction='sum'
        )

    else:
        loss_roi_reg = torch.tensor(0.0, device=device)

    # Total RoI loss
    roi_loss = loss_roi_cls + lambda_val * loss_roi_reg / N_roi

    return roi_loss

def compute_unified_loss_per_img(
    rpn_cls, rpn_reg, roi_cls, roi_reg,
    anchors_per_img, proposals_per_img,
    targets, lambda_val=10.0, *, balanced_rpn=False
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
        gt_boxes
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
        device, balanced=balanced_rpn
    )

    # Unified Faster R-CNN loss
    total_loss = rpn_loss + roi_loss

    return total_loss

def decode_boxes(ref_boxes, transforms, image_size=(640, 640)):

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

def filter_proposals(rpn_cls, rpn_reg, curr_anchors, image_size=(640, 640), *, limit=100):

    with torch.no_grad():

        # Flatten RPN predictions
        scores = rpn_cls[0].permute(1, 2, 0).flatten()
        transforms = rpn_reg[0].permute(1, 2, 0).reshape(-1, 4)

        # STEP 4.1. Select top-scoring anchors
        pre_nms_topk = min(10 * limit, scores.numel())

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
            iou_threshold=RPN_NMS_IOU
        )

        keep_idx = keep_idx[:limit]
        return refined_boxes[keep_idx], top_scores[keep_idx].sigmoid()


def diagnostic_indices(dataset, count):
    """Take a fixed mixture of original and public images containing all classes."""
    first = [i for i, name in enumerate(dataset.names) if name == "orig_016.jpg"]
    chosen = first[:1]
    for origin in ("original", "public"):
        pool = [i for i, source in enumerate(dataset.origins) if source == origin and i not in chosen]
        limit = min(count - len(chosen), count // 2 if origin == "original" else count)
        for class_id in range(3):
            match = next((i for i in pool if class_id in dataset.labels[i]), None)
            if match is not None and limit:
                chosen.append(match)
                pool.remove(match)
                limit -= 1
        chosen.extend(pool[:limit])
    chosen.extend(i for i in range(len(dataset)) if i not in set(chosen))
    return chosen[:count]


def train_batch(model, images, targets, device):
    """Notebook training path with explicit counts from the original RoI sampler."""
    losses, counts = [], []
    for image, target in zip(images, targets):
        image = image.to(device)
        target = {key: target[key].to(device) for key in ("boxes", "labels")}
        features, anchors, logits, deltas = model([image])
        boxes, _ = filter_proposals(logits, deltas, anchors[0], image.shape[-2:],
                                    limit=TRAIN_PROPOSALS if model.dense else 100)
        boxes = torch.cat((boxes, target["boxes"]))
        overlap, _ = _match_boxes_to_gt(boxes, target["boxes"])
        positive = torch.where(overlap > 0.50)[0]
        negative = torch.where(overlap <= 0.50)[0]
        positive = positive[torch.randperm(len(positive), device=device)[:8]]
        negative = negative[torch.randperm(len(negative), device=device)[:32 - len(positive)]]
        sampled = boxes[torch.cat((positive, negative))]
        roi_cls, roi_reg = model.forward_roi(features, [sampled])
        loss = compute_unified_loss_per_img(
            logits, deltas, roi_cls, roi_reg, anchors[0], sampled, target, balanced_rpn=model.dense)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite Faster R-CNN loss.")
        losses.append(loss)
        counts.append((len(positive), len(negative)))
    return torch.stack(losses).mean(), counts


def decode_detections(proposals, probabilities, regression, image_size, score_floor, *, limit=20):
    boxes_out, scores_out, labels_out = [], [], []
    for label in range(1, 4):
        scores = probabilities[:, label]
        keep = scores >= score_floor
        boxes = decode_boxes(proposals[keep], regression[keep, label - 1], image_size)
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


@torch.inference_mode()
def diagnose(model, dataset, indices, gt, device, score_floor, *, verbose=True):
    model.eval()
    rows, targets, predictions = [], [], []
    for index in indices:
        image, target = dataset[index]
        size = image.shape[-2:]
        features, anchors, logits, deltas = model([image.to(device)])
        proposals, objectness = filter_proposals(
            logits, deltas, anchors[0], size, limit=INFERENCE_PROPOSALS if model.dense else 100)
        roi_cls, roi_reg = model.forward_roi(features, [proposals])
        probs = roi_cls.softmax(1)
        actual_boxes = target["boxes"].to(device)
        actual_labels = target["labels"].to(device) + 1
        if len(proposals):
            best_iou = torchvision.ops.box_iou(actual_boxes, proposals).amax(1)
            best_iou_100 = torchvision.ops.box_iou(actual_boxes, proposals[:100]).amax(1)
        else:
            best_iou = torch.zeros(len(actual_boxes), device=device)
            best_iou_100 = best_iou
        # Check whether the classifier sees zero features after its fully connected ReLU.
        exact_features = []
        handle = model.stage2.fc.register_forward_hook(
            lambda _module, _inputs, output: exact_features.append(output))
        try:
            exact_logits, _ = model.forward_roi(features, [actual_boxes])
        finally:
            handle.remove()
        hidden = exact_features[0]
        exact = exact_logits.softmax(1)
        bias_only = torch.isclose(
            exact_logits, model.stage2.cls_score.bias[None], atol=1e-6, rtol=1e-5
        ).all(1)
        index_per_box = torch.arange(len(actual_labels), device=device)
        # Discount RoI predictions attached to proposals with weak RPN support.
        scores = probs * objectness[:, None].sqrt() if model.dense else probs
        output = decode_detections(proposals, scores, roi_reg.reshape(-1, 3, 4), size, score_floor,
                                   limit=100 if model.dense else 20)
        row = {
            "image": target["file_name"], "proposals": len(proposals),
            "max_foreground_score": probs[:, 1:].max().item() if len(probs) else None,
            "detections_by_cutoff": {str(c): int((output["scores"] >= c).sum())
                                     for c in (score_floor, .05, .20) if c >= score_floor},
            "best_proposal_iou_per_gt": best_iou.tolist(),
            "best_top100_proposal_iou_per_gt": best_iou_100.tolist(),
            "gt_class": actual_labels.tolist(),
            "roi_argmax_on_gt": exact.argmax(1).tolist(),
            "correct_class_score_on_gt": exact[index_per_box, actual_labels].tolist(),
            "background_score_on_gt": exact[:, 0].tolist(),
            "active_roi_features_on_gt": (hidden != 0).sum(1).tolist(),
            "roi_logits_equal_classifier_bias": bias_only.tolist(),
        }
        if verbose:
            print(json.dumps(row), flush=True)
        rows.append(row)
        targets.append(target)
        predictions.append(output)
    export = scoring.predictions_from_outputs(
        targets, predictions, gt, "Faster R-CNN", label_mapping={1: 0, 2: 1, 3: 2})
    return rows, export


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("checkpoint", "overfit", "reproduce"))
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--anchor-profile", choices=tuple(ANCHOR_PROFILES),
                        help="Training defaults to dense; checkpoint mode reads the saved scales.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--results", type=Path, help="Separate location for metrics and figures.")
    parser.add_argument("--images", type=int, help="Default: all validation images or 12 for overfit.")
    parser.add_argument("--epochs", type=int, help="Default: 10 for reproduce or 25 for overfit.")
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--patience", type=int, help="Stop after this many epochs without a validation AP50 gain.")
    parser.add_argument("--device", help="Defaults to CUDA when available.")
    parser.add_argument("--score-floor", type=float, default=0.001)
    parser.add_argument("--random-init", action="store_true", help="Skip pretrained download for a wiring check.")
    args = parser.parse_args(argv)
    if args.mode == "checkpoint" and (not args.checkpoint or not args.checkpoint.is_file()):
        parser.error("Checkpoint mode requires an existing .pth file.")
    if args.mode == "checkpoint" and args.anchor_profile is not None:
        parser.error("Checkpoint mode uses the anchor scales saved with the weights.")
    if args.mode != "checkpoint" and args.checkpoint:
        parser.error("Training modes start a fresh model.")
    if args.epochs is None:
        args.epochs = 25 if args.mode == "overfit" else 10
    if (args.images is not None and args.images < 1) or args.epochs < 1 or not 0 <= args.score_floor <= .20 or args.lr <= 0:
        parser.error("Provide positive images, epochs and learning rate, and a score floor from 0 to 0.20.")
    if args.batch < 1 or args.weight_decay < 0 or (args.patience is not None and args.patience < 1):
        parser.error("Use positive batch size and patience, and nonnegative weight decay.")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use a fresh output folder.")
    import random
    import numpy as np
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    report = args.results or args.output
    if args.results and report.exists() and any(report.iterdir()):
        parser.error("Use a fresh results folder.")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this Python environment.")
    if device.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    overfit = args.mode == "overfit"
    split = "train" if overfit else "valid"
    dataset = ToolDataset(args.dataset_root, split, augment=False)
    indices = diagnostic_indices(dataset, args.images or (12 if overfit else len(dataset)))
    names = {dataset.names[i] for i in indices}
    gt = scoring.load_person1_ground_truth(args.dataset_root, split)
    gt["images"] = [row for row in gt["images"] if row["image_id"] in names]
    if overfit and {label for row in gt["images"] for label in row["labels"]} != {0, 1, 2}:
        parser.error("Increase --images until the subset includes all three classes.")
    checkpoint = None
    if args.mode == "checkpoint":
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        # Checkpoints from the old notebook did not save their original scales.
        anchor_scales = tuple(checkpoint.get("anchor_scales", ANCHOR_PROFILES["original"]))
        if anchor_scales not in ANCHOR_PROFILES.values():
            raise ValueError(f"Unsupported checkpoint anchor scales: {anchor_scales}")
    else:
        anchor_scales = ANCHOR_PROFILES[args.anchor_profile or "dense"]
    model = FasterRCNN_Modular(4, pretrained=args.mode != "checkpoint" and not args.random_init,
                              anchor_scales=anchor_scales).to(device)
    print(f"Anchor scales: {anchor_scales} | anchors per location: "
          f"{model.anchor_generator.num_anchors_per_location()[0]}", flush=True)
    print(f"Device: {device} | mode: {args.mode} | split: {split} | images: {len(indices)}", flush=True)
    history = []
    config = scoring.EvaluationConfig(ap_backend="coco", ap_score_floor=args.score_floor,
                                      operating_confidence=max(args.score_floor, .05))
    checkpoint_path = args.output / "best.pth"
    if args.mode != "checkpoint":
        if overfit:
            loader = DataLoader(Subset(dataset, indices), batch_size=args.batch, shuffle=True, collate_fn=collate)
        else:
            training = ToolDataset(args.dataset_root, "train", augment=True, mosaic_p=0.0)
            loader = DataLoader(training, batch_size=args.batch, sampler=make_train_sampler(training),
                                collate_fn=collate)
        print(f"Training images: {len(loader.dataset)} | epochs: {args.epochs} | lr: {args.lr:g}", flush=True)
        parameters = model.parameters()
        if model.dense:
            parameters = [
                {"params": model.backbone.parameters()},
                {"params": model.rpn.parameters(), "lr": 4 * args.lr},
                {"params": model.stage2.parameters(), "lr": 4 * args.lr},
            ]
        optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=.9, weight_decay=args.weight_decay)
        args.output.mkdir(parents=True, exist_ok=True)
        best_ap = -1.
        best_epoch = 0
        for epoch in range(1, args.epochs + 1):
            model.train()
            if model.dense:
                # Each forward processes one image. Keep pretrained BN statistics stable.
                for layer in model.backbone.modules():
                    if isinstance(layer, nn.BatchNorm2d):
                        layer.eval()
            total, positive, negative = 0., 0, 0
            for images, targets in loader:
                optimizer.zero_grad(set_to_none=True)
                loss, counts = train_batch(model, images, targets, device)
                loss.backward()
                if model.dense:
                    nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                optimizer.step()
                total += loss.item() * len(images)
                positive += sum(p for p, _ in counts)
                negative += sum(n for _, n in counts)
            row = {"epoch": epoch, "train_loss": total / len(loader.dataset),
                   "sampled_foreground": positive, "sampled_background": negative}
            if not overfit:
                _, export = diagnose(model, dataset, indices, gt, device, args.score_floor, verbose=False)
                row["val_map50"] = scoring.evaluate_model(gt, export, config)["map50"]
            print(json.dumps(row), flush=True)
            history.append(row)
            if (overfit and epoch == args.epochs) or (not overfit and row["val_map50"] > best_ap):
                if not overfit:
                    best_ap = row["val_map50"]
                best_epoch = epoch
                torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                            "anchor_scales": anchor_scales, "random_init": args.random_init}, checkpoint_path)
            if not overfit and args.patience and epoch - best_epoch >= args.patience:
                break
        best = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(best["model_state_dict"])
        provenance = {"checkpoint": str(checkpoint_path.resolve()), "epoch": best["epoch"],
                      "pretrained": not args.random_init, "learning_rate": args.lr, "seed": 42,
                      "head_learning_rate": 4 * args.lr if model.dense else args.lr,
                      "batch_size": args.batch, "weight_decay": args.weight_decay,
                      "train_images": len(loader.dataset),
                      "scope": "same training images for fit and scoring" if overfit
                      else "full training split with augmentation; separate validation diagnostic"}
        print(f"Checkpoint saved: {checkpoint_path.resolve()}", flush=True)
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
        provenance = {"checkpoint": str(args.checkpoint.resolve()), "epoch": checkpoint.get("epoch")}
    provenance["anchor_scales"] = anchor_scales
    provenance["proposal_limit"] = INFERENCE_PROPOSALS if model.dense else 100
    provenance["rpn_objectness_exponent"] = 0.5 if model.dense else 0.
    print("Run:", json.dumps(provenance), flush=True)
    print("Diagnostic RoI class IDs: 0=background, 1=drill, 2=scissor, 3=screwdriver", flush=True)
    rows, export = diagnose(model, dataset, indices, gt, device, args.score_floor, verbose=False)
    export["metadata"].update(
        nms_iou=RPN_NMS_IOU, nms_stage="rpn", nms_rule="class_aware",
        final_nms_iou=FINAL_NMS_IOU, export_score_floor=args.score_floor,
        checkpoint=provenance["checkpoint"],
    )
    total = sum(len(row["gt_class"]) for row in rows)
    covered = sum(sum(iou >= .50 for iou in row["best_proposal_iou_per_gt"]) for row in rows)
    correct = sum(sum(a == b for a, b in zip(row["gt_class"], row["roi_argmax_on_gt"])) for row in rows)
    background = sum(row["roi_argmax_on_gt"].count(0) for row in rows)
    summary = {"objects": total, "rpn_covered_at_iou_0.50": covered,
               "roi_correct_on_exact_boxes": correct, "roi_background_on_exact_boxes": background}
    summary["rpn_top100_covered_at_iou_0.50"] = sum(
        sum(iou >= .50 for iou in row["best_top100_proposal_iou_per_gt"]) for row in rows)
    summary["roi_zero_features_on_exact_boxes"] = sum(
        row["active_roi_features_on_gt"].count(0) for row in rows)
    summary["roi_bias_only_on_exact_boxes"] = sum(
        sum(row["roi_logits_equal_classifier_bias"]) for row in rows)
    per_class = {}
    for class_id, name in enumerate(("drill", "scissor", "screwdriver"), start=1):
        objects = [(iou, top100_iou, predicted) for row in rows for label, iou, top100_iou, predicted in
                   zip(row["gt_class"], row["best_proposal_iou_per_gt"],
                       row["best_top100_proposal_iou_per_gt"], row["roi_argmax_on_gt"])
                   if label == class_id]
        per_class[name] = {"objects": len(objects),
                           "rpn_covered_at_iou_0.50": sum(iou >= .50 for iou, _, _ in objects),
                           "rpn_top100_covered_at_iou_0.50": sum(iou >= .50 for _, iou, _ in objects),
                           "roi_correct_on_exact_boxes": sum(predicted == class_id for _, _, predicted in objects)}
    summary["per_class"] = per_class
    print("Pipeline summary:", json.dumps(summary), flush=True)
    scoring.save_json(args.output / "predictions.json", export)
    scoring.save_json(args.output / "diagnostics.json",
                      {"run": provenance, "summary": summary, "history": history, "images": rows})
    software_check = overfit or args.random_init or bool(checkpoint and checkpoint.get("random_init"))
    scoring.run_evaluation(
        gt, [export], args.output, config,
        split=split, image_root=args.dataset_root / split / "images",
        evaluation_purpose="software_check" if software_check else "model_evaluation",
        project_protocol=not software_check and len(indices) == len(dataset),
        results_dir=args.results,
    )
    if history:
        scoring.plot_training_history(history, report, f"Faster R-CNN {args.mode}")


if __name__ == "__main__":
    main()
