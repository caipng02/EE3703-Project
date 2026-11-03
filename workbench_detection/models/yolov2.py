"""YOLOv2 with a ResNet-18 backbone, built from individual PyTorch layers.

Input: (B, 3, 640, 640) normalized images. Output: raw predictions (B, A, 20, 20, 5 + C) for A = 5
anchors on a 20 x 20 grid (stride 32): per anchor x, y, w, h offsets, objectness and C class scores.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..boxes import nms, wh_iou, xywh_to_xyxy
from ..config import CLASSES, IMAGE_SIZE, SEED


# %% backbone
def conv_bn(cin, cout, k=3, stride=1):
    """Convolution, batch norm and leaky ReLU: the building block of the YOLOv2 layers."""
    return nn.Sequential(nn.Conv2d(cin, cout, k, stride, k // 2, bias=False), nn.BatchNorm2d(cout),
                         nn.LeakyReLU(.1, inplace=True))


class BasicBlock(nn.Module):
    """ResNet basic block: two 3x3 convolutions with a skip connection (1x1 projection when shapes change)."""

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.skip = (nn.Identity() if cin == cout and stride == 1 else
                     nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout)))

    def forward(self, x):
        return F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x))))) + self.skip(x))


# %% detector
class YOLOv2ResNet18(nn.Module):
    """ResNet-18 stages (stem, s1-s4) feed the YOLOv2 head.

    The passthrough layer reorganizes the stride-16 features of s3 into the stride-32 grid
    (pixel_unshuffle) and concatenates them with the deep features, keeping finer detail for small tools.
    """

    def __init__(self, anchors, classes=len(CLASSES)):
        super().__init__()
        self.register_buffer("anchors", anchors.clone().float())  # (A, 2) normalized width, height
        self.classes = classes
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 7, 2, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(3, 2, 1))
        self.s1 = nn.Sequential(BasicBlock(64, 64), BasicBlock(64, 64))
        self.s2 = nn.Sequential(BasicBlock(64, 128, 2), BasicBlock(128, 128))
        self.s3 = nn.Sequential(BasicBlock(128, 256, 2), BasicBlock(256, 256))
        self.s4 = nn.Sequential(BasicBlock(256, 512, 2), BasicBlock(512, 512))
        self.deep = nn.Sequential(conv_bn(512, 1024), conv_bn(1024, 1024))
        self.passthrough = conv_bn(256, 64, 1)
        self.head = nn.Sequential(conv_bn(1280, 1024), nn.Conv2d(1024, len(anchors) * (5 + classes), 1))

    def forward(self, x):
        fine = self.s3(self.s2(self.s1(self.stem(x))))                  # stride 16, 256 channels
        deep = self.deep(self.s4(fine))                                   # stride 32, 1024 channels
        route = F.pixel_unshuffle(self.passthrough(fine), 2)              # stride 32, 256 channels
        raw = self.head(torch.cat((deep, route), 1))
        b, _, h, w = raw.shape
        return raw.view(b, len(self.anchors), 5 + self.classes, h, w).permute(0, 1, 3, 4, 2).contiguous()


# %% initialization
def load_imagenet_backbone(model):
    """Copy ImageNet ResNet-18 weights (torchvision) into the hand-built backbone: stem and s1-s4.

    Only the backbone starts from ImageNet; the YOLOv2 layers (deep, passthrough, head) stay random.
    The layers mirror torchvision's ResNet-18, so the weights map by name and shape.
    """
    from torchvision.models import ResNet18_Weights, resnet18

    source = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).state_dict()
    target = model.state_dict()
    mapped = {}
    for key, value in source.items():
        if key.startswith("fc."):  # the ImageNet classifier is not part of the backbone
            continue
        if key.startswith("conv1."):
            name = "stem.0." + key.removeprefix("conv1.")
        elif key.startswith("bn1."):
            name = "stem.1." + key.removeprefix("bn1.")
        else:  # layer1.0.conv1.weight -> s1.0.c1.weight
            name = "s" + key.removeprefix("layer")
            for old, new in ((".conv1.", ".c1."), (".bn1.", ".b1."), (".conv2.", ".c2."),
                             (".bn2.", ".b2."), (".downsample.", ".skip.")):
                name = name.replace(old, new)
        if name not in target or target[name].shape != value.shape:
            raise ValueError(f"ImageNet weight {key} does not match backbone layer {name}.")
        mapped[name] = value
    model.load_state_dict(mapped, strict=False)
    return len(mapped)


OBJECTNESS_PRIOR = 0.01


def init_objectness_prior(model, prior=OBJECTNESS_PRIOR):
    """Start every anchor's objectness at `prior` instead of 0.5.

    With about 1.5 objects per image and 2000 anchors, a 0.5 start makes the background term about 89%
    of the initial loss and destabilizes the first epochs.
    """
    head = model.head[-1]
    with torch.no_grad():
        head.bias.view(len(model.anchors), 5 + model.classes)[:, 4] = math.log(prior / (1 - prior))


# %% anchors
def cluster_anchors(wh, k=5, seed=SEED):
    """k-means on box width/height with 1 - IoU as the distance and medians as centres (YOLOv2's priors).

    wh: (N, 2) normalized training box sizes. Returns k anchors sorted by area.
    """
    if len(wh) < k:
        raise ValueError("Not enough training boxes to cluster anchors")
    generator = torch.Generator().manual_seed(seed)
    centers = wh[torch.randperm(len(wh), generator=generator)[:k]].clone()
    for _ in range(100):
        assign = wh_iou(wh, centers).argmax(1)
        updated = torch.stack([wh[assign == j].median(0).values if (assign == j).any() else centers[j]
                               for j in range(k)])
        if torch.allclose(updated, centers, atol=1e-6):
            break
        centers = updated
    return centers[centers.prod(1).argsort()].clamp(min=1 / IMAGE_SIZE)


def anchors_from_dataset(dataset):
    """Anchors from the training split's boxes only (validation and test never influence them)."""
    sizes = torch.from_numpy(np.concatenate([boxes[:, 2:] - boxes[:, :2] for boxes in dataset.boxes]))
    return cluster_anchors(sizes / IMAGE_SIZE)


# %% decoding
def decode(raw, anchors):
    """Raw offsets -> normalized centre-format boxes (B, A, H, W, 4).

    x, y: sigmoid offset within the grid cell plus the cell index; w, h: anchor size times exp(offset).
    """
    _, a, h, w, _ = raw.shape
    gy, gx = torch.meshgrid(torch.arange(h, device=raw.device), torch.arange(w, device=raw.device), indexing="ij")
    grid = torch.stack((gx, gy), -1).to(raw.dtype)
    xy = (raw[..., :2].sigmoid() + grid) / raw.new_tensor([w, h])
    wh = raw[..., 2:4].clamp(-8, 8).exp() * anchors[None, :, None, None, :]
    return torch.cat((xy, wh), -1)


def postprocess(raw, anchors, conf=.05, nms_iou=.5, topk=300):
    """Decode, filter by confidence and apply class-wise NMS. Returns normalized xyxy boxes per image.

    Score = objectness x class probability. At most 1000 candidates enter NMS and topk boxes are kept.
    """
    boxes = xywh_to_xyxy(decode(raw, anchors)).clamp(0, 1)
    results = []
    probs = raw[..., 4:5].sigmoid() * raw[..., 5:].softmax(-1)
    for b in range(len(raw)):
        flat = boxes[b].reshape(-1, 4)
        scores, classes = probs[b].reshape(-1, len(CLASSES)).max(1)
        ok = (scores >= conf) & ((flat[:, 2:] - flat[:, :2]).min(1).values > 0)
        flat, scores, classes = flat[ok], scores[ok], classes[ok]
        idx = scores.argsort(descending=True)[:1000]
        flat, scores, classes = flat[idx], scores[idx], classes[idx]
        chosen = []
        for c in range(len(CLASSES)):
            idx = torch.where(classes == c)[0]
            chosen.extend(idx[nms(flat[idx], scores[idx], nms_iou)].tolist())
        idx = torch.tensor(chosen, device=raw.device, dtype=torch.long)
        idx = idx[scores[idx].argsort(descending=True)[:topk]]
        results.append({"boxes": flat[idx].detach().cpu(), "scores": scores[idx].detach().cpu(),
                        "labels": classes[idx].detach().cpu()})
    return results
