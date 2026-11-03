"""Training augmentation on the 640 x 640 canvas (proposal section 2.5).

Geometric changes move the boxes with the image; photometric changes leave them in place. A box must stay at
least 30% visible after any change (at most 70% hidden), otherwise the change is retried or skipped.
Applied to training images only; validation and test images are never augmented.
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from ..config import IMAGE_SIZE, LETTERBOX_FILL

# %% settings
SCISSOR = 1                    # class ID; scissors get their own scale range and partial occlusion
MIN_VISIBLE = 0.3              # a box must stay at least 30% visible
MIN_SIDE = 4.0                 # and keep both sides at least 4 pixels
GEOMETRIC_TRIES = 5
MOSAIC_TRIES = 10
AUGMENTATION = {
    "horizontal_flip_p": 0.5,
    "rotate_deg": 15.0,
    "translate_fraction": 0.08,
    "scale_range": [0.8, 1.2],
    "brightness_contrast": 0.2,
    "gamma_range": [0.8, 1.25],
    "hue_shift_deg": 8,
    "saturation_range": [0.8, 1.2],
    "blur_p": 0.15,
    "noise_p": 0.15,
    "noise_sigma": 6.0,
    "cutout_p": 0.3,
    "cutout_fraction": 0.2,
    "scissor_scale_range": [0.7, 1.3],
    "scissor_occlusion_p": 0.5,
    "scissor_occlusion_area": [0.1, 0.3],
}


# %% geometric
def _visible(boxes, reference_area, bounds=(0, 0, IMAGE_SIZE, IMAGE_SIZE)):
    """Return boxes clipped to bounds and a mask of those that stay usable."""
    clipped = boxes.copy()
    clipped[:, [0, 2]] = clipped[:, [0, 2]].clip(bounds[0], bounds[2])
    clipped[:, [1, 3]] = clipped[:, [1, 3]].clip(bounds[1], bounds[3])
    width, height = clipped[:, 2] - clipped[:, 0], clipped[:, 3] - clipped[:, 1]
    return clipped, (width >= MIN_SIDE) & (height >= MIN_SIDE) & (width * height >= MIN_VISIBLE * reference_area)


def _warp_boxes(boxes, matrix):
    """Apply an affine matrix to the four corners of each box and take their bounding box."""
    corners = np.ones((len(boxes), 4, 3))
    corners[:, 0, :2] = boxes[:, [0, 1]]
    corners[:, 1, :2] = boxes[:, [2, 1]]
    corners[:, 2, :2] = boxes[:, [2, 3]]
    corners[:, 3, :2] = boxes[:, [0, 3]]
    moved = corners @ matrix.T
    return np.concatenate([moved.min(axis=1), moved.max(axis=1)], axis=1)


def geometric(image, boxes, labels, rng):
    """Flip, then try up to 5 random rotation/scale/shift combinations that keep every box usable.

    If none works, only the flip is kept.
    """
    if rng.random() < AUGMENTATION["horizontal_flip_p"]:
        image = image[:, ::-1].copy()
        if len(boxes):
            boxes = boxes.copy()
            boxes[:, [0, 2]] = IMAGE_SIZE - boxes[:, [2, 0]]
    scale_range = AUGMENTATION["scissor_scale_range"] if (labels == SCISSOR).any() else AUGMENTATION["scale_range"]
    shift = AUGMENTATION["translate_fraction"] * IMAGE_SIZE
    for _ in range(GEOMETRIC_TRIES):
        matrix = cv2.getRotationMatrix2D((IMAGE_SIZE / 2, IMAGE_SIZE / 2),
                                         rng.uniform(-AUGMENTATION["rotate_deg"], AUGMENTATION["rotate_deg"]),
                                         rng.uniform(*scale_range))
        matrix[:, 2] += (rng.uniform(-shift, shift), rng.uniform(-shift, shift))
        if not len(boxes):
            return cv2.warpAffine(image, matrix, (IMAGE_SIZE, IMAGE_SIZE), flags=cv2.INTER_LINEAR,
                                  borderValue=LETTERBOX_FILL), boxes, labels
        moved = _warp_boxes(boxes, matrix)
        clipped, usable = _visible(moved, (moved[:, 2] - moved[:, 0]) * (moved[:, 3] - moved[:, 1]))
        if usable.all():
            return cv2.warpAffine(image, matrix, (IMAGE_SIZE, IMAGE_SIZE), flags=cv2.INTER_LINEAR,
                                  borderValue=LETTERBOX_FILL), clipped.astype(np.float32), labels
    return image, boxes, labels


# %% photometric
def photometric(image, rng):
    """Brightness, contrast, gamma, hue, saturation, blur and sensor noise. Boxes do not move."""
    limit = AUGMENTATION["brightness_contrast"]
    output = image.astype(np.float32) * rng.uniform(1 - limit, 1 + limit) + 255 * rng.uniform(-limit, limit) / 2
    output = 255 * (output.clip(0, 255) / 255) ** rng.uniform(*AUGMENTATION["gamma_range"])
    hsv = cv2.cvtColor(output.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-AUGMENTATION["hue_shift_deg"], AUGMENTATION["hue_shift_deg"]) / 2) % 180
    hsv[..., 1] = (hsv[..., 1] * rng.uniform(*AUGMENTATION["saturation_range"])).clip(0, 255)
    output = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    if rng.random() < AUGMENTATION["blur_p"]:
        output = cv2.GaussianBlur(output, (5, 5), 0)
    if rng.random() < AUGMENTATION["noise_p"]:
        noise = np.random.default_rng(rng.randrange(2**32)).normal(0, AUGMENTATION["noise_sigma"], output.shape)
        output = (output + noise).clip(0, 255).astype(np.uint8)
    return output


# %% occlusion
def _try_patch(image, hidden, boxes, x, y, w, h):
    """Grey out a rectangle unless the total hidden area of any box would pass the limit."""
    trial = hidden.copy()
    trial[y:y + h, x:x + w] = True
    for x1, y1, x2, y2 in boxes.astype(int):
        area = trial[y1:y2, x1:x2]
        if area.size and area.mean() > 1 - MIN_VISIBLE:
            return False
    hidden[y:y + h, x:x + w] = True
    image[y:y + h, x:x + w] = LETTERBOX_FILL[0]
    return True


def scissor_occlusion(image, hidden, boxes, labels, rng):
    """Partly cover one scissor box, since thin blades and handles are often occluded."""
    scissors = boxes[labels == SCISSOR]
    if not len(scissors) or rng.random() >= AUGMENTATION["scissor_occlusion_p"]:
        return
    x1, y1, x2, y2 = scissors[rng.randrange(len(scissors))]
    share = rng.uniform(*AUGMENTATION["scissor_occlusion_area"])
    w, h = max(int((x2 - x1) * share ** 0.5), 1), max(int((y2 - y1) * share ** 0.5), 1)
    x = int(rng.uniform(x1, max(x2 - w, x1)))
    y = int(rng.uniform(y1, max(y2 - h, y1)))
    _try_patch(image, hidden, boxes, x, y, min(w, IMAGE_SIZE - x), min(h, IMAGE_SIZE - y))


def cutout(image, hidden, boxes, rng):
    """A grey square at a random place, skipped if it would hide too much of a box."""
    if rng.random() >= AUGMENTATION["cutout_p"]:
        return
    side = int(IMAGE_SIZE * AUGMENTATION["cutout_fraction"] * rng.uniform(0.5, 1))
    _try_patch(image, hidden, boxes, rng.randrange(IMAGE_SIZE - side), rng.randrange(IMAGE_SIZE - side), side, side)


# %% pipelines
def augment(image, boxes, labels, rng):
    """The full training augmentation for one image."""
    image, boxes, labels = geometric(image, boxes, labels, rng)
    image = photometric(image, rng).copy()
    hidden = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
    scissor_occlusion(image, hidden, boxes, labels, rng)
    cutout(image, hidden, boxes, rng)
    return image, boxes, labels


def light_augment(image, boxes, labels, rng):
    """A lighter alternative: horizontal flip, brightness and contrast only."""
    photo = Image.fromarray(image)
    if rng.random() < 0.5:
        photo = ImageOps.mirror(photo)
        boxes = boxes.copy()
        boxes[:, [0, 2]] = image.shape[1] - boxes[:, [2, 0]]
    photo = ImageEnhance.Brightness(photo).enhance(rng.uniform(0.8, 1.2))
    photo = ImageEnhance.Contrast(photo).enhance(rng.uniform(0.8, 1.2))
    return np.asarray(photo), boxes, labels


AUGMENTATIONS = {"full": augment, "light": light_augment}


# %% mosaic
def mosaic(samples, rng):
    """Join four training images around a random centre; each quadrant keeps only boxes that stay usable."""
    cx, cy = rng.randint(IMAGE_SIZE // 4, 3 * IMAGE_SIZE // 4), rng.randint(IMAGE_SIZE // 4, 3 * IMAGE_SIZE // 4)
    canvas = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), LETTERBOX_FILL[0], dtype=np.uint8)
    regions = [(0, 0, cx, cy), (cx, 0, IMAGE_SIZE, cy), (0, cy, cx, IMAGE_SIZE), (cx, cy, IMAGE_SIZE, IMAGE_SIZE)]
    all_boxes, all_labels = [np.zeros((0, 4), np.float32)], [np.zeros(0, np.int64)]
    for (x1, y1, x2, y2), (image, boxes, labels) in zip(regions, samples):
        width, height = x2 - x1, y2 - y1
        best = None
        # Choose the crop that leaves the fewest cut-off remnants without a label.
        for _ in range(MOSAIC_TRIES):
            offset_x, offset_y = rng.randint(0, IMAGE_SIZE - width), rng.randint(0, IMAGE_SIZE - height)
            moved = boxes.copy()
            moved[:, [0, 2]] += x1 - offset_x
            moved[:, [1, 3]] += y1 - offset_y
            clipped, usable = _visible(moved, (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), (x1, y1, x2, y2))
            touching = (moved[:, 2] > x1) & (moved[:, 0] < x2) & (moved[:, 3] > y1) & (moved[:, 1] < y2)
            remnants = int((touching & ~usable).sum())
            score = (remnants, -int(usable.sum()))
            if best is None or score < best[0]:
                best = (score, offset_x, offset_y, clipped[usable], labels[usable])
        _, offset_x, offset_y, kept, kept_labels = best
        canvas[y1:y2, x1:x2] = image[offset_y:offset_y + height, offset_x:offset_x + width]
        all_boxes.append(kept.astype(np.float32))
        all_labels.append(kept_labels)
    return canvas, np.concatenate(all_boxes), np.concatenate(all_labels)
