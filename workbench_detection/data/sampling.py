"""Class-aware sampling (proposal section 2.4) and original-photo weighting (section 4.3)."""

import numpy as np
from torch.utils.data import WeightedRandomSampler

from ..config import CLASSES
from .dataset import TRAINING_SPLITS

# %% sampling
MAX_CLASS_WEIGHT = 2.0  # minority classes are upweighted by at most 2x


def image_weights(dataset, original_weight=1.0):
    """Per-image sampling weights computed from training boxes only.

    Original photos are weighted first; class weights are then the inverse box frequency under those
    weights, capped at 2x, and each image takes the weight of its rarest class.
    """
    if dataset.split not in TRAINING_SPLITS:
        raise ValueError("Sampling weights are computed from training data only.")
    base = [original_weight if origin == "original" else 1.0 for origin in dataset.origins]
    counts = np.zeros(len(CLASSES))
    for weight, labels in zip(base, dataset.labels):
        counts += weight * np.bincount(labels, minlength=len(CLASSES))
    class_weight = np.minimum(counts.max() / counts, MAX_CLASS_WEIGHT)
    return [weight * (float(class_weight[labels].max()) if len(labels) else 1.0)
            for weight, labels in zip(base, dataset.labels)]


def expected_box_counts(dataset, weights):
    """Boxes per class one epoch is expected to contain under the given weights."""
    scale = len(weights) / sum(weights)
    counts = np.zeros(len(CLASSES))
    for weight, labels in zip(weights, dataset.labels):
        counts += scale * weight * np.bincount(labels, minlength=len(CLASSES))
    return counts


def make_train_sampler(dataset, original_weight=1.0, generator=None):
    """A weighted random sampler over the training images (image_weights), drawing one epoch's worth with replacement."""
    weights = image_weights(dataset, original_weight)
    dataset.sample_weights = weights
    return WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True, generator=generator)
