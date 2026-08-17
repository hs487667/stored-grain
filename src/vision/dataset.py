"""Torch datasets over the manifest.

One thing to keep in view while reading the numbers this produces. The corpus
is two datasets with very different acquisition:

* GrainSet maize -- P600 laboratory rig, roughly 600 x 455 px per kernel
* Corn Seeds -- roughly 132 x 132 px per kernel

and ``seed_coat_cracked``, the class the whole project rests on, exists **only**
in Corn Seeds. So a model can score well on that class by learning "this image
is low resolution" rather than "this kernel has a split pericarp". Resizing
everything to a common input size hides the cue but does not remove it: an
upscaled 132 px crop stays visibly softer than a downscaled 600 px one.

That is why :func:`evaluate` reports per-dataset as well as overall figures. If
``seed_coat_cracked`` recall is high while Corn Seeds accuracy overall is much
higher than GrainSet's, the model is reading the acquisition, not the kernel.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.data.manifest import Sample
from src.mapping.classes import CLASSES

#: Fixed ordering so a checkpoint's head means the same thing on reload.
CLASS_NAMES: tuple[str, ...] = tuple(sorted(CLASSES))
CLASS_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def train_transform(size: int = 224):
    """Geometry and colour jitter only.

    No aggressive augmentation: Kumari et al. (2026) report that advanced
    augmentation did not recover their acquisition-shift gap, and a
    heavy-handed transform risks destroying the fine pericarp crack that the
    rarest and most important class is defined by.
    """
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180, fill=0),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def eval_transform(size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


@dataclass
class KernelDataset(Dataset):
    samples: list[Sample]
    transform: object

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        sample = self.samples[i]
        image = Image.open(sample.path).convert("RGB")
        return (
            self.transform(image),
            CLASS_INDEX[sample.class_name],
            0 if sample.dataset == "grainset" else 1,
        )


def class_weights(samples: list[Sample]) -> torch.Tensor:
    """Inverse-frequency weights, so the rare classes are not simply ignored.

    ``sound`` is roughly half the corpus and ``seed_coat_cracked`` about three
    percent of it. Unweighted, the cheapest way to a good loss is to predict
    ``sound`` and accept the rest as noise -- which is exactly the outcome that
    would make the damage measurement useless while the accuracy looked fine.
    """
    counts = torch.zeros(len(CLASS_NAMES))
    for s in samples:
        counts[CLASS_INDEX[s.class_name]] += 1
    weights = torch.where(counts > 0, counts.sum() / (counts * len(CLASS_NAMES)), 0.0)
    return weights
