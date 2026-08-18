"""The photograph with the measurement drawn on it.

Two colours, not nine. The number that matters is the mechanical-damage
percentage, so the overlay answers the question that produces it -- which
kernels counted as damaged -- rather than displaying a class palette nobody
can hold in their head while looking at a tray.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from src.mapping.classes import CLASSES, Admissibility
from src.pipeline import Detections

#: Kernels that entered the damage term.
DAMAGED_OUTLINE = (220, 60, 40)

#: Kernels that did not. Biological deterioration is included here, because it
#: is measured and reported but never reaches the equation.
SOUND_OUTLINE = (60, 190, 120)


def _boundary(mask: np.ndarray) -> np.ndarray:
    """Pixels of the mask that touch something outside it.

    A one-pixel outline rather than a filled tint: the kernel itself has to
    stay visible, since the point of looking at the overlay is to disagree
    with it.
    """
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    interior = (
        padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
    )
    return mask & ~interior


def draw(image: Image.Image, detections: Detections) -> Image.Image:
    """Outline every detected kernel, coloured by whether it counted as damage."""
    pixels = np.array(image.convert("RGB"))

    for label, class_name in zip(detections.labels, detections.predictions):
        mask = detections.instances == label
        if not mask.any():
            continue
        damaged = CLASSES[class_name].admissibility is Admissibility.MECHANICAL
        pixels[_boundary(mask)] = DAMAGED_OUTLINE if damaged else SOUND_OUTLINE

    return Image.fromarray(pixels)
