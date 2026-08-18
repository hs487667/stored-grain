"""Turning what a phone sent into what the models expect.

The localiser is fully convolutional and has no scale invariance: it learned
kernels at one size in pixels and a photograph presenting them at another is a
different problem. Every upload is therefore brought to one working scale
before it reaches the pipeline. The size, and the drift it tolerates, were
measured -- see docs/superpowers/plans/scale-experiment-result.md.
"""

from __future__ import annotations

import io

from PIL import Image, ImageOps, UnidentifiedImageError

#: Long edge every upload is resized to, matching the scale the localiser was
#: trained at. Measured, not guessed.
WORKING_LONG_EDGE_PX = 1391


class NotAnImage(Exception):
    """The upload could not be decoded as an image."""


def decode(data: bytes) -> Image.Image:
    """Decode an upload, honouring EXIF rotation and forcing RGB.

    Phones record orientation in EXIF rather than in the pixels, so a tray
    photographed in portrait arrives sideways unless it is transposed. The
    pipeline indexes three channels, so a greyscale or RGBA upload has to be
    converted here rather than failing deep inside inference.
    """
    if not data:
        raise NotAnImage("the upload was empty")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise NotAnImage(f"could not decode the upload as an image: {exc}") from exc
    return ImageOps.exif_transpose(image).convert("RGB")


def to_working_scale(image: Image.Image) -> Image.Image:
    """Resize so the long edge matches the trained scale, never upscaling.

    Enlarging a small photograph would invent detail that reads as kernel
    texture, so an image already below the working size is left as it is and
    allowed to fail honestly if its kernels are too small.
    """
    long_edge = max(image.size)
    if long_edge <= WORKING_LONG_EDGE_PX:
        return image
    factor = WORKING_LONG_EDGE_PX / long_edge
    size = (round(image.width * factor), round(image.height * factor))
    return image.resize(size, Image.LANCZOS)
