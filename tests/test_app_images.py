"""Tests for what arrives from a phone camera.

A phone photograph is nothing like the images the models were trained on: tens
of megapixels, arbitrary orientation, and occasionally not an image at all.
"""

import io

import pytest
from PIL import Image

from src.app.images import WORKING_LONG_EDGE_PX, NotAnImage, decode, to_working_scale


def _png_bytes(width, height, colour=(120, 90, 40)):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), colour).save(buffer, format="PNG")
    return buffer.getvalue()


def test_a_photograph_decodes():
    image = decode(_png_bytes(64, 48))
    assert image.size == (64, 48)


def test_something_that_is_not_an_image_is_refused():
    with pytest.raises(NotAnImage):
        decode(b"this is not a photograph")


def test_an_empty_upload_is_refused():
    with pytest.raises(NotAnImage):
        decode(b"")


def test_a_decoded_image_is_rgb_whatever_arrived():
    # Phones send JPEG, some tools send greyscale or RGBA PNGs. The pipeline
    # indexes three channels and would fail late and confusingly otherwise.
    buffer = io.BytesIO()
    Image.new("L", (32, 32), 128).save(buffer, format="PNG")
    assert decode(buffer.getvalue()).mode == "RGB"


def test_a_large_photograph_is_scaled_to_the_working_size():
    scaled = to_working_scale(Image.new("RGB", (8160, 6120)))
    assert max(scaled.size) == WORKING_LONG_EDGE_PX


def test_scaling_preserves_the_aspect_ratio():
    scaled = to_working_scale(Image.new("RGB", (4000, 3000)))
    assert scaled.width / scaled.height == pytest.approx(4000 / 3000, rel=1e-2)


def test_a_portrait_photograph_scales_on_its_long_edge():
    scaled = to_working_scale(Image.new("RGB", (3000, 4000)))
    assert scaled.height == WORKING_LONG_EDGE_PX


def test_a_small_image_is_left_alone():
    # Upscaling invents detail the localiser would read as kernel texture.
    small = Image.new("RGB", (200, 150))
    assert to_working_scale(small).size == (200, 150)
