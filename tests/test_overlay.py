"""Tests for the annotated photograph.

The overlay is what a person actually looks at, so its job is to be faithful
to the measurement rather than attractive: a kernel the pipeline called
damaged must be marked as damaged, in the place the pipeline found it.
"""

import numpy as np
from PIL import Image

from src.app.overlay import DAMAGED_OUTLINE, SOUND_OUTLINE, draw
from src.pipeline import Detections


def _one_kernel_scene(class_name):
    # A 20x20 grey field with a 6x6 instance in the middle.
    instances = np.zeros((20, 20), dtype=int)
    instances[7:13, 7:13] = 1
    image = Image.new("RGB", (20, 20), (128, 128, 128))
    return image, Detections(instances=instances, labels=[1], predictions=[class_name])


def test_the_overlay_keeps_the_photograph_size():
    image, detections = _one_kernel_scene("sound")
    assert draw(image, detections).size == image.size


def test_the_original_photograph_is_not_modified():
    image, detections = _one_kernel_scene("sound")
    before = image.tobytes()
    draw(image, detections)
    assert image.tobytes() == before


def test_a_damaged_kernel_is_outlined_in_the_damage_colour():
    image, detections = _one_kernel_scene("fragment")
    pixels = np.array(draw(image, detections))
    assert (pixels == np.array(DAMAGED_OUTLINE)).all(axis=-1).any()


def test_a_sound_kernel_is_outlined_in_the_sound_colour():
    image, detections = _one_kernel_scene("sound")
    pixels = np.array(draw(image, detections))
    assert (pixels == np.array(SOUND_OUTLINE)).all(axis=-1).any()


def test_the_outline_lands_on_the_kernel_not_the_background():
    image, detections = _one_kernel_scene("fragment")
    pixels = np.array(draw(image, detections))
    marked = (pixels == np.array(DAMAGED_OUTLINE)).all(axis=-1)
    ys, xs = np.where(marked)
    # Every marked pixel sits within one pixel of the 7..12 instance box.
    assert ys.min() >= 6 and ys.max() <= 13
    assert xs.min() >= 6 and xs.max() <= 13


def test_an_empty_scene_returns_the_photograph_unchanged():
    image = Image.new("RGB", (10, 10), (200, 100, 50))
    detections = Detections(
        instances=np.zeros((10, 10), dtype=int), labels=[], predictions=[]
    )
    assert np.array_equal(np.array(draw(image, detections)), np.array(image))
