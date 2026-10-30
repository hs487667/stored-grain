import random

import numpy as np
from PIL import Image

from src.vision.backgrounds import compose_one, scatter


def _tray(tmp_path):
    image = np.zeros((200, 160, 3), np.uint8)
    instances = np.zeros((200, 160), np.int32)
    for i in range(12):
        r, c = divmod(i, 4)
        instances[10 + r * 60 : 40 + r * 60, 10 + c * 38 : 30 + c * 38] = i + 1
    image[instances > 0] = (220, 170, 60)
    Image.fromarray(image).save(tmp_path / "tray.png")
    Image.fromarray(instances.astype(np.uint16)).save(tmp_path / "tray_inst.png")
    Image.fromarray(np.full((300, 300, 3), 90, np.uint8)).save(tmp_path / "cloth.jpg")
    return image, instances


def test_scattered_kernels_keep_their_shape_and_never_overlap(tmp_path):
    image, instances = _tray(tmp_path)
    _, moved = scatter(image.astype(np.float32), instances, random.Random(0))

    labels, areas = np.unique(moved[moved > 0], return_counts=True)
    assert len(labels) >= 6
    # A kernel is 30x20 px; rotation keeps its area, and an overlap would
    # leave one smaller than that.
    assert set(areas) == {600}


def test_a_composite_keeps_every_label_it_shows(tmp_path):
    _tray(tmp_path)
    for seed in range(6):
        scene, instances = compose_one(tmp_path / "tray.png", tmp_path / "cloth.jpg", random.Random(seed))
        assert scene.shape[:2] == instances.shape
        # A hotspot can make the cloth brighter than the kernels, which is the
        # point, but the labelled pixels must still be the yellow ones.
        yellow = scene[..., 0].astype(int) - scene[..., 2]
        assert yellow[instances > 0].mean() > yellow[instances == 0].mean() + 50


def test_rescaling_keeps_the_frame_and_scales_kernel_area(tmp_path):
    from src.vision.backgrounds import rescale

    image, instances = _tray(tmp_path)
    for factor in (0.5, 2.0):
        _, out = rescale(image.astype(np.float32), instances, factor, random.Random(1))
        assert out.shape == instances.shape
        _, areas = np.unique(out[out > 0], return_counts=True)
        # The largest surviving kernel is whole: 600 px times the area factor.
        assert abs(areas.max() - 600 * factor**2) <= 0.1 * 600 * factor**2
