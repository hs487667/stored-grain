"""Tests for scene synthesis.

A synthetic scene is training data and a test fixture at once, so a scene that
is quietly wrong corrupts both the localiser and the evidence that the pipeline
works. These pin the ways that can happen.
"""

import numpy as np
import pytest

from src.mapping.classes import CLASSES
from src.vision.synth import (
    BACKGROUND,
    MECHANICAL,
    Cutout,
    canvas_for,
    compose,
    pool_for_damage,
)


def _cutout(class_name="sound", size=(40, 20), value=200) -> Cutout:
    h, w = size
    return Cutout(
        rgb=np.full((h, w, 3), value, dtype=np.uint8),
        alpha=np.ones((h, w), dtype=bool),
        class_name=class_name,
        relative_mass=CLASSES[class_name].relative_mass or 1.0,
        source=None,
    )


def _pool():
    return [_cutout("sound") for _ in range(20)] + [
        _cutout("fragment", value=120) for _ in range(20)
    ]


# --- The scene is what it claims to be ------------------------------------

def test_scene_truth_matches_what_was_placed():
    scene = compose([_cutout("sound"), _cutout("fragment")] * 20, kernel_px=30, seed=1)
    counts = {}
    for name in scene.class_names:
        counts[name] = counts.get(name, 0) + 1
    assert sum(counts.values()) == scene.placed
    assert scene.truth.kernels_counted == scene.placed


def test_instance_map_has_one_id_per_placed_kernel():
    scene = compose(_pool(), n_kernels=30, kernel_px=30, seed=2)
    ids = set(np.unique(scene.instances)) - {0}
    # Every id must be reachable; a kernel fully buried by later ones would
    # vanish from the map while still being counted, which would silently
    # break the correspondence between the mask and the truth.
    assert len(ids) == scene.placed
    assert max(ids) == scene.placed


def test_background_is_the_protocol_card_not_pure_black():
    scene = compose(_pool(), n_kernels=5, kernel_px=30, seed=3)
    corner = scene.image[0, 0]
    assert tuple(corner) == BACKGROUND
    assert tuple(corner) != (0, 0, 0)


def test_instances_and_image_agree_on_where_kernels_are():
    scene = compose(_pool(), n_kernels=20, kernel_px=30, seed=4)
    painted = (scene.image != np.array(BACKGROUND)).any(axis=2)
    assert painted[scene.instances > 0].all()


# --- Damage targeting ------------------------------------------------------

def test_damage_target_is_respected_by_count():
    pool = _pool()
    for target in (0, 10, 25, 50):
        sel = pool_for_damage(pool, target, 100, seed=target)
        n_damaged = sum(1 for c in sel if c.class_name in MECHANICAL)
        assert n_damaged == round(100 * target / 100.0)


def test_fragments_pull_the_mass_percentage_below_the_count():
    sel = pool_for_damage(_pool(), 40, 100, seed=5)
    scene = compose(sel, kernel_px=30, seed=5)
    truth = scene.truth
    assert truth.mechanical_damage_mass_pct < truth.mechanical_damage_count_pct


def test_zero_damage_scene_has_no_mechanical_damage():
    sel = pool_for_damage(_pool(), 0, 50, seed=6)
    assert compose(sel, kernel_px=30, seed=6).truth.mechanical_damage_mass_pct == 0.0


def test_damage_target_needs_a_damaged_population():
    with pytest.raises(ValueError, match="no mechanically damaged"):
        pool_for_damage([_cutout("sound")], 10, 50)


def test_damage_target_out_of_range_is_rejected():
    with pytest.raises(ValueError, match="damage must be in"):
        pool_for_damage(_pool(), 101, 50)


# --- Determinism and inputs ------------------------------------------------

def test_same_seed_gives_the_same_scene():
    a = compose(_pool(), n_kernels=25, kernel_px=30, seed=7)
    b = compose(_pool(), n_kernels=25, kernel_px=30, seed=7)
    assert np.array_equal(a.image, b.image)
    assert a.class_names == b.class_names


def test_different_seeds_give_different_scenes():
    a = compose(_pool(), n_kernels=25, kernel_px=30, seed=8)
    b = compose(_pool(), n_kernels=25, kernel_px=30, seed=9)
    assert not np.array_equal(a.image, b.image)


def test_empty_cutout_list_is_rejected():
    with pytest.raises(ValueError, match="no cutouts"):
        compose([])


def test_overlap_must_be_a_fraction():
    with pytest.raises(ValueError, match="max_overlap"):
        compose(_pool(), n_kernels=5, max_overlap=1.0)


def test_canvas_grows_with_kernel_count():
    assert canvas_for(400, 110)[0] > canvas_for(100, 110)[0]


def test_zero_overlap_still_places_kernels():
    scene = compose(_pool(), n_kernels=10, kernel_px=25, max_overlap=0.0, seed=10)
    assert scene.placed > 0


# --- The mechanical set is not restated ------------------------------------

def test_mechanical_set_comes_from_the_mapping_layer():
    assert MECHANICAL == {"seed_coat_cracked", "fragment"}


# --- Instances must stay in one piece -------------------------------------
# Regression: kernels used to overwrite their neighbours, so at realistic
# density an early kernel was nibbled by later ones until its instance was
# shattered into disconnected fragments -- 2.55 visible pieces per instance,
# measured. The counting target was unlearnable as a result.

def test_every_instance_is_a_single_connected_blob():
    from scipy import ndimage

    scene = compose(_pool(), n_kernels=120, kernel_px=30, seed=11)
    for label in np.unique(scene.instances):
        if label == 0:
            continue
        _, pieces = ndimage.label(scene.instances == label)
        assert pieces == 1, f"instance {label} split into {pieces} pieces"


def test_kernels_never_stack():
    scene = compose(_pool(), n_kernels=120, kernel_px=30, seed=12)
    ids = set(np.unique(scene.instances)) - {0}
    assert len(ids) == scene.placed


def test_dense_scenes_still_keep_instances_whole():
    from scipy import ndimage

    # Deliberately crowded: the density at which the old bug appeared.
    scene = compose(_pool(), n_kernels=300, kernel_px=30, size=(400, 400), seed=13)
    broken = 0
    for label in np.unique(scene.instances):
        if label == 0:
            continue
        _, pieces = ndimage.label(scene.instances == label)
        broken += pieces > 1
    assert broken == 0


# --- Kernel views ----------------------------------------------------------
# Regression: a GrainSet file holds one kernel photographed from both sides,
# side by side. Every maize mask splits ~50/50 into two components. Treating
# the pair as one object composited double-width blobs; requiring exactly one
# component rejected every real placement and produced empty scenes. Tests
# used solid rectangles, so neither fault was caught.

def test_a_two_view_mask_yields_two_cutouts(tmp_path):
    from PIL import Image
    from src.data.manifest import Sample
    from src.mapping.classes import CLASSES
    from src.vision.synth import load_cutouts

    mask = np.zeros((40, 80), dtype=np.uint8)
    mask[8:32, 5:35] = 255      # left view
    mask[8:32, 45:75] = 255     # right view
    image = np.full((40, 80, 3), 180, dtype=np.uint8)

    image_path, mask_path = tmp_path / "k.png", tmp_path / "k_mask.png"
    Image.fromarray(image).save(image_path)
    Image.fromarray(mask).save(mask_path)

    cutouts = load_cutouts(
        Sample(
            path=image_path, dataset="grainset", source_label="NOR",
            kernel_class=CLASSES["sound"], session="s", mask=mask_path,
        )
    )
    assert len(cutouts) == 2
    for c in cutouts:
        assert c.alpha.shape[1] == 30      # one view, not the 80px pair


def test_debris_specks_are_not_treated_as_views(tmp_path):
    from PIL import Image
    from src.data.manifest import Sample
    from src.mapping.classes import CLASSES
    from src.vision.synth import load_cutouts

    mask = np.zeros((40, 80), dtype=np.uint8)
    mask[8:32, 5:35] = 255      # the kernel
    mask[1:3, 70:72] = 255      # a speck
    Image.fromarray(np.full((40, 80, 3), 180, dtype=np.uint8)).save(tmp_path / "k.png")
    Image.fromarray(mask).save(tmp_path / "k_mask.png")

    cutouts = load_cutouts(
        Sample(
            path=tmp_path / "k.png", dataset="grainset", source_label="NOR",
            kernel_class=CLASSES["sound"], session="s", mask=tmp_path / "k_mask.png",
        )
    )
    assert len(cutouts) == 1


def test_speckled_masks_still_get_placed():
    # The bug that emptied every scene: a stray component made the placement
    # check reject the position, every time, for every kernel.
    speckled = _cutout("sound", size=(40, 20))
    speckled.alpha[0, 0] = True
    scene = compose([speckled] * 30, kernel_px=25, seed=14)
    assert scene.placed > 0
