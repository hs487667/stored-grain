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
