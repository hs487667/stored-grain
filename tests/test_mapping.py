"""Tests for the mapping layer.

Each one pins a way the damage term can be fed the wrong number while the
pipeline still looks like it is working.
"""

import pytest

from src.mapping.classes import (
    CORNSEEDS,
    GRAINSET_MAIZE,
    Admissibility,
    measure,
    translate,
)


# --- Mechanical only -------------------------------------------------------
# Trap: biological deterioration inflating a multiplier fitted only on
# mechanical damage.

def test_biological_damage_never_enters_the_damage_term():
    m = measure({"sound": 50, "mould_suspect": 25, "insect_damaged": 25})
    assert m.mechanical_damage_mass_pct == 0.0
    assert m.biological_count_pct == {"mould_suspect": 25.0, "insect_damaged": 25.0}


def test_biological_kernels_stay_in_the_denominator():
    # 10 cracked among 100 kernels is 10% damage whether or not some of the
    # other 90 are mouldy -- they are still kernels in the sample.
    plain = measure({"sound": 90, "seed_coat_cracked": 10})
    mixed = measure({"sound": 60, "mould_suspect": 30, "seed_coat_cracked": 10})
    assert plain.mechanical_damage_mass_pct == pytest.approx(10.0)
    assert mixed.mechanical_damage_mass_pct == pytest.approx(10.0)


def test_impurities_leave_both_sides_of_the_ratio():
    without = measure({"sound": 90, "seed_coat_cracked": 10})
    with_chaff = measure({"sound": 90, "seed_coat_cracked": 10, "impurity": 40})
    assert with_chaff.mechanical_damage_mass_pct == pytest.approx(
        without.mechanical_damage_mass_pct
    )
    assert with_chaff.non_kernel_count == 40
    assert with_chaff.kernels_counted == 100


# --- Cracked and broken are both mechanical, at different masses -----------
# Trap: treating a count fraction as a mass fraction.

def test_cracked_kernels_give_equal_count_and_mass_percentages():
    m = measure({"sound": 90, "seed_coat_cracked": 10})
    assert m.mechanical_damage_count_pct == pytest.approx(10.0)
    assert m.mechanical_damage_mass_pct == pytest.approx(10.0)
    assert m.count_to_mass_ratio == pytest.approx(1.0)


def test_fragments_weigh_less_than_they_count():
    m = measure({"sound": 90, "fragment": 10})
    assert m.mechanical_damage_count_pct == pytest.approx(10.0)
    # 10 fragments at 0.367 against 90 sound: 3.67 / 93.67.
    assert m.mechanical_damage_mass_pct == pytest.approx(3.92, abs=0.01)
    assert m.count_to_mass_ratio < 0.5


def test_silkcut_and_broken_are_both_admissible():
    m = measure({"sound": 80, "seed_coat_cracked": 10, "fragment": 10})
    assert m.mechanical_damage_count_pct == pytest.approx(20.0)
    assert m.mechanical_damage_mass_pct > 0.0


# --- Dataset vocabularies --------------------------------------------------
# Trap: GrainSet's BN is fragments, so training on it alone leaves Steele's
# cracked-but-intact case unrepresented.

def test_grainset_broken_maps_to_fragment_not_crack():
    assert GRAINSET_MAIZE["BN"].name == "fragment"
    assert GRAINSET_MAIZE["BN"].relative_mass < 0.5


def test_corn_seeds_silkcut_is_the_cracked_intact_case():
    assert CORNSEEDS["silkcut"].name == "seed_coat_cracked"
    assert CORNSEEDS["silkcut"].relative_mass == 1.0


def test_grainset_has_no_cracked_intact_class_at_all():
    # The gap that makes the Corn Seeds Dataset load-bearing despite its size.
    assert not any(c.name == "seed_coat_cracked" for c in GRAINSET_MAIZE.values())


def test_grainset_defect_classes_are_biological_except_broken():
    for label in ("SD", "AP", "F&S", "HD", "MY"):
        assert GRAINSET_MAIZE[label].admissibility is Admissibility.BIOLOGICAL


def test_translate_merges_labels_sharing_a_class():
    counts = translate({"MY": 5, "F&S": 3, "NOR": 92}, GRAINSET_MAIZE)
    assert counts == {"mould_suspect": 8, "sound": 92}


def test_translate_rejects_an_unknown_label():
    with pytest.raises(ValueError, match="not in this vocabulary"):
        translate({"NOR": 1, "wheat_thing": 1}, GRAINSET_MAIZE)


def test_measure_rejects_an_unknown_class():
    with pytest.raises(ValueError, match="unknown class"):
        measure({"sound": 10, "slightly_off": 1})


def test_measure_rejects_a_sample_with_no_kernels():
    with pytest.raises(ValueError, match="no kernels"):
        measure({"impurity": 30})
