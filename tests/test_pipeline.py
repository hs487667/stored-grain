"""Tests for the ranking logic at the end of the pipeline.

Model inference is not exercised here -- that is what
``src.validate_pipeline`` does against scenes of known damage. These pin the
decisions the pipeline makes after the numbers come back.
"""

import pytest

from src.mapping.classes import measure
from src.measurement.sampling import DamageCalibration
from src.physics.deterioration import assess
from src.pipeline import (
    LotReading,
    assert_crop_framing,
    correct_measurement,
    rank_lots,
)


def _reading(lot_id, damage, kernels=200, temperature=20.0, moisture=14.0):
    return LotReading(
        lot_id=lot_id,
        kernels_counted=kernels,
        damage_mass_pct=damage,
        damage_count_pct=damage,
        biological_pct={},
        class_counts={},
        temperature_c=temperature,
        moisture_pct_wb=moisture,
        assessment=assess(damage, temperature, moisture),
    )


def test_more_damage_ranks_as_faster_degrading():
    ranked = rank_lots([_reading("low", 3.0), _reading("high", 25.0)])
    assert [r.reading.lot_id for r in ranked] == ["high", "low"]


def test_ranking_is_by_degradation_rate_not_damage_alone():
    # A cooler, drier lot can outlast a slightly more damaged one. Ordering on
    # damage alone would miss that; the model exists to combine the three.
    warm_damp = _reading("warm", 8.0, temperature=30.0, moisture=18.0)
    cool_dry = _reading("cool", 12.0, temperature=5.0, moisture=13.0)
    ranked = rank_lots([cool_dry, warm_damp])
    assert ranked[0].reading.lot_id == "warm"
    assert warm_damp.damage_mass_pct < cool_dry.damage_mass_pct


def test_lots_within_the_noise_are_reported_as_tied():
    # At 200 kernels the resolvable gap at ~5% damage is about 4 points, so a
    # 1-point difference is not something the measurement can support.
    # b is the more damaged lot, so it ranks first and is tied with a.
    ranked = rank_lots([_reading("a", 5.0), _reading("b", 6.0)])
    assert ranked[0].reading.lot_id == "b"
    assert ranked[0].tied_with == ("a",)
    assert ranked[1].tied_with == ("b",)


def test_clearly_separated_lots_are_not_tied():
    ranked = rank_lots([_reading("a", 3.0), _reading("b", 30.0)])
    assert all(r.tied_with == () for r in ranked)


def test_counting_more_kernels_separates_closer_lots():
    coarse = rank_lots([_reading("a", 5.0, kernels=200), _reading("b", 7.0, kernels=200)])
    fine = rank_lots([_reading("a", 5.0, kernels=5000), _reading("b", 7.0, kernels=5000)])
    assert coarse[0].tied_with == ("a",)
    assert fine[0].tied_with == ()


def test_resolvable_gap_shrinks_with_sample_size():
    assert _reading("a", 5.0, kernels=5000).resolvable_gap_pct < _reading(
        "a", 5.0, kernels=200
    ).resolvable_gap_pct


def test_absolute_days_stay_gated_off():
    # The DML coefficients are still unverified, so every reading must ship as
    # ranking only. If this ever fails, someone flipped the gate.
    ranked = rank_lots([_reading("a", 10.0)])
    assert ranked[0].reading.assessment.days_to_threshold is None
    assert ranked[0].reading.assessment.mode == "ranking"


def test_ranks_are_sequential():
    ranked = rank_lots([_reading(str(i), float(i * 5 + 2)) for i in range(5)])
    assert [r.rank for r in ranked] == [1, 2, 3, 4, 5]


# --- Undoing the classifier's compression ----------------------------------
# The rates are counted per kernel, so the correction has to be applied on the
# count basis and converted afterwards. Correcting the mass percentage directly
# subtracts a count-basis false-positive floor from a mass-basis number.

CALIBRATION = DamageCalibration.from_counts(
    damaged_total=1000, damaged_missed=178, sound_total=1000, sound_flagged=48
)


def test_no_calibration_leaves_the_measurement_untouched():
    m = measure({"sound": 180, "fragment": 20})
    corrected = correct_measurement(m, None)
    assert corrected.mass_pct == pytest.approx(m.mechanical_damage_mass_pct)
    assert corrected.count_pct == pytest.approx(m.mechanical_damage_count_pct)


def test_correction_is_applied_on_the_count_basis_then_converted():
    m = measure({"sound": 180, "fragment": 20})       # 10% by count
    corrected = correct_measurement(m, CALIBRATION)

    expected_count = CALIBRATION.correct(m.mechanical_damage_count_pct)
    assert corrected.count_pct == pytest.approx(expected_count)
    assert corrected.mass_pct == pytest.approx(
        expected_count * m.count_to_mass_ratio
    )


def test_correcting_the_mass_percentage_directly_would_differ():
    # Guard against the shortcut. Fragments weigh under half what they count,
    # so the two routes disagree by more than rounding.
    m = measure({"sound": 180, "fragment": 20})
    corrected = correct_measurement(m, CALIBRATION)
    naive = CALIBRATION.correct(m.mechanical_damage_mass_pct)
    assert abs(corrected.mass_pct - naive) > 0.5


def test_a_reading_at_the_false_positive_floor_corrects_to_zero():
    # 4.8% of sound kernels are flagged, so a sample reading 4.8% by count is
    # consistent with no damage at all.
    m = measure({"sound": 952, "seed_coat_cracked": 48})
    corrected = correct_measurement(m, CALIBRATION)
    assert corrected.count_pct == pytest.approx(0.0, abs=0.05)
    assert corrected.mass_pct == pytest.approx(0.0, abs=0.05)


def test_correction_recovers_the_damage_a_classifier_would_have_reported():
    # End to end on the count basis: take a true 20% damaged sample, let the
    # calibrated error rates distort it, and check the correction gives it back.
    from src.measurement.sampling import biased_damage_pct

    seen = biased_damage_pct(
        20.0, CALIBRATION.false_positive_rate, CALIBRATION.false_negative_rate
    )
    damaged = round(seen * 10)
    m = measure({"sound": 1000 - damaged, "seed_coat_cracked": damaged})
    corrected = correct_measurement(m, CALIBRATION)
    assert corrected.count_pct == pytest.approx(20.0, abs=0.2)


def test_correction_preserves_ranking():
    # The property the whole architecture rests on: correcting changes the
    # numbers, never their order.
    samples = [
        measure({"sound": 1000 - d, "fragment": d}) for d in (10, 40, 90, 200, 400)
    ]
    corrected = [correct_measurement(m, CALIBRATION).mass_pct for m in samples]
    assert corrected == sorted(corrected)


MASS_CALIBRATION = DamageCalibration.from_affine(
    intercept_pct=0.96, slope=0.84, kernels=1200, basis="mass"
)


def test_a_mass_basis_calibration_corrects_the_mass_figure_directly():
    m = measure({"sound": 180, "fragment": 20})
    corrected = correct_measurement(m, MASS_CALIBRATION)
    assert corrected.mass_pct == pytest.approx(
        MASS_CALIBRATION.correct(m.mechanical_damage_mass_pct)
    )


def test_a_mass_basis_correction_carries_the_count_figure_with_it():
    # Both numbers are reported to the user, so both must move together --
    # a corrected mass beside an uncorrected count is two different claims.
    m = measure({"sound": 180, "fragment": 20})
    corrected = correct_measurement(m, MASS_CALIBRATION)
    assert corrected.count_pct == pytest.approx(
        corrected.mass_pct / m.count_to_mass_ratio
    )


def test_the_two_bases_disagree_on_the_same_measurement():
    # If they ever agree, the basis field is doing nothing and one of the two
    # paths is wrong.
    m = measure({"sound": 180, "fragment": 20})
    by_count = correct_measurement(m, CALIBRATION).mass_pct
    by_mass = correct_measurement(m, MASS_CALIBRATION).mass_pct
    assert abs(by_count - by_mass) > 0.5


# --- Checkpoint framing ----------------------------------------------------
# A classifier trained on split GrainSet views scores well on its own metric
# and is unusable here: the pipeline hands it tight bounding-box crops, and on
# a tray of undamaged kernels it called 56 of 161 of them fragments -- 16.91%
# damage against a true 0.00%. The mismatch is invisible in macro-F1, so the
# checkpoint has to declare its framing and be refused on it.

def test_a_checkpoint_trained_on_split_views_is_refused():
    with pytest.raises(ValueError, match="split views"):
        assert_crop_framing({"single_view": True})


def test_a_checkpoint_trained_on_whole_images_is_accepted():
    assert_crop_framing({"single_view": False}) is None


def test_a_checkpoint_that_does_not_say_is_accepted():
    # Checkpoints predating the flag. Silence is not a claim of mismatch, and
    # refusing them would break every model already trained.
    assert_crop_framing({}) is None
