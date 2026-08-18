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


def test_a_reading_in_range_carries_absolute_days():
    # 14% moisture and 10% damage sit inside the published ranges, which is
    # where bagged maize actually lives, so a reading there must reach the
    # absolute answer rather than stopping at a rank.
    ranked = rank_lots([_reading("a", 10.0)])
    assessment = ranked[0].reading.assessment
    assert assessment.days_to_threshold is not None
    assert assessment.days_to_threshold_error_pct is not None
    assert assessment.mode == "ranking+absolute"


def test_an_out_of_range_reading_still_ranks():
    # Below the moisture floor the absolute answer is refused, but the lot
    # still has to take its place in the ordering.
    ranked = rank_lots([_reading("a", 10.0, moisture=11.0)])
    assessment = ranked[0].reading.assessment
    assert assessment.days_to_threshold is None
    assert assessment.mode == "ranking"
    assert assessment.degradation_rate > 0.0


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


# --- Raw figures are always present ----------------------------------------

def test_a_reading_built_without_a_correction_reports_itself_as_raw():
    # An uncorrected reading is its own raw figure. Leaving these None would
    # make every consumer test for it, and the first one to forget gets a
    # TypeError halfway through a validation run.
    reading = _reading("a", 7.5)
    assert reading.raw_damage_mass_pct == pytest.approx(7.5)
    assert reading.raw_damage_count_pct == pytest.approx(7.5)


def test_an_explicit_raw_figure_is_kept():
    reading = LotReading(
        lot_id="a",
        kernels_counted=200,
        damage_mass_pct=6.0,
        damage_count_pct=6.0,
        biological_pct={},
        class_counts={},
        temperature_c=20.0,
        moisture_pct_wb=14.0,
        assessment=assess(6.0, 20.0, 14.0),
        raw_damage_mass_pct=7.5,
        raw_damage_count_pct=7.4,
    )
    assert reading.raw_damage_mass_pct == pytest.approx(7.5)
    assert reading.raw_damage_count_pct == pytest.approx(7.4)


# --- Per-instance detail ---------------------------------------------------
# The overlay needs what the reading throws away. These pin the relationship
# between the two so the app can never draw a different measurement from the
# one the pipeline reported.

def test_detections_line_up_one_to_one_with_predictions():
    import numpy as np

    from src.pipeline import Detections

    detections = Detections(
        instances=np.array([[0, 1], [2, 2]]),
        labels=[1, 2],
        predictions=["sound", "fragment"],
    )
    assert len(detections.labels) == len(detections.predictions)


def test_detections_reject_a_mismatched_pairing():
    import numpy as np

    from src.pipeline import Detections

    with pytest.raises(ValueError, match="one prediction per instance"):
        Detections(
            instances=np.zeros((2, 2), dtype=int),
            labels=[1, 2],
            predictions=["sound"],
        )


# --- Framing -------------------------------------------------------------
# The localiser has no scale invariance. Handed a close-up it does not degrade,
# it merges the frame into one blob and reports a confident percentage of
# nothing. These pin the refusal, because a wrong number that looks calm is
# worse than an error.

def test_a_close_up_is_refused_and_says_which_way_to_move():
    import numpy as np

    from src.pipeline import FramingError, check_framing

    # Enough instances to clear the count floor, each far too large: the phone
    # is close enough that only a handful of kernels are in view.
    instances = np.zeros((100, 100), dtype=int)
    labels = []
    for i in range(12):
        y, x = (i // 4) * 30, (i % 4) * 25
        instances[y : y + 15, x : x + 15] = i + 1
        labels.append(i + 1)

    with pytest.raises(FramingError, match="further away"):
        check_framing(instances, labels)


def test_a_single_blob_is_refused_for_having_no_kernels_to_count():
    import numpy as np

    from src.pipeline import FramingError, check_framing

    # What a GrainSet single-kernel file actually produces: the localiser
    # merges the frame into one instance rather than finding kernels.
    instances = np.zeros((100, 100), dtype=int)
    instances[5:95, 5:95] = 1

    with pytest.raises(FramingError, match="only 1 kernels"):
        check_framing(instances, [1])


def test_kernels_far_too_small_are_refused():
    import numpy as np

    from src.pipeline import FramingError, check_framing

    # Many 1-pixel specks: the tray is a long way off, or this is noise.
    instances = np.zeros((400, 400), dtype=int)
    labels = []
    for i in range(60):
        instances[i * 6, (i * 7) % 400] = i + 1
        labels.append(i + 1)

    with pytest.raises(FramingError, match="further|too small|farther"):
        check_framing(instances, labels)


def test_too_few_kernels_to_support_a_percentage_is_refused():
    import numpy as np

    from src.pipeline import FramingError, check_framing

    instances = np.zeros((1391, 1391), dtype=int)
    labels = []
    for i in range(4):
        instances[i * 100 : i * 100 + 88, 0:88] = i + 1
        labels.append(i + 1)

    with pytest.raises(FramingError, match="kernels"):
        check_framing(instances, labels)


def test_a_properly_framed_tray_passes():
    import numpy as np

    from src.pipeline import check_framing

    # 150 kernels of the trained size in a 1391px frame.
    instances = np.zeros((1391, 1391), dtype=int)
    labels = []
    side = 88
    n = 0
    for row in range(13):
        for col in range(13):
            if n >= 150:
                break
            y, x = row * 100, col * 100
            n += 1
            instances[y : y + side, x : x + side] = n
            labels.append(n)
    check_framing(instances, labels)


def test_the_validation_trays_all_pass_framing():
    # The guard must not reject the very images the measurement was validated
    # on, which is the only way to know the band was not set too tight.
    import glob

    import numpy as np
    from PIL import Image

    from src.pipeline import MIN_RELATIVE_AREA, check_framing

    trays = sorted(glob.glob("data/interim/validation/lot_*.png"))
    if not trays:
        pytest.skip("validation trays not on disk")

    from src.pipeline import Pipeline
    from src.vision.evaluate_localiser import predict_scene
    from src.vision.localise import instances_from_logits

    pipeline = Pipeline()
    for tray in trays:
        image = np.array(Image.open(tray).convert("RGB"))
        instances = instances_from_logits(
            predict_scene(pipeline.localiser, image, pipeline.device)
        )
        labels = [int(l) for l in np.unique(instances) if l != 0]
        areas = {l: int((instances == l).sum()) for l in labels}
        floor = np.median(list(areas.values())) * MIN_RELATIVE_AREA
        kept = [l for l in labels if areas[l] >= floor]
        check_framing(instances, kept)


test_the_validation_trays_all_pass_framing = pytest.mark.slow(
    test_the_validation_trays_all_pass_framing
)
