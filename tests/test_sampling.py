"""Tests for measurement resolution and the fragmentation trap.

The ranking is the product. These tests pin the conditions under which it
means something.
"""

import pytest

from src.mapping.classes import measure
from src.measurement.sampling import (
    DamageCalibration,
    FragmentAudit,
    biased_damage_pct,
    count_pct_from_mass_pct,
    kernels_required,
    mass_pct_from_count_pct,
    minimum_resolvable_gap_pct,
    preserves_ordering,
    resolution,
    sample_mass_g,
    standard_error_pct,
    unbias_damage_pct,
)
from src.physics.deterioration import assess, rank


# --- Counting error --------------------------------------------------------

def test_standard_error_falls_as_root_n():
    a = standard_error_pct(10.0, 300)
    b = standard_error_pct(10.0, 1200)
    assert b == pytest.approx(a / 2.0, rel=1e-9)


def test_three_hundred_kernels_cannot_separate_five_from_seven_percent():
    # The headline limit: the demo mixtures are far apart and will separate
    # beautifully, while real lots clustered at 5-7% will not.
    r = resolution(5.0, 300)
    assert r.minimum_resolvable_gap_pct == pytest.approx(3.49, abs=0.05)
    assert not r.can_separate(7.0)


def test_a_thousand_kernels_can_separate_five_from_seven_percent():
    assert resolution(5.0, 1000).can_separate(7.0)


def test_kernels_required_round_trips_with_the_resolvable_gap():
    n = kernels_required(5.0, 2.0)
    assert minimum_resolvable_gap_pct(5.0, n) <= 2.0


def test_sample_mass_is_a_weighable_quantity():
    # ~300 g for 1000 kernels: one photograph, one tray, one scale reading.
    assert sample_mass_g(1000) == pytest.approx(300.0)


# --- Classifier error ------------------------------------------------------

def test_classifier_error_biases_but_does_not_invert():
    rates = (0.10, 0.20)
    measured = [biased_damage_pct(d, *rates) for d in (2.0, 5.0, 10.0, 20.0, 40.0)]
    assert measured == sorted(measured)
    assert measured[0] > 2.0  # biased upward at low damage


def test_any_classifier_better_than_a_coin_preserves_ordering():
    assert preserves_ordering(0.10, 0.20)
    assert preserves_ordering(0.45, 0.45)
    assert not preserves_ordering(0.60, 0.60)


def test_lot_ranking_survives_a_bad_classifier_at_equal_conditions():
    # The claim the whole ranking-first architecture rests on: with lots at
    # the same temperature and moisture, damage is the only ordering term and
    # an affine bias cannot reorder it.
    truth = [3.0, 6.0, 9.0, 15.0, 25.0]
    seen = [biased_damage_pct(d, 0.10, 0.20) for d in truth]

    true_order = [a.damage_pct for a in rank([assess(d, 20.0, 14.0) for d in truth])]
    seen_order = rank([assess(d, 20.0, 14.0) for d in seen])

    assert true_order == sorted(truth, reverse=True)
    assert [a.damage_pct for a in seen_order] == sorted(seen, reverse=True)


# --- Undoing the classifier's bias -----------------------------------------

def test_unbiasing_recovers_the_damage_the_classifier_distorted():
    for truth in (0.0, 2.0, 5.0, 10.0, 20.0, 40.0, 100.0):
        measured = biased_damage_pct(truth, 0.08, 0.15)
        assert unbias_damage_pct(measured, 0.08, 0.15) == pytest.approx(truth)


def test_unbiasing_a_perfect_classifier_changes_nothing():
    assert unbias_damage_pct(7.5, 0.0, 0.0) == pytest.approx(7.5)


def test_unbiasing_stays_inside_the_percentage_range():
    # Sampling noise can push a reading below the false-positive floor, which
    # inverts to a negative damage. A negative percentage is not a measurement.
    assert unbias_damage_pct(1.0, 0.10, 0.20) == 0.0
    assert unbias_damage_pct(99.0, 0.10, 0.20) == 100.0


def test_unbiasing_refuses_a_classifier_no_better_than_a_coin():
    # At FPR + FNR >= 1 the affine map is flat or inverted, so no reading can
    # be attributed to a damage level.
    with pytest.raises(ValueError, match="coin"):
        unbias_damage_pct(10.0, 0.60, 0.60)


def test_calibration_measures_error_rates_from_counted_outcomes():
    # 100 damaged kernels, 20 of them missed; 400 sound, 40 called damaged.
    cal = DamageCalibration.from_counts(
        damaged_total=100, damaged_missed=20, sound_total=400, sound_flagged=40
    )
    assert cal.false_negative_rate == pytest.approx(0.20)
    assert cal.false_positive_rate == pytest.approx(0.10)


def test_calibration_undoes_the_bias_it_measured():
    cal = DamageCalibration.from_counts(
        damaged_total=100, damaged_missed=20, sound_total=400, sound_flagged=40
    )
    assert cal.correct(biased_damage_pct(12.0, 0.10, 0.20)) == pytest.approx(12.0)


def test_calibration_survives_a_round_trip_through_json():
    cal = DamageCalibration.from_counts(
        damaged_total=100, damaged_missed=20, sound_total=400, sound_flagged=40
    )
    assert DamageCalibration.from_dict(cal.to_dict()) == cal


def test_calibration_reports_how_many_kernels_it_rests_on():
    # Rates from a handful of kernels are noise. The count travels with them so
    # a reader can judge whether the correction is worth trusting.
    cal = DamageCalibration.from_counts(
        damaged_total=100, damaged_missed=20, sound_total=400, sound_flagged=40
    )
    assert cal.kernels == 500


def test_calibration_rejects_more_errors_than_kernels():
    with pytest.raises(ValueError):
        DamageCalibration.from_counts(
            damaged_total=10, damaged_missed=11, sound_total=10, sound_flagged=0
        )


# --- Fragmentation ---------------------------------------------------------

def test_whole_cracked_kernels_keep_count_and_mass_in_agreement():
    for mass_pct in (2.0, 5.0, 10.0, 20.0, 40.0):
        assert count_pct_from_mass_pct(mass_pct, 1.0) == pytest.approx(mass_pct)


def test_shattering_inflates_a_ten_percent_mixture_to_twenty_five():
    # The number that condemns impact cracking: the scale says 10%, the
    # camera says 25%, and the equation is handed 25%.
    assert count_pct_from_mass_pct(10.0, 3.0) == pytest.approx(25.0)


def test_inflation_grows_with_fragment_count():
    seen = [count_pct_from_mass_pct(10.0, f) for f in (1.0, 2.0, 3.0, 5.0)]
    assert seen == sorted(seen)
    assert seen[-1] == pytest.approx(35.7, abs=0.1)


def test_mass_recovery_inverts_the_inflation():
    for f in (1.0, 2.0, 3.0, 5.0):
        counted = count_pct_from_mass_pct(10.0, f)
        assert mass_pct_from_count_pct(counted, f) == pytest.approx(10.0)


def test_pressure_cracking_passes_the_audit():
    audit = FragmentAudit(
        kernels_cracked=100, pieces_counted=108,
        mass_before_g=30.0, mass_after_g=29.8,
    )
    assert audit.fragments_per_kernel == pytest.approx(1.08)
    assert audit.acceptable
    assert audit.verdict == "acceptable"


def test_impact_cracking_fails_the_audit_and_says_why():
    audit = FragmentAudit(
        kernels_cracked=100, pieces_counted=290,
        mass_before_g=30.0, mass_after_g=28.5,
    )
    assert not audit.acceptable
    assert "pieces per kernel" in audit.verdict
    assert "fines" in audit.verdict


def test_fines_loss_alone_fails_the_audit():
    audit = FragmentAudit(
        kernels_cracked=100, pieces_counted=105,
        mass_before_g=30.0, mass_after_g=28.0,
    )
    assert not audit.acceptable
    assert "fines" in audit.verdict


# --- The two modules agree ------------------------------------------------

def test_mapping_and_sampling_agree_on_what_fragments_cost():
    # A 10% mass mixture shattered into pieces, classified perfectly: the
    # mapping layer's mass reconstruction should undo what fragmentation did
    # to the count, because both use the same relative-mass anchor.
    m = measure({"sound": 90, "fragment": 10})
    assert m.mechanical_damage_count_pct > m.mechanical_damage_mass_pct


# --- Calibrating the whole pipeline rather than the classifier alone --------

def test_affine_calibration_inverts_the_fitted_line():
    # measured = 0.96 + 0.84 * true, fitted over trays of known damage.
    cal = DamageCalibration.from_affine(intercept_pct=0.96, slope=0.84, kernels=1200)
    assert cal.correct(0.96 + 0.84 * 20.0) == pytest.approx(20.0)
    assert cal.correct(0.96) == pytest.approx(0.0)


def test_affine_calibration_reads_as_error_rates():
    cal = DamageCalibration.from_affine(intercept_pct=5.0, slope=0.75, kernels=1000)
    assert cal.false_positive_rate == pytest.approx(0.05)
    assert cal.false_negative_rate == pytest.approx(0.20)


def test_affine_calibration_refuses_a_flat_or_falling_line():
    # A slope at or below zero means the measurement carries no information
    # about damage, and no correction can invent it.
    with pytest.raises(ValueError):
        DamageCalibration.from_affine(intercept_pct=1.0, slope=0.0, kernels=100)


def test_calibration_records_which_basis_it_was_fitted_on():
    # Count-basis rates come from per-kernel confusion, mass-basis ones from a
    # fit against weighed trays. Applying one as the other is a real error.
    counted = DamageCalibration.from_counts(
        damaged_total=100, damaged_missed=20, sound_total=400, sound_flagged=40
    )
    fitted = DamageCalibration.from_affine(
        intercept_pct=0.96, slope=0.84, kernels=1200, basis="mass"
    )
    assert counted.basis == "count"
    assert fitted.basis == "mass"
    assert DamageCalibration.from_dict(fitted.to_dict()).basis == "mass"


def test_calibration_rejects_an_unknown_basis():
    with pytest.raises(ValueError, match="basis"):
        DamageCalibration.from_affine(
            intercept_pct=1.0, slope=0.9, kernels=100, basis="volume"
        )
