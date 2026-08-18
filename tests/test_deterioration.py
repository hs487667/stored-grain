"""Tests for the deterioration model.

These are not coverage tests. Each one pins a specific way the model can be
silently wrong -- the failure modes catalogued in master plan section 11.7,
where the output stays plausible while the arithmetic is inverted.
"""

import math

import pytest

from src.physics import constants as C
from src.physics.deterioration import (
    assess,
    check_ranges,
    damage_multiplier,
    days_to_dml,
    dry_matter_loss_pct,
    equivalent_hours_for_dml,
    moisture_db_to_wb,
    moisture_multiplier,
    moisture_wb_to_db,
    rank,
    temperature_multiplier,
)


# --- Reference conditions --------------------------------------------------
# Trap 1: the multipliers are normalised to 30% damage, not to zero damage.

def test_moisture_multiplier_is_one_at_reference():
    assert moisture_multiplier(C.REFERENCE_MOISTURE_PCT_WB) == pytest.approx(1.0, abs=0.01)


def test_temperature_multiplier_is_one_at_reference():
    mt = temperature_multiplier(C.REFERENCE_TEMPERATURE_C, C.REFERENCE_MOISTURE_PCT_WB)
    assert mt == pytest.approx(1.0, abs=0.01)


def test_damage_multiplier_is_one_at_reference():
    assert damage_multiplier(C.REFERENCE_DAMAGE_PCT) == pytest.approx(1.0, abs=0.03)


def test_damage_multiplier_is_not_one_at_zero_damage():
    """The trap itself. Coding MD = 1 at zero damage is the wrong instinct."""
    assert damage_multiplier(0.0) == pytest.approx(C.MD_A)
    assert damage_multiplier(0.0) > 2.0


# --- Direction of the damage effect ---------------------------------------
# The composition is division. Getting it backwards inverts the ranking, which
# is the system's primary output, while every number still looks reasonable.

def test_damage_multiplier_decreases_with_damage():
    values = [damage_multiplier(d) for d in (2, 5, 10, 20, 30, 40)]
    assert values == sorted(values, reverse=True)


def test_more_damage_degrades_faster():
    low = assess(damage_pct=5.0, temperature_c=20.0, moisture_pct_wb=25.0)
    high = assess(damage_pct=35.0, temperature_c=20.0, moisture_pct_wb=25.0)
    assert high.degradation_rate > low.degradation_rate


def test_more_damage_shortens_time_to_threshold():
    kwargs = dict(temperature_c=20.0, moisture_pct_wb=25.0, constants_verified=True)
    low = assess(damage_pct=5.0, **kwargs)
    high = assess(damage_pct=35.0, **kwargs)
    assert low.days_to_threshold is not None
    assert high.days_to_threshold is not None
    assert high.days_to_threshold < low.days_to_threshold


def test_warmer_degrades_faster():
    cool = assess(damage_pct=10.0, temperature_c=10.0, moisture_pct_wb=25.0)
    warm = assess(damage_pct=10.0, temperature_c=30.0, moisture_pct_wb=25.0)
    assert warm.degradation_rate > cool.degradation_rate


def test_wetter_degrades_faster():
    dry = assess(damage_pct=10.0, temperature_c=20.0, moisture_pct_wb=16.0)
    wet = assess(damage_pct=10.0, temperature_c=20.0, moisture_pct_wb=28.0)
    assert wet.degradation_rate > dry.degradation_rate


# --- The DML curve ---------------------------------------------------------

def test_dml_is_zero_at_zero_time():
    assert dry_matter_loss_pct(0.0) == pytest.approx(0.0)


def test_dml_is_strictly_increasing():
    values = [dry_matter_loss_pct(t) for t in (0, 50, 100, 200, 400)]
    assert values == sorted(values)
    assert len(set(values)) == len(values)


def test_threshold_inversion_round_trips():
    tr = equivalent_hours_for_dml(C.DML_THRESHOLD_PCT)
    assert dry_matter_loss_pct(tr) == pytest.approx(C.DML_THRESHOLD_PCT, abs=1e-6)


def test_threshold_at_reference_matches_literature():
    """0.5% DML at reference lands near 9.6 days.

    The literature reports that 25% m.c. maize can lose 0.5% dry matter in
    about a week. Same order, which is the most that can be claimed while
    Thompson's own coefficients are unverified.
    """
    tr = equivalent_hours_for_dml(C.DML_THRESHOLD_PCT)
    assert 200.0 < tr < 260.0
    assert 8.0 < tr / 24.0 < 11.0


def test_reference_conditions_give_unit_product():
    a = assess(
        damage_pct=C.REFERENCE_DAMAGE_PCT,
        temperature_c=C.REFERENCE_TEMPERATURE_C,
        moisture_pct_wb=C.REFERENCE_MOISTURE_PCT_WB,
        constants_verified=True,
    )
    assert a.mm * a.mt * a.md == pytest.approx(1.0, abs=0.03)
    tr = equivalent_hours_for_dml(C.DML_THRESHOLD_PCT)
    assert a.days_to_threshold == pytest.approx(tr / 24.0, rel=0.03)


# --- Moisture basis --------------------------------------------------------

def test_moisture_basis_round_trip():
    for wb in (10.0, 14.5, 25.0, 33.0):
        assert moisture_db_to_wb(moisture_wb_to_db(wb)) == pytest.approx(wb)


def test_reference_moisture_dry_basis():
    assert moisture_wb_to_db(25.0) == pytest.approx(100.0 / 3.0)


def test_published_moisture_range_matches_across_bases():
    """13-35% wet basis and 0.149-0.538 kg/kg dry basis are the same range.

    Two sources quote the moisture validity range on different bases, which
    reads like a disagreement until converted. Pinning it prevents someone
    "fixing" one to match the other.
    """
    lo, hi = C.MM_VALID_WB_PCT
    assert moisture_wb_to_db(lo) / 100.0 == pytest.approx(0.149, abs=0.001)
    assert moisture_wb_to_db(hi) / 100.0 == pytest.approx(0.538, abs=0.001)


# --- The gate --------------------------------------------------------------

def test_days_to_threshold_does_not_wait_on_thompson():
    """The absolute figure comes from Steele, whose dissertation is on disk.

    Thompson supplies the dry-matter-loss curve, which answers how much has
    been lost after N hours. Days-to-threshold never asks that question, so
    gating it on Thompson withheld an answer the sources already supported.
    """
    a = assess(damage_pct=10.0, temperature_c=20.0, moisture_pct_wb=25.0,
               constants_verified=False)
    assert a.days_to_threshold is not None
    assert a.mode == "ranking+absolute"
    assert a.days_to_threshold_thompson is None


def test_the_absolute_figure_never_travels_without_its_error():
    """An 11% standard error quoted separately is an error quoted never."""
    a = assess(damage_pct=10.0, temperature_c=20.0, moisture_pct_wb=25.0)
    assert a.days_to_threshold_error_pct == C.STEELE_STANDARD_ERROR_PCT[
        C.DML_THRESHOLD_PCT
    ]
    assert any("standard error" in n for n in a.model_notes)


def test_thompson_is_a_note_not_a_suppression_reason():
    """Conflating the two produced a screen that withheld a figure it showed."""
    a = assess(damage_pct=10.0, temperature_c=20.0, moisture_pct_wb=25.0,
               constants_verified=False)
    assert a.suppression_reasons == ()
    assert any("Thompson" in n for n in a.model_notes)


def test_ranking_survives_unverified_constants():
    """The whole point of ranking-first: it does not need any absolute path."""
    lots = [
        assess(damage_pct=d, temperature_c=20.0, moisture_pct_wb=25.0,
               constants_verified=False)
        for d in (5.0, 30.0, 12.0)
    ]
    ordered = rank(lots)
    assert [a.damage_pct for a in ordered] == [30.0, 12.0, 5.0]
    assert all(a.days_to_threshold_thompson is None for a in ordered)


def test_module_gate_is_still_closed():
    """Guards against someone flipping the flag without obtaining the paper."""
    assert C.CONSTANTS_VERIFIED is False


def test_the_two_absolute_routes_agree_when_both_are_open():
    """If Thompson is ever obtained, the routes must not contradict.

    They are independent. One inverts Thompson's loss curve to find the hours
    at which 0.5% is reached; the other reads the reference time Steele fitted
    to his own residuals. Different authors, different data, different method.

    They land 0.4% apart -- 230.9 hours against 230. That is not a check this
    project engineered, and it is the strongest evidence available that the
    Thompson coefficients transcribed from a review are the right ones, since
    a transcription error would have to be a coincidence of this size to
    survive. It does not open the gate, because agreement is not provenance,
    but it is worth pinning tightly: if an edit ever moves it, one of the two
    sources has been misread.
    """
    a = assess(damage_pct=20.0, temperature_c=15.6, moisture_pct_wb=25.0,
               constants_verified=True)
    assert a.days_to_threshold_thompson is not None
    ratio = a.days_to_threshold_thompson / a.days_to_threshold
    assert ratio == pytest.approx(1.0, abs=0.01)


def test_the_route_agreement_is_independent_of_conditions():
    """Both routes are the same multiplier product times a constant.

    So the ratio is that constant's ratio and nothing else. Pinning this stops
    someone reading a single agreeing case as agreement across the range.
    """
    mild = assess(damage_pct=5.0, temperature_c=15.0, moisture_pct_wb=14.0,
                  constants_verified=True)
    harsh = assess(damage_pct=35.0, temperature_c=30.0, moisture_pct_wb=30.0,
                   constants_verified=True)
    assert (mild.days_to_threshold_thompson / mild.days_to_threshold) == pytest.approx(
        harsh.days_to_threshold_thompson / harsh.days_to_threshold
    )


# --- Validity ranges -------------------------------------------------------
# Trap 2: refusing out-of-range inputs is honest but useless on its own, so
# the ranking path must stay alive when the absolute path is refused.

def test_dried_maize_is_out_of_range_but_still_ranks():
    """The most common real query sits below the model's moisture floor."""
    a = assess(damage_pct=8.0, temperature_c=25.0, moisture_pct_wb=12.0)
    assert not a.ranges.moisture_ok
    assert a.suppression_reasons
    assert a.days_to_threshold is None
    assert a.mode == "ranking"
    assert a.degradation_rate > 0.0


def test_out_of_range_damage_is_flagged():
    assert not check_ranges(20.0, 25.0, 1.0).damage_ok
    assert not check_ranges(20.0, 25.0, 45.0).damage_ok
    assert check_ranges(20.0, 25.0, 20.0).damage_ok


def test_range_notes_explain_each_refusal():
    checked = check_ranges(60.0, 12.0, 50.0)
    assert not checked.all_ok
    assert len(checked.notes) == 3


def test_in_range_opens_the_absolute_path():
    a = assess(damage_pct=20.0, temperature_c=20.0, moisture_pct_wb=25.0)
    assert a.ranges.all_ok
    assert a.days_to_threshold is not None
    assert a.mode == "ranking+absolute"


def test_worse_conditions_shorten_the_absolute_answer():
    """The direction the whole composition can silently invert.

    Days-to-threshold multiplies by the same multipliers that equivalent hours
    divides by. Getting one right and the other backwards is possible, so both
    directions are pinned.
    """
    clean = assess(damage_pct=5.0, temperature_c=15.0, moisture_pct_wb=14.0)
    dirty = assess(damage_pct=35.0, temperature_c=30.0, moisture_pct_wb=30.0)
    assert dirty.days_to_threshold < clean.days_to_threshold
    assert dirty.degradation_rate > clean.degradation_rate


# --- Input validation ------------------------------------------------------

def test_negative_damage_rejected():
    with pytest.raises(ValueError):
        damage_multiplier(-1.0)


def test_impossible_moisture_rejected():
    with pytest.raises(ValueError):
        moisture_wb_to_db(100.0)


def test_unknown_dml_level_rejected():
    with pytest.raises(ValueError):
        damage_multiplier(10.0, dml_level=0.25)


# --- Against the review's re-fit -------------------------------------------

def test_steele_and_kaleta_agree_only_near_the_reference_point():
    """Documents why the review's quadratic is not used.

    Both are monotone, so ranking is unaffected -- but the absolute numbers
    diverge by up to 65% in the low-damage regime that dried maize occupies.
    """
    def kaleta(d):
        return 0.001 * d * d - 0.1101 * d + 3.426

    assert kaleta(30.0) == pytest.approx(damage_multiplier(30.0), rel=0.02)
    assert kaleta(2.0) / damage_multiplier(2.0) > 1.5

    steele_values = [damage_multiplier(d) for d in (2, 10, 20, 30, 40)]
    kaleta_values = [kaleta(d) for d in (2, 10, 20, 30, 40)]
    assert steele_values == sorted(steele_values, reverse=True)
    assert kaleta_values == sorted(kaleta_values, reverse=True)


def test_multipliers_are_finite_across_the_valid_domain():
    for t in (2.0, 15.6, 30.0, 48.0):
        for m in (14.0, 20.0, 25.0, 33.0):
            for d in (2.0, 20.0, 40.0):
                a = assess(damage_pct=d, temperature_c=t, moisture_pct_wb=m)
                assert math.isfinite(a.degradation_rate)
                assert a.degradation_rate > 0.0
