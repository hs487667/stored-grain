"""The model against Steele's published numbers.

If any of these fail, a constant has been edited into disagreement with the
dissertation it came from, and the paper's validation section is no longer
true. That is the point of pinning them here rather than only in a script.
"""

import math

import pytest

from src.physics import constants as C
from src.physics.deterioration import days_to_dml
from src.physics.steele_agreement import (
    HAND_SHELLED_1965_HOURS,
    TABLE9,
    all_checks,
    damage_adjustment_checks,
    moisture_adjustment_checks,
    multiplier_checks,
    reference_time_checks,
    ranks_the_two_field_lots_correctly,
    relative_deterioration_checks,
)


@pytest.mark.parametrize("check", multiplier_checks(), ids=lambda c: c.label)
def test_multipliers_reproduce_steeles_worked_example(check):
    """Page 112 states four multiplier values. All four must come back."""
    assert check.agrees, f"{check.label}: Steele {check.published}, model {check.model}"


@pytest.mark.parametrize("check", damage_adjustment_checks(), ids=lambda c: c.label)
def test_damage_adjustment_matches_table_9(check):
    assert check.agrees, f"{check.label}: Steele {check.published}, model {check.model}"


@pytest.mark.parametrize("check", moisture_adjustment_checks(), ids=lambda c: c.label)
def test_moisture_adjustment_matches_table_9(check):
    assert check.agrees, f"{check.label}: Steele {check.published}, model {check.model}"


@pytest.mark.parametrize("check", relative_deterioration_checks(), ids=lambda c: c.label)
def test_predicted_damage_effect_matches_the_measured_one(check):
    """The non-circular check: measured storage life, not a re-applied equation."""
    assert check.physical
    assert check.agrees, f"{check.label}: Steele {check.published}, model {check.model}"


def test_ranking_agrees_with_observed_storage_times():
    """The system's own output, against two lots with measured times."""
    assert ranks_the_two_field_lots_correctly()


def test_at_least_one_check_is_a_measurement():
    """Guards the argument, not the arithmetic.

    An agreement table made entirely of Steele's own adjusted times would be
    circular and would not answer the "monotone by construction" objection.
    """
    assert any(c.physical for c in all_checks())


def test_the_damage_multiplier_shift_between_levels_is_what_carries_table_9():
    """A single all-levels damage form would fail the 0.1% row.

    Steele fitted one exponential per loss level and the levels genuinely
    differ; using the 0.5% coefficients everywhere puts the 0.1% adjustment
    outside its 2% tolerance, so this pins why the dictionary exists.
    """
    observed = TABLE9[("1965", "observed")]
    adjusted = TABLE9[("1965", "adjusted")]
    published = adjusted[0.1] / observed[0.1]

    a, b = C.MD_COEFFICIENTS_BY_DML_LEVEL[0.5]
    wrong = math.exp(b * (adjusted["damage"] - observed["damage"]))
    assert abs(wrong - published) / published > 0.02


def test_hand_shelled_is_slower_than_field_shelled_at_every_level():
    """Sanity on the transcription itself, independent of the model."""
    field = TABLE9[("1965", "observed")]
    for level in (0.1, 0.5, 1.0):
        assert HAND_SHELLED_1965_HOURS[level] > field[level]


@pytest.mark.parametrize("check", reference_time_checks(), ids=lambda c: c.label)
def test_absolute_times_match_steeles_observed_ones(check):
    """The route behind days-to-threshold, against nine measured storage lives."""
    assert check.physical
    assert check.agrees, f"{check.label}: Steele {check.published}, model {check.model}"


def test_the_reference_times_are_steeles_own():
    """Page 108, quoted. A typo here silently rescales every absolute answer."""
    assert C.STEELE_REFERENCE_HOURS == {0.1: 58.0, 0.5: 230.0, 1.0: 356.0}


def test_days_to_dml_refuses_a_level_steele_did_not_publish():
    with pytest.raises(ValueError):
        days_to_dml(10.0, 20.0, 25.0, dml_level=0.25)
