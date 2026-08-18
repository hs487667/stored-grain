"""Tests for deriving moisture content from a hygrometer reading.

The deterioration model needs moisture content. Most people who store grain
own a hygrometer and not a moisture meter, so without this conversion the
system cannot be used by the person it is for. A derived figure carries more
error than a measured one, and these pin that it is never passed off as one.
"""

import pytest

from src.physics.moisture import (
    EMC_VERIFIED,
    DerivedMoisture,
    RH_VALID_MAX,
    RH_VALID_MIN,
    moisture_from_humidity,
)


def test_corn_at_room_conditions_lands_where_the_literature_puts_it():
    # Published equilibrium tables put shelled corn near 13.5% wet basis at
    # 25 C and 65% RH. A conversion that misses this is wrong in a way no
    # downstream test would reveal.
    derived = moisture_from_humidity(temperature_c=25.0, relative_humidity=0.65)
    assert derived.moisture_pct_wb == pytest.approx(13.5, abs=0.3)


def test_a_damper_atmosphere_gives_wetter_grain():
    dry = moisture_from_humidity(25.0, 0.55)
    damp = moisture_from_humidity(25.0, 0.80)
    assert damp.moisture_pct_wb > dry.moisture_pct_wb


def test_warmer_air_at_the_same_humidity_gives_drier_grain():
    # Equilibrium moisture falls with temperature at fixed humidity. Getting
    # this backwards would invert the seasonal behaviour of the whole model.
    cool = moisture_from_humidity(10.0, 0.70)
    warm = moisture_from_humidity(35.0, 0.70)
    assert warm.moisture_pct_wb < cool.moisture_pct_wb


def test_the_result_says_it_was_derived():
    derived = moisture_from_humidity(25.0, 0.65)
    assert derived.derived is True
    assert derived.source


def test_the_result_carries_more_uncertainty_than_a_meter_would():
    derived = moisture_from_humidity(25.0, 0.65)
    assert derived.uncertainty_pct_wb > 0.0


def test_humidity_outside_the_fitted_range_is_refused():
    # The isotherm was fitted over a band of humidities. Outside it the
    # equation still returns a number, and the number is not evidence.
    with pytest.raises(ValueError, match="humidity"):
        moisture_from_humidity(25.0, RH_VALID_MAX + 0.05)
    with pytest.raises(ValueError, match="humidity"):
        moisture_from_humidity(25.0, RH_VALID_MIN - 0.05)


def test_humidity_given_as_a_percentage_is_refused_rather_than_misread():
    # 65 and 0.65 are both plausible things to type. Silently accepting 65
    # would sail past the range check as a fraction and produce nonsense.
    with pytest.raises(ValueError, match="humidity"):
        moisture_from_humidity(25.0, 65.0)


def test_absolute_zero_of_humidity_does_not_divide_by_zero():
    with pytest.raises(ValueError):
        moisture_from_humidity(25.0, 0.0)


def test_the_constants_are_marked_unverified_until_the_standard_is_obtained():
    # Mirrors CONSTANTS_VERIFIED for the dry-matter-loss curve: the isotherm
    # parameters come from one published corn sample, not from ASABE D245.6.
    assert EMC_VERIFIED is False


def test_a_derived_moisture_can_be_fed_to_the_deterioration_model():
    from src.physics.deterioration import assess

    derived = moisture_from_humidity(25.0, 0.70)
    assessment = assess(10.0, 25.0, derived.moisture_pct_wb)
    assert assessment.degradation_rate > 0.0


def test_the_dataclass_is_not_silently_mistaken_for_a_float():
    derived = moisture_from_humidity(25.0, 0.65)
    assert isinstance(derived, DerivedMoisture)
    with pytest.raises(TypeError):
        float(derived) + 1  # noqa
