"""Tests for the ranking logic at the end of the pipeline.

Model inference is not exercised here -- that is what
``src.validate_pipeline`` does against scenes of known damage. These pin the
decisions the pipeline makes after the numbers come back.
"""

import pytest

from src.physics.deterioration import assess
from src.pipeline import LotReading, rank_lots


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
