"""Tests for measurement resolution and the fragmentation trap.

The ranking is the product. These tests pin the conditions under which it
means something.
"""

import pytest

from src.mapping.classes import measure
from src.measurement.sampling import (
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
