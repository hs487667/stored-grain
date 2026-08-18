"""Tests for the app's session store.

A session is one sitting at one tray. These pin the ways a reading can go
missing, be silently replaced, or reach the ranking in the wrong order.
"""

import pytest

from src.app.store import SessionStore
from src.physics.deterioration import assess
from src.pipeline import LotReading


def _reading(lot_id, damage, kernels=200):
    return LotReading(
        lot_id=lot_id,
        kernels_counted=kernels,
        damage_mass_pct=damage,
        damage_count_pct=damage,
        biological_pct={},
        class_counts={},
        temperature_c=20.0,
        moisture_pct_wb=14.0,
        assessment=assess(damage, 20.0, 14.0),
    )


def test_a_stored_reading_comes_back():
    store = SessionStore()
    store.add(_reading("lot_a", 5.0))
    assert store.get("lot_a").damage_mass_pct == pytest.approx(5.0)


def test_an_unknown_lot_is_none_rather_than_an_error():
    assert SessionStore().get("nobody") is None


def test_adding_the_same_lot_twice_replaces_it():
    # Re-photographing a lot is the normal way to fix a bad shot, so the
    # second reading must win rather than accumulate beside the first.
    store = SessionStore()
    store.add(_reading("lot_a", 5.0))
    store.add(_reading("lot_a", 9.0))
    assert len(store.readings()) == 1
    assert store.get("lot_a").damage_mass_pct == pytest.approx(9.0)


def test_removing_a_lot_reports_whether_it_was_there():
    store = SessionStore()
    store.add(_reading("lot_a", 5.0))
    assert store.remove("lot_a") is True
    assert store.remove("lot_a") is False


def test_reset_empties_the_session():
    store = SessionStore()
    store.add(_reading("lot_a", 5.0))
    store.add(_reading("lot_b", 8.0))
    store.reset()
    assert store.readings() == []


def test_ranking_orders_by_degradation_rate():
    store = SessionStore()
    store.add(_reading("slow", 3.0))
    store.add(_reading("fast", 25.0))
    assert [r.reading.lot_id for r in store.ranking()] == ["fast", "slow"]


def test_ranking_does_not_depend_on_insertion_order():
    forward, backward = SessionStore(), SessionStore()
    for store, order in ((forward, ["a", "b", "c"]), (backward, ["c", "b", "a"])):
        for lot_id in order:
            store.add(_reading(lot_id, {"a": 4.0, "b": 12.0, "c": 30.0}[lot_id]))
    assert [r.reading.lot_id for r in forward.ranking()] == \
           [r.reading.lot_id for r in backward.ranking()]


def test_ranking_an_empty_session_is_empty_not_an_error():
    assert SessionStore().ranking() == []


def test_membership_is_testable_without_fetching():
    store = SessionStore()
    store.add(_reading("lot_a", 5.0))
    assert "lot_a" in store
    assert "lot_z" not in store
