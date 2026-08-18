"""Tests for per-visitor session isolation.

The store was process-wide, which is correct for one photographer on a LAN and
wrong the moment the app is published: two visitors would share one set of lots
and rank each other's grain. These pin the isolation and the eviction that
stops a public URL from growing memory without bound.
"""

import pytest

from src.app.sessions import Session, SessionRegistry


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def test_a_visitor_without_an_id_is_given_one():
    session_id, session = SessionRegistry().get_or_create(None)
    assert session_id
    assert isinstance(session, Session)


def test_the_same_id_comes_back_to_the_same_session():
    registry = SessionRegistry()
    first_id, first = registry.get_or_create(None)
    second_id, second = registry.get_or_create(first_id)
    assert second_id == first_id
    assert second is first


def test_two_visitors_get_separate_sessions():
    registry = SessionRegistry()
    _, one = registry.get_or_create(None)
    _, two = registry.get_or_create(None)
    assert one is not two
    assert one.store is not two.store
    assert one.overlays is not two.overlays


def test_an_unknown_id_does_not_resurrect_a_session():
    # An expired or forged cookie must start a clean session rather than
    # error, and must not be adopted as-is: a visitor could otherwise pick
    # their own id and land in someone else's session by guessing it.
    registry = SessionRegistry()
    session_id, _ = registry.get_or_create("never-issued")
    assert session_id != "never-issued"


def test_an_idle_session_is_evicted():
    clock = FakeClock()
    registry = SessionRegistry(idle_timeout_s=1800, clock=clock)
    session_id, _ = registry.get_or_create(None)

    clock.advance(1801)
    returned_id, _ = registry.get_or_create(session_id)
    assert returned_id != session_id


def test_activity_keeps_a_session_alive():
    clock = FakeClock()
    registry = SessionRegistry(idle_timeout_s=1800, clock=clock)
    session_id, _ = registry.get_or_create(None)

    for _ in range(5):
        clock.advance(1000)
        session_id, _ = registry.get_or_create(session_id)

    clock.advance(100)
    returned_id, _ = registry.get_or_create(session_id)
    assert returned_id == session_id


def test_the_oldest_session_goes_when_the_cap_is_reached():
    clock = FakeClock()
    registry = SessionRegistry(max_sessions=3, clock=clock)

    ids = []
    for _ in range(3):
        clock.advance(1)
        ids.append(registry.get_or_create(None)[0])

    clock.advance(1)
    registry.get_or_create(None)

    assert len(registry) == 3
    # The first issued is the least recently touched, so it is the one dropped.
    assert registry.get_or_create(ids[0])[0] != ids[0]


def test_a_session_holds_its_own_readings_overlays_and_photographs():
    _, session = SessionRegistry().get_or_create(None)
    assert session.store.readings() == []
    assert session.overlays == {}
    assert session.photographs == {}
