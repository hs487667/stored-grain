"""Tests for the guards a public URL needs.

Every upload runs a neural network on this machine, so the endpoint is
expensive in a way an ordinary form post is not. These pin the three ways an
open URL can be made to hurt: an enormous file, a small file that decodes to an
enormous one, and simple repetition.
"""

import pytest

from src.app.limits import RateLimiter


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def test_requests_inside_the_limit_pass():
    limiter = RateLimiter(limit=3, window_s=60, clock=FakeClock())
    assert all(limiter.check("ip") is None for _ in range(3))


def test_the_request_over_the_limit_is_refused():
    limiter = RateLimiter(limit=3, window_s=60, clock=FakeClock())
    for _ in range(3):
        limiter.check("ip")
    assert limiter.check("ip") is not None


def test_a_refusal_says_how_long_to_wait():
    clock = FakeClock()
    limiter = RateLimiter(limit=2, window_s=60, clock=clock)
    limiter.check("ip")
    clock.advance(10)
    limiter.check("ip")

    retry_after = limiter.check("ip")
    # The oldest of the two is 10s old, so its slot frees in 50s.
    assert retry_after == pytest.approx(50.0, abs=0.01)


def test_the_window_slides_rather_than_resetting():
    # A fixed window lets a caller spend a full allowance either side of the
    # boundary and get double the rate in an instant.
    clock = FakeClock()
    limiter = RateLimiter(limit=2, window_s=60, clock=clock)
    limiter.check("ip")
    limiter.check("ip")
    assert limiter.check("ip") is not None

    clock.advance(61)
    assert limiter.check("ip") is None


def test_callers_are_counted_separately():
    limiter = RateLimiter(limit=1, window_s=60, clock=FakeClock())
    assert limiter.check("first") is None
    assert limiter.check("second") is None


def test_idle_callers_are_forgotten():
    # Otherwise a public URL accumulates one entry per address, forever.
    clock = FakeClock()
    limiter = RateLimiter(limit=5, window_s=60, clock=clock)
    limiter.check("passing-by")
    assert len(limiter) == 1

    clock.advance(600)
    limiter.check("someone-else")
    assert len(limiter) == 1
