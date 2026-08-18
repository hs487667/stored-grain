"""The deterioration model: pure functions, no state, no learning.

The whole module is arithmetic over published constants. It has no dependency
on the vision pipeline and must stay that way -- a user who already knows their
damage percentage can get an answer while the inference service is down.

Two output paths, never one:

* **Ranking** is always available. It needs only that the damage multiplier be
  monotone in damage, which holds for every candidate form of the equation, so
  it survives unverified coefficients and out-of-range inputs alike.
* **Absolute** days-to-threshold appears when the inputs sit inside the
  published validity ranges. It comes from Steele's own reference times
  (Equation 13, t = t_R * MT * MM * MD), which are in the dissertation on
  disk, so it does not wait on Thompson. The figure carries Steele's stated
  standard error with it and must never be quoted without it.

The Thompson dry-matter-loss curve remains behind ``CONSTANTS_VERIFIED`` and
answers a different question -- how much dry matter is lost after a given
number of hours -- which nothing in the ranking or the storage decision asks.

Composition note, because getting it backwards silently inverts the primary
output: the multipliers are DIVISORS.

    tr = t / (MM * MT * MD)

Each multiplier equals 1.0 at the reference conditions and *decreases* as
conditions worsen, so dividing by them inflates equivalent time. Writing
``tr = t * MM * MT * MD`` would make heavily damaged maize appear to degrade
more slowly and would reverse the lot ranking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from . import constants as C


# --- Moisture basis conversion --------------------------------------------

def moisture_wb_to_db(moisture_pct_wb: float) -> float:
    """Convert moisture from a wet basis to a dry basis, both as percentages."""
    if not 0.0 <= moisture_pct_wb < 100.0:
        raise ValueError(f"moisture must be in [0, 100) % wet basis, got {moisture_pct_wb}")
    return 100.0 * moisture_pct_wb / (100.0 - moisture_pct_wb)


def moisture_db_to_wb(moisture_pct_db: float) -> float:
    """Convert moisture from a dry basis to a wet basis, both as percentages."""
    if moisture_pct_db < 0.0:
        raise ValueError(f"dry-basis moisture cannot be negative, got {moisture_pct_db}")
    return 100.0 * moisture_pct_db / (100.0 + moisture_pct_db)


# --- The three multipliers -------------------------------------------------

def moisture_multiplier(moisture_pct_wb: float) -> float:
    """Steele (1967) Appendix D moisture multiplier. Equals 1.0 at 25% w.b."""
    m_db = moisture_wb_to_db(moisture_pct_wb)
    return C.MM_A * (
        math.exp(C.MM_B * m_db ** C.MM_C) - C.MM_D * m_db + C.MM_E
    )


def temperature_multiplier(temperature_c: float, moisture_pct_wb: float) -> float:
    """Steele (1967) Appendix D temperature multiplier. Equals 1.0 at 15.6 C.

    Piecewise: the warm branches carry an extra term that only applies above
    the reference temperature and above 19% moisture.
    """
    base = C.MT_A * math.exp(
        C.MT_B * (C.MT_C_SLOPE * temperature_c + C.MT_C_INTERCEPT)
    )
    if temperature_c <= C.REFERENCE_TEMPERATURE_C or moisture_pct_wb <= 19.0:
        return base

    warm = math.exp(
        C.MT_WET_EXP_A * (C.MT_C_SLOPE * temperature_c - C.MT_WET_EXP_OFFSET)
    )
    if moisture_pct_wb <= 28.0:
        return base + C.MT_WET_SLOPE * (moisture_pct_wb - 19.0) * warm
    return base + C.MT_WET_CAP * warm


def damage_multiplier(damage_pct: float, dml_level: float = C.DML_THRESHOLD_PCT) -> float:
    """Steele (1967) Appendix D mechanical-damage multiplier.

    ``damage_pct`` is the percentage **by weight** of kernels with a ruptured
    seed coat -- Steele's definition, which is broader than the grain
    standards' "broken kernel". Equals 1.0 at 30% damage, and 2.08 at zero.

    Monotone decreasing over the whole valid range, which is what makes lot
    ranking robust to the coefficients being wrong.
    """
    try:
        a, b = C.MD_COEFFICIENTS_BY_DML_LEVEL[dml_level]
    except KeyError:
        raise ValueError(
            f"no published damage multiplier for DML level {dml_level}%; "
            f"available: {sorted(C.MD_COEFFICIENTS_BY_DML_LEVEL)}"
        ) from None
    if damage_pct < 0.0:
        raise ValueError(f"damage percentage cannot be negative, got {damage_pct}")
    return a * math.exp(b * damage_pct)


# --- The dry matter loss curve --------------------------------------------

def dry_matter_loss_pct(equivalent_hours: float) -> float:
    """Percent dry matter lost after ``equivalent_hours`` at reference conditions."""
    if equivalent_hours < 0.0:
        raise ValueError(f"equivalent hours cannot be negative, got {equivalent_hours}")
    return (
        C.DML_A * (math.exp(C.DML_B * equivalent_hours) - 1.0)
        + C.DML_C * equivalent_hours
    )


def equivalent_hours_for_dml(target_dml_pct: float) -> float:
    """Invert the DML curve: equivalent hours to reach ``target_dml_pct``.

    The curve is strictly increasing, so a bisection is exact enough and avoids
    depending on a solver library for four lines of arithmetic.
    """
    if target_dml_pct <= 0.0:
        raise ValueError(f"target DML must be positive, got {target_dml_pct}")
    lo, hi = 0.0, 1.0
    while dry_matter_loss_pct(hi) < target_dml_pct:
        hi *= 2.0
        if hi > 1e9:
            raise ValueError(f"target DML {target_dml_pct}% is unreachable")
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if dry_matter_loss_pct(mid) < target_dml_pct:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def days_to_dml(
    damage_pct: float,
    temperature_c: float,
    moisture_pct_wb: float,
    dml_level: float = C.DML_THRESHOLD_PCT,
) -> float:
    """Days for a lot at these conditions to lose ``dml_level`` percent dry matter.

    Steele's Equation 13 directly: ``t = t_R * MT * MM * MD``, with ``t_R``
    the reference time on page 108. The multipliers are divisors of equivalent
    time and therefore multipliers of real time -- the same relationship read
    the other way round, which is why worsening conditions shorten the answer
    here while lengthening equivalent hours elsewhere in this module.

    No dry-matter-loss curve is involved. This asks how long the whole trip to
    the threshold takes, not how loss accrues along the way, and Steele fitted
    the former directly.

    The answer is worth no more than Steele's standard error at that level --
    about 11% at the 0.5% threshold. Callers should carry
    ``C.STEELE_STANDARD_ERROR_PCT`` alongside it.
    """
    try:
        reference_hours = C.STEELE_REFERENCE_HOURS[dml_level]
    except KeyError:
        raise ValueError(
            f"no published reference time for DML level {dml_level}%; "
            f"available: {sorted(C.STEELE_REFERENCE_HOURS)}"
        ) from None

    hours = (
        reference_hours
        * temperature_multiplier(temperature_c, moisture_pct_wb)
        * moisture_multiplier(moisture_pct_wb)
        * damage_multiplier(damage_pct, dml_level)
    )
    return hours / 24.0


# --- Range checking --------------------------------------------------------

@dataclass(frozen=True)
class RangeCheck:
    """Which inputs sit inside the published validity ranges, and which do not."""

    temperature_ok: bool
    moisture_ok: bool
    damage_ok: bool
    notes: tuple[str, ...] = ()

    @property
    def all_ok(self) -> bool:
        return self.temperature_ok and self.moisture_ok and self.damage_ok


def check_ranges(temperature_c: float, moisture_pct_wb: float, damage_pct: float) -> RangeCheck:
    notes: list[str] = []

    t_lo, t_hi = C.MT_VALID_C
    temperature_ok = t_lo <= temperature_c <= t_hi
    if not temperature_ok:
        notes.append(
            f"temperature {temperature_c} C is outside Steele's tested "
            f"{t_lo}-{t_hi} C"
        )

    m_lo, m_hi = C.MM_VALID_WB_PCT
    moisture_ok = m_lo <= moisture_pct_wb <= m_hi
    if not moisture_ok:
        m_db = moisture_wb_to_db(moisture_pct_wb)
        notes.append(
            f"moisture {moisture_pct_wb}% w.b. ({m_db:.1f}% d.b.) is outside "
            f"the published {m_lo}-{m_hi}% wet basis"
        )

    d_lo, d_hi = C.MD_VALID_PCT
    damage_ok = d_lo <= damage_pct <= d_hi
    if not damage_ok:
        notes.append(
            f"damage {damage_pct}% is outside the published {d_lo}-{d_hi}%"
        )

    return RangeCheck(temperature_ok, moisture_ok, damage_ok, tuple(notes))


# --- The assessment --------------------------------------------------------

@dataclass(frozen=True)
class Assessment:
    """One lot's assessment. Ranking always populated; absolute may be None."""

    damage_pct: float
    temperature_c: float
    moisture_pct_wb: float

    mm: float
    mt: float
    md: float

    #: Equivalent hours accrued per real hour of storage. Higher degrades
    #: faster. This is the ranking key and it is always available.
    degradation_rate: float

    ranges: RangeCheck
    constants_verified: bool

    #: Days to the 0.5% dry-matter-loss threshold, from Steele's reference
    #: times. Populated whenever ``ranges.all_ok``; it does not wait on
    #: Thompson. This is the absolute figure the application shows.
    days_to_threshold: float | None = None

    #: Steele's standard error at the threshold level, in percent. Travels with
    #: ``days_to_threshold`` and is None whenever that is.
    days_to_threshold_error_pct: float | None = None

    #: The same quantity by the Thompson dry-matter-loss curve, populated only
    #: when ``ranges.all_ok`` and ``constants_verified``. Kept as a cross-check
    #: on the day Thompson is obtained, not as an output.
    days_to_threshold_thompson: float | None = None

    #: Why an absolute figure was withheld, if it was. Empty when one is
    #: shown -- the only thing that withholds it now is an out-of-range input.
    suppression_reasons: tuple[str, ...] = ()

    #: Standing caveats about the model itself. Always populated, never a
    #: reason anything was suppressed, and shown as a footnote rather than a
    #: warning. Conflating the two produced a screen that said the days figure
    #: was withheld while displaying it.
    model_notes: tuple[str, ...] = ()

    citations: tuple[str, ...] = field(
        default=(
            "Steele, J. L. (1967), PhD dissertation, Iowa State University, "
            "Appendix D -- moisture, temperature and mechanical-damage multipliers",
            "Steele, J. L. (1967), p. 108, Eq. 13 -- reference times and "
            "standard errors behind days to threshold",
            "Bern, C. J. et al. (2002), Appl. Eng. Agric. 18(6) -- 0.5% DML threshold",
        )
    )

    @property
    def mode(self) -> str:
        return "ranking+absolute" if self.days_to_threshold is not None else "ranking"


def assess(
    damage_pct: float,
    temperature_c: float,
    moisture_pct_wb: float,
    *,
    constants_verified: bool | None = None,
) -> Assessment:
    """Assess one lot.

    ``damage_pct`` is percent by weight of kernels with a ruptured seed coat.

    ``constants_verified`` defaults to the module-level gate and exists as a
    parameter only so tests can exercise the Thompson cross-check. It no longer
    affects ``days_to_threshold``. Application code should leave it alone.
    """
    verified = C.CONSTANTS_VERIFIED if constants_verified is None else constants_verified

    mm = moisture_multiplier(moisture_pct_wb)
    mt = temperature_multiplier(temperature_c, moisture_pct_wb)
    md = damage_multiplier(damage_pct)

    product = mm * mt * md
    if product <= 0.0:
        raise ValueError(
            f"multiplier product must be positive, got {product} "
            f"(MM={mm}, MT={mt}, MD={md})"
        )

    # tr = t / product, so equivalent hours accrue at 1/product per real hour.
    degradation_rate = 1.0 / product

    ranges = check_ranges(temperature_c, moisture_pct_wb, damage_pct)

    reasons: list[str] = list(ranges.notes)
    notes: list[str] = []

    days: float | None = None
    error_pct: float | None = None
    thompson_days: float | None = None
    if ranges.all_ok:
        days = C.STEELE_REFERENCE_HOURS[C.DML_THRESHOLD_PCT] * product / 24.0
        error_pct = C.STEELE_STANDARD_ERROR_PCT[C.DML_THRESHOLD_PCT]
        if verified:
            tr_star = equivalent_hours_for_dml(C.DML_THRESHOLD_PCT)
            thompson_days = tr_star * product / 24.0

    if not verified:
        notes.append(C.CONSTANTS_VERIFIED_NOTE)
    if days is not None:
        notes.append(
            f"Days to threshold is a Steele (1967) estimate and carries his "
            f"stated standard error of {error_pct}% at the "
            f"{C.DML_THRESHOLD_PCT}% dry-matter-loss level."
        )

    return Assessment(
        damage_pct=damage_pct,
        temperature_c=temperature_c,
        moisture_pct_wb=moisture_pct_wb,
        mm=mm,
        mt=mt,
        md=md,
        degradation_rate=degradation_rate,
        ranges=ranges,
        constants_verified=verified,
        days_to_threshold=days,
        days_to_threshold_error_pct=error_pct,
        days_to_threshold_thompson=thompson_days,
        suppression_reasons=tuple(reasons),
        model_notes=tuple(notes),
    )


def rank(assessments: Sequence[Assessment]) -> list[Assessment]:
    """Order lots fastest-degrading first. The primary output of the system."""
    return sorted(assessments, key=lambda a: a.degradation_rate, reverse=True)
