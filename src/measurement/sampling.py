"""Sampling error, resolution limits, and the fragmentation trap.

A damage percentage read off one photograph is an estimate from a finite
sample, and the system's primary output is an *ordering* of lots. Ordering two
lots whose damage differs by less than the measurement noise is not a
conservative answer -- it is a coin flip presented as a result. This module
exists so that the ranking can decline to separate lots it cannot actually
separate.

Two independent error sources, and they behave differently:

* **Counting error** shrinks as the square root of the number of kernels
  photographed. It is unbiased, it is computable in advance, and it sets a
  hard floor on the resolution of the ranking.
* **Classifier error** does not shrink with sample size. It biases the
  estimate. Crucially it is *affine* in the true damage, so it preserves
  ordering among lots stored under identical conditions -- which is why the
  ranking is defensible long before the classifier is good.

The fragmentation functions are here rather than in the mapping layer because
they describe an error in *ground truth construction*, not in classification.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

#: Mass of a broken fragment relative to a sound kernel, from the GrainSet
#: maize weight field (109.1 / 297.2). Shared with the mapping layer, restated
#: here because the calibration protocol uses it independently of any
#: classifier.
FRAGMENT_RELATIVE_MASS = 0.367

#: Sound shelled-maize kernel mass in grams. Order-of-magnitude figure used
#: only to convert a target kernel count into a sample mass on the scale.
SOUND_KERNEL_MASS_G = 0.30


def _z(confidence: float) -> float:
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    return NormalDist().inv_cdf(1.0 - (1.0 - confidence) / 2.0)


# --- Counting error --------------------------------------------------------

def standard_error_pct(damage_pct: float, kernels_counted: int) -> float:
    """Standard error of a damage percentage from counting ``n`` kernels.

    Binomial: each kernel is damaged or not, independently.
    """
    if not 0.0 <= damage_pct <= 100.0:
        raise ValueError(f"damage must be in [0, 100] %, got {damage_pct}")
    if kernels_counted <= 0:
        raise ValueError(f"must count at least one kernel, got {kernels_counted}")
    p = damage_pct / 100.0
    return 100.0 * math.sqrt(p * (1.0 - p) / kernels_counted)


def minimum_resolvable_gap_pct(
    damage_pct: float, kernels_counted: int, confidence: float = 0.95
) -> float:
    """Smallest damage difference two photographs can distinguish.

    Two independent estimates, so the difference carries ``sqrt(2)`` times the
    single-measurement error. Below this gap the ranking of the two lots is
    not supported by the data.
    """
    se = standard_error_pct(damage_pct, kernels_counted)
    return _z(confidence) * math.sqrt(2.0) * se


def kernels_required(
    damage_pct: float, target_gap_pct: float, confidence: float = 0.95
) -> int:
    """Kernels per photograph needed to resolve a gap of ``target_gap_pct``."""
    if target_gap_pct <= 0.0:
        raise ValueError(f"target gap must be positive, got {target_gap_pct}")
    p = damage_pct / 100.0
    z = _z(confidence)
    n = 2.0 * (z / (target_gap_pct / 100.0)) ** 2 * p * (1.0 - p)
    return math.ceil(n)


def sample_mass_g(kernels: int) -> float:
    """Roughly how much maize to weigh out to get ``kernels`` in frame."""
    if kernels <= 0:
        raise ValueError(f"kernels must be positive, got {kernels}")
    return kernels * SOUND_KERNEL_MASS_G


# --- Classifier error ------------------------------------------------------

def biased_damage_pct(
    true_damage_pct: float, false_positive_rate: float, false_negative_rate: float
) -> float:
    """Damage a classifier reports, given its per-kernel error rates.

    Affine in the truth: ``d' = FPR + d(1 - FNR - FPR)``. The slope is positive
    whenever ``FPR + FNR < 1`` -- that is, for any classifier better than a
    coin -- which is the whole reason lot ordering survives a mediocre model.
    """
    for name, r in (("FPR", false_positive_rate), ("FNR", false_negative_rate)):
        if not 0.0 <= r <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {r}")
    d = true_damage_pct / 100.0
    return 100.0 * (
        d * (1.0 - false_negative_rate) + (1.0 - d) * false_positive_rate
    )


def unbias_damage_pct(
    measured_damage_pct: float, false_positive_rate: float, false_negative_rate: float
) -> float:
    """Invert :func:`biased_damage_pct`: the damage that would produce this reading.

    The forward map is affine, so the inverse is too: ``d = (d' - FPR) / (1 -
    FNR - FPR)``. It undoes the compression a two-sided classifier applies --
    overstating clean samples by its false positives and understating damaged
    ones by its false negatives -- which is what pulls every reading toward the
    middle of the range.

    This changes the absolute figure only. The map is monotone, so a corrected
    reading ranks exactly where the raw one did; the correction earns its place
    in the damage percentage the user is shown, not in the ordering.

    Result is clamped to [0, 100]. Counting noise can put a reading below the
    false-positive floor, which inverts to a negative damage -- a percentage
    below zero is not a more precise measurement, it is an impossible one.
    """
    if not preserves_ordering(false_positive_rate, false_negative_rate):
        raise ValueError(
            "cannot unbias a classifier no better than a coin: "
            f"FPR {false_positive_rate} + FNR {false_negative_rate} >= 1"
        )
    slope = 1.0 - false_negative_rate - false_positive_rate
    corrected = (measured_damage_pct / 100.0 - false_positive_rate) / slope
    return 100.0 * min(max(corrected, 0.0), 1.0)


@dataclass(frozen=True)
class DamageCalibration:
    """A classifier's measured damage error rates, and the correction they buy.

    Estimated by running the classifier over kernels held out from its own
    training *and* from whatever it is later corrected on -- an error rate
    measured on the evaluation set would flatter the correction rather than
    test it.
    """

    false_positive_rate: float
    false_negative_rate: float

    #: Kernels the rates were measured over. Rates from a handful of kernels
    #: are noise, and the correction is only as trustworthy as this number.
    kernels: int

    #: Which damage percentage these rates describe. ``count`` for rates from a
    #: per-kernel confusion, ``mass`` for a line fitted against weighed trays.
    #: Applying one to the other's basis is a real error where the damaged
    #: population is fragments, since the two then live on different scales.
    basis: str = "count"

    def __post_init__(self) -> None:
        if self.basis not in ("count", "mass"):
            raise ValueError(f"basis must be 'count' or 'mass', got {self.basis!r}")

    @classmethod
    def from_affine(
        cls,
        *,
        intercept_pct: float,
        slope: float,
        kernels: int,
        basis: str = "mass",
    ) -> "DamageCalibration":
        """Rates from a line fitted to measured-against-true damage.

        Fitting the whole pipeline rather than the classifier alone. The
        localiser's crops are not the dataset's images -- they are cut from a
        crowded tray, at whatever scale the kernels happened to occupy -- so the
        classifier's error rates on clean dataset images understate what it
        does downstream. A line fitted to trays of known damage absorbs every
        stage's contribution at once.

        The fitted line has the same two degrees of freedom as the per-kernel
        map, so it is stored the same way: the intercept is what a sound tray
        reads, and the slope is what a fully damaged one loses.
        """
        if slope <= 0.0:
            raise ValueError(
                f"slope must be positive to invert; got {slope}. A flat or "
                "falling line means the reading carries no damage information."
            )
        fpr = intercept_pct / 100.0
        return cls(
            false_positive_rate=fpr,
            false_negative_rate=1.0 - slope - fpr,
            kernels=kernels,
            basis=basis,
        )

    @classmethod
    def from_counts(
        cls,
        *,
        damaged_total: int,
        damaged_missed: int,
        sound_total: int,
        sound_flagged: int,
    ) -> "DamageCalibration":
        """Rates from a binary damaged/not-damaged confusion count."""
        if damaged_total <= 0 or sound_total <= 0:
            raise ValueError("need both damaged and sound kernels to calibrate")
        if not 0 <= damaged_missed <= damaged_total:
            raise ValueError(f"{damaged_missed} missed of {damaged_total} damaged")
        if not 0 <= sound_flagged <= sound_total:
            raise ValueError(f"{sound_flagged} flagged of {sound_total} sound")
        return cls(
            false_positive_rate=sound_flagged / sound_total,
            false_negative_rate=damaged_missed / damaged_total,
            kernels=damaged_total + sound_total,
        )

    def correct(self, measured_damage_pct: float) -> float:
        """The damage this reading implies, with the measured bias removed."""
        return unbias_damage_pct(
            measured_damage_pct, self.false_positive_rate, self.false_negative_rate
        )

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "false_positive_rate": self.false_positive_rate,
            "false_negative_rate": self.false_negative_rate,
            "kernels": self.kernels,
            "basis": self.basis,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DamageCalibration":
        return cls(
            false_positive_rate=float(data["false_positive_rate"]),
            false_negative_rate=float(data["false_negative_rate"]),
            kernels=int(data["kernels"]),
            basis=str(data.get("basis", "count")),
        )


def preserves_ordering(false_positive_rate: float, false_negative_rate: float) -> bool:
    """Whether these error rates leave lot ordering intact.

    True for every classifier better than random. Holds only for lots at equal
    temperature and moisture: when those differ between lots, damage is no
    longer the sole ordering term and a biased estimate can reorder pairs.
    """
    return false_positive_rate + false_negative_rate < 1.0


# --- Fragmentation ---------------------------------------------------------

def count_pct_from_mass_pct(mass_pct: float, fragments_per_kernel: float) -> float:
    """What a counter sees when a weighed mass fraction has been shattered.

    Calibration mixtures are built by weighing out a mass fraction of damaged
    maize. Cracking conserves mass but not object count: split each kernel into
    ``fragments_per_kernel`` pieces and the camera counts every piece, so the
    counted fraction runs ahead of the weighed one -- a 10% mixture reads as
    25% at three pieces per kernel.

    ``fragments_per_kernel = 1`` is the target: kernels cracked under steady
    pressure until the pericarp ruptures, staying in one piece. There the two
    bases agree and the mixture means what the scale said it meant.
    """
    if fragments_per_kernel < 1.0:
        raise ValueError(
            f"a damaged kernel is at least one piece, got {fragments_per_kernel}"
        )
    if not 0.0 <= mass_pct <= 100.0:
        raise ValueError(f"mass percentage must be in [0, 100], got {mass_pct}")

    # Per 100 g of mixture, in units of one sound kernel's mass: the damaged
    # material weighs mass_pct and is therefore mass_pct kernels' worth, each
    # now presenting as `fragments_per_kernel` separate objects.
    n_sound = 100.0 - mass_pct
    n_damaged = mass_pct * fragments_per_kernel
    return 100.0 * n_damaged / (n_sound + n_damaged)


def mass_pct_from_count_pct(count_pct: float, fragments_per_kernel: float) -> float:
    """Recover the mass fraction from a counted one. Inverse of the above.

    This is the correction to apply when fragmentation is known and cannot be
    eliminated -- but eliminating it is strictly better, because
    ``fragments_per_kernel`` has to be measured to be used, and measuring it
    per lot is more work than cracking the kernels properly in the first place.
    """
    if fragments_per_kernel < 1.0:
        raise ValueError(
            f"a damaged kernel is at least one piece, got {fragments_per_kernel}"
        )
    if not 0.0 <= count_pct <= 100.0:
        raise ValueError(f"count percentage must be in [0, 100], got {count_pct}")
    c = count_pct / 100.0
    return 100.0 * c / (fragments_per_kernel - c * (fragments_per_kernel - 1.0))


@dataclass(frozen=True)
class FragmentAudit:
    """Result of counting pieces in a pilot cracking batch."""

    kernels_cracked: int
    pieces_counted: int
    mass_before_g: float
    mass_after_g: float

    @property
    def fragments_per_kernel(self) -> float:
        return self.pieces_counted / self.kernels_cracked

    @property
    def fines_loss_pct(self) -> float:
        """Mass that left the mortar as dust. Silently shifts every mixture."""
        return 100.0 * (self.mass_before_g - self.mass_after_g) / self.mass_before_g

    @property
    def acceptable(self) -> bool:
        """Whether this cracking technique may be used to build the mixtures.

        Thresholds are deliberately tight. Fragmentation is not averaged away
        by making more mixtures -- it biases every one of them in the same
        direction, and it is the one error in the chain that the weighed
        ground truth cannot catch by construction.
        """
        return self.fragments_per_kernel <= 1.15 and self.fines_loss_pct <= 2.0

    @property
    def verdict(self) -> str:
        if self.acceptable:
            return "acceptable"
        problems = []
        if self.fragments_per_kernel > 1.15:
            problems.append(
                f"{self.fragments_per_kernel:.2f} pieces per kernel -- crack by "
                f"steady pressure, not impact"
            )
        if self.fines_loss_pct > 2.0:
            problems.append(
                f"{self.fines_loss_pct:.1f}% mass lost as fines -- crack over a "
                f"tray and return the dust to the sample"
            )
        return "; ".join(problems)


# --- Putting it together ---------------------------------------------------

@dataclass(frozen=True)
class Resolution:
    """Whether a measurement can support the ordering it is being asked for."""

    damage_pct: float
    kernels_counted: int
    confidence: float
    standard_error_pct: float
    minimum_resolvable_gap_pct: float

    def can_separate(self, other_damage_pct: float) -> bool:
        return abs(other_damage_pct - self.damage_pct) >= self.minimum_resolvable_gap_pct


def resolution(
    damage_pct: float, kernels_counted: int, confidence: float = 0.95
) -> Resolution:
    return Resolution(
        damage_pct=damage_pct,
        kernels_counted=kernels_counted,
        confidence=confidence,
        standard_error_pct=standard_error_pct(damage_pct, kernels_counted),
        minimum_resolvable_gap_pct=minimum_resolvable_gap_pct(
            damage_pct, kernels_counted, confidence
        ),
    )
