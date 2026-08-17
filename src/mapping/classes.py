"""The mapping layer: visual class -> the `MD` term, or not.

This module is the project's conceptual contribution, and it exists because
the deterioration model's damage term is not "anything wrong with the kernel".
Steele (1967) defines it exactly once, in the methods section:

    "Mechanical damage was determined for each test as the percent by weight
     of kernels with broken seed coats."

Three consequences follow, and every one of them is a way to get a
plausible-looking wrong number:

1. **Mechanical only.** A mouldy, sprouted or insect-eaten kernel has not had
   its seed coat mechanically ruptured. Feeding those into `MD` inflates the
   damage term with deterioration that the multiplier was never fitted on.
   They are counted and reported, never passed to the equation.

2. **Seed-coat rupture, not fragmentation.** Steele's definition is *broader*
   than the grain standards' "broken kernel": a kernel that is cracked but
   still whole counts. This is the opposite of the usual failure -- most
   datasets under-count, not over-count.

3. **By weight, not by count.** A classifier counts kernels. The equation
   wants mass. Those differ whenever the damaged population has a different
   mean mass from the sound one, which for fragments it emphatically does --
   see ``RELATIVE_MASS`` below.

Dataset label vocabularies are translated into this project's classes here and
nowhere else, so that a dataset's labelling convention can never leak into the
physics.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum


class Admissibility(Enum):
    """Whether a class may enter the mechanical-damage term."""

    #: Seed coat mechanically ruptured. Enters `MD`.
    MECHANICAL = "mechanical"

    #: Deterioration of biological origin. Reported to the user, never in `MD`.
    BIOLOGICAL = "biological"

    #: Undamaged kernel. In the denominator, not the numerator.
    SOUND = "sound"

    #: Not a kernel at all. Excluded from both numerator and denominator.
    NON_KERNEL = "non_kernel"


@dataclass(frozen=True)
class KernelClass:
    name: str
    admissibility: Admissibility
    definition: str

    #: Mass of an average kernel of this class relative to a sound kernel.
    #: Used to convert a counted fraction into the mass fraction the equation
    #: requires. ``None`` for non-kernel material, which is weighed out of the
    #: sample rather than modelled.
    relative_mass: float | None


# Relative masses are anchored on the GrainSet maize subset's per-kernel weight
# field: sound kernels average 297.2 against 109.1 for the fragmented class,
# a ratio of 0.367. A cracked-but-intact kernel has lost no material and is
# therefore 1.0 -- which is precisely why fragments and cracks cannot share a
# mass weight even though they share an admissibility.
SOUND = KernelClass(
    "sound", Admissibility.SOUND,
    "Intact kernel, seed coat unbroken.", 1.0,
)
SEED_COAT_CRACKED = KernelClass(
    "seed_coat_cracked", Admissibility.MECHANICAL,
    "Pericarp split or cracked, kernel otherwise whole. Steele's definition "
    "in its purest form, and the case the grain standards miss.", 1.0,
)
FRAGMENT = KernelClass(
    "fragment", Admissibility.MECHANICAL,
    "Kernel broken into pieces. Also a ruptured seed coat, but carrying a "
    "fraction of the mass.", 0.367,
)
MOULD_SUSPECT = KernelClass(
    "mould_suspect", Admissibility.BIOLOGICAL,
    "Visible discolouration consistent with fungal growth. A visual class, "
    "not a diagnosis -- no species and no mycotoxin is claimed.", 1.0,
)
INSECT_DAMAGED = KernelClass(
    "insect_damaged", Admissibility.BIOLOGICAL,
    "Boring holes or feeding galleries.", 1.0,
)
SPROUTED = KernelClass(
    "sprouted", Admissibility.BIOLOGICAL,
    "Germination has begun.", 1.0,
)
HEATED = KernelClass(
    "heated", Admissibility.BIOLOGICAL,
    "Heat damage from drying or from storage self-heating.", 1.0,
)
DISCOLOURED = KernelClass(
    "discoloured", Admissibility.BIOLOGICAL,
    "Off-colour without a mechanical break. Cause not determined visually.", 1.0,
)
IMPURITY = KernelClass(
    "impurity", Admissibility.NON_KERNEL,
    "Chaff, stones, dust, foreign matter.", None,
)

CLASSES: dict[str, KernelClass] = {
    c.name: c
    for c in (
        SOUND, SEED_COAT_CRACKED, FRAGMENT, MOULD_SUSPECT,
        INSECT_DAMAGED, SPROUTED, HEATED, DISCOLOURED, IMPURITY,
    )
}


# --- Dataset label vocabularies -------------------------------------------

#: GrainSet maize subset, ISO 5527 categories.
#:
#: `BN` maps to ``fragment``, NOT to ``seed_coat_cracked``, and the distinction
#: is load-bearing. The dataset's own weight field settles it: `BN` kernels
#: average 109.1 against 297.2 for `NOR`, so they are broken pieces. A cracked
#: but intact kernel would weigh close to 297 and there is no such population
#: in the dataset at all. Training a damage classifier on GrainSet alone
#: therefore teaches it to find fragments and leaves Steele's cracked-intact
#: case entirely unrepresented.
GRAINSET_MAIZE: dict[str, KernelClass] = {
    "NOR": SOUND,
    "BN": FRAGMENT,
    "SD": SPROUTED,
    "AP": INSECT_DAMAGED,
    "F&S": MOULD_SUSPECT,
    "HD": HEATED,
    "MY": MOULD_SUSPECT,
    "IM": IMPURITY,
}

#: Corn Seeds Dataset. `silkcut` -- a lateral split in the pericarp of an
#: otherwise whole kernel -- is the only public source of Steele's
#: cracked-but-intact case, which is why this dataset carries weight in the
#: training mix out of proportion to its size.
CORNSEEDS: dict[str, KernelClass] = {
    "pure": SOUND,
    "broken": FRAGMENT,
    "silkcut": SEED_COAT_CRACKED,
    "discolored": DISCOLOURED,
}


def translate(counts: dict[str, int], vocabulary: dict[str, KernelClass]) -> dict[str, int]:
    """Convert dataset-label counts into project-class counts."""
    out: dict[str, int] = {}
    for label, n in counts.items():
        try:
            cls = vocabulary[label]
        except KeyError:
            raise ValueError(
                f"label {label!r} is not in this vocabulary; "
                f"known: {sorted(vocabulary)}"
            ) from None
        out[cls.name] = out.get(cls.name, 0) + n
    return out


# --- The measurement -------------------------------------------------------

@dataclass(frozen=True)
class DamageMeasurement:
    """What a classified sample says, split the way the physics requires."""

    #: Percent by weight of kernels with a ruptured seed coat. The `MD` input.
    mechanical_damage_mass_pct: float

    #: The same quantity by count. Reported alongside because it is what the
    #: classifier natively produces and the two diverge under fragmentation.
    mechanical_damage_count_pct: float

    #: Biological deterioration, by count, per class. Shown to the user and
    #: withheld from the equation.
    biological_count_pct: dict[str, float]

    kernels_counted: int
    non_kernel_count: int

    @property
    def count_to_mass_ratio(self) -> float:
        """How far the two bases diverge. 1.0 means no fragmentation at all.

        Well below 1.0 means the damaged population is mostly fragments, and a
        pipeline that fed the count straight into `MD` would be overstating
        damage by this factor.
        """
        if self.mechanical_damage_count_pct == 0.0:
            return 1.0
        return self.mechanical_damage_mass_pct / self.mechanical_damage_count_pct


def damage_confusion(pairs: Iterable[tuple[str, str]]) -> dict[str, int]:
    """Collapse ``(true, predicted)`` class names into a damaged/sound confusion.

    The classifier has nine classes but the damage term has one boundary: a
    kernel either carries mechanical damage or it does not. Error rates have to
    be counted across *that* boundary, because confusing a fragment for a crack
    costs the damage percentage nothing while confusing either for a sound
    kernel costs it everything.

    Non-kernel material is excluded from the base, matching :func:`measure`,
    which sieves it out of both sides of the ratio. A kernel *predicted* to be
    non-kernel is dropped before it reaches the damage count, so it reads as
    undamaged -- the same thing the pipeline's debris threshold does to it.
    """
    counts = {
        "damaged_total": 0,
        "damaged_missed": 0,
        "sound_total": 0,
        "sound_flagged": 0,
    }
    for true_name, predicted_name in pairs:
        for name in (true_name, predicted_name):
            if name not in CLASSES:
                raise ValueError(f"unknown class {name!r}; known: {sorted(CLASSES)}")

        if CLASSES[true_name].admissibility is Admissibility.NON_KERNEL:
            continue

        called_damaged = (
            CLASSES[predicted_name].admissibility is Admissibility.MECHANICAL
        )
        if CLASSES[true_name].admissibility is Admissibility.MECHANICAL:
            counts["damaged_total"] += 1
            counts["damaged_missed"] += not called_damaged
        else:
            counts["sound_total"] += 1
            counts["sound_flagged"] += called_damaged
    return counts


def measure(class_counts: dict[str, int]) -> DamageMeasurement:
    """Turn per-class kernel counts into the damage figure the model wants.

    ``class_counts`` is keyed by project class name -- run dataset labels
    through :func:`translate` first.

    Non-kernel material is excluded from both numerator and denominator: it is
    removed from the sample by sieving before weighing, so including it would
    dilute a percentage that the equation defines over grain alone.
    """
    for name in class_counts:
        if name not in CLASSES:
            raise ValueError(f"unknown class {name!r}; known: {sorted(CLASSES)}")
    if any(n < 0 for n in class_counts.values()):
        raise ValueError(f"counts cannot be negative: {class_counts}")

    kernels = 0
    non_kernel = 0
    mech_count = 0
    mech_mass = 0.0
    total_mass = 0.0
    bio_counts: dict[str, int] = {}

    for name, n in class_counts.items():
        cls = CLASSES[name]
        if cls.admissibility is Admissibility.NON_KERNEL:
            non_kernel += n
            continue

        kernels += n
        mass = n * cls.relative_mass
        total_mass += mass

        if cls.admissibility is Admissibility.MECHANICAL:
            mech_count += n
            mech_mass += mass
        elif cls.admissibility is Admissibility.BIOLOGICAL:
            bio_counts[name] = bio_counts.get(name, 0) + n

    if kernels == 0:
        raise ValueError("sample contains no kernels")

    return DamageMeasurement(
        mechanical_damage_mass_pct=100.0 * mech_mass / total_mass,
        mechanical_damage_count_pct=100.0 * mech_count / kernels,
        biological_count_pct={k: 100.0 * v / kernels for k, v in bio_counts.items()},
        kernels_counted=kernels,
        non_kernel_count=non_kernel,
    )
