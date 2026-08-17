"""One manifest over both public datasets, and splits that hold out sessions.

The single most consequential number in the reviewed literature is Kumari et
al. (2026): holding out a different *acquisition session* rather than a random
sample of images cost roughly nine points of balanced accuracy on an otherwise
identical model and instrument. A random split does not measure generalisation,
it measures memorisation of an acquisition condition.

That finding is not hypothetical here. GrainSet ships a train/test split, and
it is a random one:

    43.7% of capture events appear on both sides of it,
    so 49.2% of all images sit in an event that straddles the boundary.

Training on the released split and reporting its test accuracy would therefore
report the optimistic number for nearly half the test set. This module exists
so that cannot happen by accident: the split is a property of the *session*,
derived from the capture timestamp encoded in every GrainSet filename, and a
test asserts that no session ever appears in two splits.

None of this is a criticism of GrainSet, which is the dataset this project
depends on. A random split is the right choice for the benchmark the authors
were publishing. It is the wrong choice for estimating deployment performance,
which is what this project needs.
"""

from __future__ import annotations

import csv
import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from src.mapping.classes import CORNSEEDS, GRAINSET_MAIZE, KernelClass

#: `Grainset_maize_<YYYY-MM-DD-HH-MM-SS>_<kernel index>_p600s.png`.
#: The timestamp is the acquisition event and therefore the split unit.
GRAINSET_FILENAME = re.compile(
    r"Grainset_maize_(?P<session>\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})"
    r"_(?P<index>\d+)_p(?P<pixels>\d+)s\.png$"
)

#: GrainSet's directories are `0_NOR`, `5_BN` and so on.
GRAINSET_DIRECTORY = re.compile(r"^\d+_(?P<label>.+)$")

SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Sample:
    """One labelled kernel image."""

    path: Path
    dataset: str
    source_label: str
    kernel_class: KernelClass

    #: The acquisition event this image belongs to. Every split decision is
    #: made on this value and never on the image itself.
    session: str

    #: Pixel-level mask, where the dataset provides one.
    mask: Path | None = None

    @property
    def class_name(self) -> str:
        return self.kernel_class.name


def _assign(session: str, val_fraction: float, test_fraction: float) -> str:
    """Deterministically place a session in a split.

    Hashing the session id rather than shuffling means the assignment is stable
    across machines and runs, and survives new data being added -- an existing
    session never migrates just because the corpus grew.
    """
    digest = hashlib.sha256(session.encode()).digest()
    x = int.from_bytes(digest[:8], "big") / 2**64
    if x < test_fraction:
        return "test"
    if x < test_fraction + val_fraction:
        return "val"
    return "train"


# --- GrainSet --------------------------------------------------------------

def load_grainset(root: Path) -> list[Sample]:
    """Index the extracted GrainSet maize subset.

    ``root`` is the directory holding ``train/``, ``test/`` and ``mask/``. The
    released train/test division is deliberately ignored -- see the module
    docstring -- and only the class directory and the filename are read.
    """
    root = Path(root)
    masks = {p.name: p for p in (root / "mask").glob("*.png")}

    samples: list[Sample] = []
    for split_dir in ("train", "test"):
        for class_dir in sorted((root / split_dir).iterdir()):
            if not class_dir.is_dir():
                continue
            m = GRAINSET_DIRECTORY.match(class_dir.name)
            if m is None:
                raise ValueError(f"unexpected GrainSet directory {class_dir.name!r}")
            label = m.group("label")
            try:
                kernel_class = GRAINSET_MAIZE[label]
            except KeyError:
                raise ValueError(
                    f"GrainSet label {label!r} has no mapping; "
                    f"known: {sorted(GRAINSET_MAIZE)}"
                ) from None

            for image in sorted(class_dir.glob("*.png")):
                fm = GRAINSET_FILENAME.match(image.name)
                if fm is None:
                    raise ValueError(f"unparsable GrainSet filename {image.name!r}")
                samples.append(
                    Sample(
                        path=image,
                        dataset="grainset",
                        source_label=label,
                        kernel_class=kernel_class,
                        session=fm.group("session"),
                        mask=masks.get(image.name),
                    )
                )
    return samples


# --- Corn Seeds ------------------------------------------------------------

def load_cornseeds(root: Path) -> list[Sample]:
    """Index the labelled portion of the Corn Seeds Dataset.

    The ``test`` folder ships placeholder labels rather than ground truth, so
    only ``train.csv`` is read and validation is carved from it.

    This dataset has no acquisition timestamp. The kernel id is used as the
    session instead, which is weaker -- it prevents the two views of one kernel
    from straddling a split, but cannot prevent an acquisition condition from
    doing so. Recorded honestly rather than papered over.
    """
    root = Path(root)
    samples: list[Sample] = []
    with (root / "train.csv").open() as fh:
        for row in csv.DictReader(fh):
            label = row["label"]
            try:
                kernel_class = CORNSEEDS[label]
            except KeyError:
                raise ValueError(
                    f"Corn Seeds label {label!r} has no mapping; "
                    f"known: {sorted(CORNSEEDS)}"
                ) from None
            samples.append(
                Sample(
                    path=root / row["image"],
                    dataset="cornseeds",
                    source_label=label,
                    kernel_class=kernel_class,
                    session=f"cornseeds-{row['seed_id']}",
                )
            )
    return samples


# --- The manifest ----------------------------------------------------------

@dataclass(frozen=True)
class Manifest:
    """Samples plus the split each one landed in.

    Assignment is per image so that both split strategies are representable by
    the same object and can be compared directly. Which strategy produced it is
    recorded in ``strategy``, and :meth:`straddling_sessions` is the diagnostic
    that tells the two apart on the data rather than on the label.
    """

    samples: tuple[Sample, ...]
    strategy: str
    _assignment: dict[Path, str]

    def split(self, name: str) -> list[Sample]:
        if name not in SPLITS:
            raise ValueError(f"unknown split {name!r}; expected one of {SPLITS}")
        return [s for s in self.samples if self._assignment[s.path] == name]

    def split_of(self, sample: Sample) -> str:
        return self._assignment[sample.path]

    def class_counts(self, name: str) -> dict[str, int]:
        return dict(Counter(s.class_name for s in self.split(name)))

    @property
    def sessions(self) -> set[str]:
        return {s.session for s in self.samples}

    def sessions_in(self, name: str) -> set[str]:
        return {s.session for s in self.split(name)}

    def straddling_sessions(self) -> set[str]:
        """Sessions whose images landed in more than one split.

        Empty for a session split, by construction. Large for a random split --
        which is the whole point of measuring it.
        """
        seen: dict[str, set[str]] = {}
        for s in self.samples:
            seen.setdefault(s.session, set()).add(self._assignment[s.path])
        return {session for session, splits in seen.items() if len(splits) > 1}

    def leakage_pct(self) -> float:
        """Share of images sitting in a session that straddles the split."""
        straddling = self.straddling_sessions()
        if not self.samples:
            return 0.0
        n = sum(1 for s in self.samples if s.session in straddling)
        return 100.0 * n / len(self.samples)


def _validate(samples: list[Sample], val_fraction: float, test_fraction: float) -> None:
    if not samples:
        raise ValueError("cannot build a manifest from no samples")
    if val_fraction < 0.0 or test_fraction < 0.0:
        raise ValueError(
            f"fractions cannot be negative, got val={val_fraction} test={test_fraction}"
        )
    if val_fraction + test_fraction >= 1.0:
        raise ValueError(
            f"val + test fractions must leave room for training, got "
            f"{val_fraction} + {test_fraction}"
        )


def by_session(sample: Sample) -> str:
    """One capture event -- a single tray moment, seconds long."""
    return sample.session


def by_day(sample: Sample) -> str:
    """The calendar day of capture.

    A much coarser unit than a session, and the one that actually tests what
    Kumari et al. (2026) tested. A GrainSet session is a few seconds holding
    several *different* kernels, so splitting on it still puts different
    kernels either side and leaves nothing session-specific to memorise --
    measured on this corpus, session and random splits score within 0.2 points
    of each other. Holding out whole days separates different rig warm-ups,
    calibrations and operator sessions, which is where acquisition shift
    actually lives.

    Corn Seeds carries no capture date, so its samples keep their kernel-level
    key. Read the GrainSet figure when interpreting a day split.
    """
    return sample.session[:10] if sample.dataset == "grainset" else sample.session


def build(
    samples: list[Sample],
    *,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    key=by_session,
) -> Manifest:
    """Split by a grouping key, so no group spans two splits.

    ``key`` defaults to the capture session; pass :func:`by_day` for the
    stricter holdout.
    """
    _validate(samples, val_fraction, test_fraction)
    return Manifest(
        samples=tuple(samples),
        strategy=getattr(key, "__name__", "custom"),
        _assignment={
            s.path: _assign(key(s), val_fraction, test_fraction) for s in samples
        },
    )


def random_split_baseline(
    samples: list[Sample], *, val_fraction: float = 0.15, test_fraction: float = 0.15
) -> Manifest:
    """A deliberately leaky image-level split, for measuring the inflation.

    Not an alternative to :func:`build`. It exists so the paper can report both
    numbers from the same code path and show the gap, which is the honest way
    to present a figure most of the literature reports without qualification.
    Never select a model with it.
    """
    _validate(samples, val_fraction, test_fraction)
    return Manifest(
        samples=tuple(samples),
        strategy="random",
        _assignment={
            s.path: _assign(str(s.path), val_fraction, test_fraction) for s in samples
        },
    )
