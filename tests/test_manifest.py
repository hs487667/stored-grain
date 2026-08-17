"""Tests for the manifest and the split discipline.

The failure these prevent is the quiet one: a model that scores well because
its test set shares an acquisition condition with its training set.
"""

from pathlib import Path

import pytest

from src.data.manifest import (
    GRAINSET_FILENAME,
    Sample,
    build,
    random_split_baseline,
)
from src.mapping.classes import CORNSEEDS, GRAINSET_MAIZE


def _sample(session: str, index: int, cls="sound", dataset="grainset") -> Sample:
    from src.mapping.classes import CLASSES

    return Sample(
        path=Path(f"/fake/{session}_{index}.png"),
        dataset=dataset,
        source_label="NOR",
        kernel_class=CLASSES[cls],
        session=session,
    )


def _corpus(n_sessions=200, per_session=6) -> list[Sample]:
    return [
        _sample(f"2021-03-{i:02d}-10-00-{i:02d}", k)
        for i in range(n_sessions)
        for k in range(per_session)
    ]


# --- The split unit --------------------------------------------------------

def test_session_split_never_straddles():
    m = build(_corpus())
    assert m.straddling_sessions() == set()
    assert m.leakage_pct() == 0.0


def test_random_split_leaks_and_we_can_measure_it():
    # The comparison the paper reports: same corpus, same code, two splits.
    m = random_split_baseline(_corpus())
    assert m.straddling_sessions()
    assert m.leakage_pct() > 50.0


def test_splits_partition_the_corpus():
    samples = _corpus()
    m = build(samples)
    total = sum(len(m.split(s)) for s in ("train", "val", "test"))
    assert total == len(samples)


def test_every_split_is_populated():
    m = build(_corpus())
    for name in ("train", "val", "test"):
        assert m.split(name), f"{name} split is empty"


def test_assignment_is_stable_across_runs():
    samples = _corpus()
    assert build(samples)._assignment == build(samples)._assignment


def test_adding_data_does_not_move_existing_sessions():
    # Hashing rather than shuffling: a session's split is a property of its
    # identity, so growing the corpus cannot silently reshuffle the test set.
    small = _corpus(n_sessions=50)
    large = small + _corpus(n_sessions=200)[50 * 6 :]
    a, b = build(small), build(large)
    for s in small:
        assert a.split_of(s) == b.split_of(s)


def test_split_proportions_are_roughly_as_asked():
    m = build(_corpus(n_sessions=1000), val_fraction=0.2, test_fraction=0.2)
    n = len(m.samples)
    assert 0.15 < len(m.split("test")) / n < 0.25
    assert 0.15 < len(m.split("val")) / n < 0.25


def test_unknown_split_name_is_rejected():
    with pytest.raises(ValueError, match="unknown split"):
        build(_corpus()).split("holdout")


def test_fractions_must_leave_training_data():
    with pytest.raises(ValueError, match="leave room for training"):
        build(_corpus(), val_fraction=0.6, test_fraction=0.5)


def test_empty_corpus_is_rejected():
    with pytest.raises(ValueError, match="no samples"):
        build([])


# --- Filename parsing ------------------------------------------------------

def test_grainset_filename_yields_the_session():
    m = GRAINSET_FILENAME.match("Grainset_maize_2020-10-31-22-59-21_3_p600s.png")
    assert m is not None
    assert m.group("session") == "2020-10-31-22-59-21"
    assert m.group("index") == "3"


def test_grainset_filename_rejects_other_species():
    assert GRAINSET_FILENAME.match("Grainset_wheat_2020-10-31-22-59-21_3_p600s.png") is None


# --- The vocabularies line up with the directories on disk -----------------

def test_every_grainset_directory_label_has_a_mapping():
    on_disk = {"NOR", "F&S", "SD", "MY", "AP", "BN", "HD", "IM"}
    assert on_disk == set(GRAINSET_MAIZE)


def test_every_cornseeds_label_has_a_mapping():
    on_disk = {"pure", "broken", "discolored", "silkcut"}
    assert on_disk == set(CORNSEEDS)
