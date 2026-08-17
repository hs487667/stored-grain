# Capture App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A phone-facing web app that photographs trays of shelled maize, measures the percentage of kernels that are mechanically damaged, and ranks lots by how fast they are degrading.

**Architecture:** One FastAPI process on the developer's Mac holds a single `Pipeline` instance and serialises requests behind a lock. The phone is a camera: it uploads a photograph over the LAN and renders the JSON that comes back. Session state is in memory. Nothing about the measurement changes — `Pipeline.read` and `rank_lots` are called as they already exist.

**Tech Stack:** Python 3.13 (`.venv` at the repo root), FastAPI, uvicorn, python-multipart, Pillow, numpy, PyTorch (already present). Client is plain HTML/CSS/JS with no framework and no CDN.

**Spec:** `docs/superpowers/specs/2026-08-18-capture-app-design.md`

## Global Constraints

- All commands run from `~/Documents/grain-capstone/stored-grain`. The interpreter is `.venv/bin/python`; there is no `python` on PATH.
- Tests run with `PYTHONPATH=$PWD .venv/bin/python -m pytest`. Imports are absolute from `src.` — never relative, never `sys.path` manipulation.
- TDD is not optional: write the failing test, run it, watch it fail for the right reason, then implement. A test that passes the first time you run it is a test that proves nothing.
- Comments explain **why**, never what. The existing modules are the style reference — read `src/pipeline.py` before writing any.
- Never add AI or Claude attribution to any file, commit message, or comment. This is absolute.
- `models/` and `data/` are gitignored. Never `git add -f` anything under them.
- Commit after every task with a conventional-commit subject (`feat:`, `fix:`, `test:`, `docs:`, `chore:`).
- The full suite must pass before every commit: `PYTHONPATH=$PWD .venv/bin/python -m pytest tests/ -q`. It stands at 139 passing.
- The app must never present a number as validated. The provenance banner, the ranking mode, ties, and the resolvable gap are requirements, not decoration.

---

### Task 1: Dependencies and the scale experiment

The experiment decides whether the app is usable at all, so it runs before any app code. The localiser learned kernels at one size in pixels; a phone photograph presents them at another. This measures how far that can drift before counting fails, and fixes the working size the server will resize every upload to.

**Files:**
- Create: `requirements.txt`
- Create: `src/app/__init__.py`
- Create: `experiments/scale_sensitivity.py`
- Create: `docs/superpowers/plans/scale-experiment-result.md`

**Interfaces:**
- Consumes: `src.pipeline.Pipeline`, an existing tray at `data/interim/validation/lot_10.png` (1391x1391, true damage 4.82%, 197 kernels placed).
- Produces: the constant `WORKING_LONG_EDGE_PX` (an integer) and the tolerated scale band, both recorded in the result document and consumed by Task 3.

- [ ] **Step 1: Install the dependencies**

```bash
.venv/bin/pip install fastapi uvicorn python-multipart httpx
```

`httpx` is required by FastAPI's `TestClient`. Then pin what is now installed:

```bash
.venv/bin/pip freeze | grep -iE "^(fastapi|uvicorn|python-multipart|httpx|torch|torchvision|numpy|pillow|pytest|scipy|scikit-image)==" | sort > requirements.txt
```

- [ ] **Step 2: Create the package marker**

```bash
mkdir -p src/app experiments
touch src/app/__init__.py
```

- [ ] **Step 3: Write the experiment**

Create `experiments/scale_sensitivity.py`:

```python
"""How far can a photograph's scale drift before the localiser stops counting?

The localiser learned kernels at one size in pixels. A phone photograph
presents them at another, and the network has no scale invariance to fall back
on -- it is a fully convolutional segmenter trained at one magnification. This
rescales a tray of known composition across a range of factors and measures
what counting does, which fixes both the size every upload is resized to and
how tightly the capture screen has to constrain framing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from src.pipeline import Pipeline

#: Factors applied to the native synthetic tray size. Spans a phone held too
#: close through one held too far.
FACTORS = (0.4, 0.5, 0.6, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 3.0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tray", default="data/interim/validation/lot_10.png")
    p.add_argument("--localiser", default="models/localiser.pt")
    p.add_argument("--classifier", default="models/classifier.session.pt")
    p.add_argument("--out", default="data/interim/scale/")
    p.add_argument("--results", default="models/scale-sensitivity.json")
    p.add_argument("--true-kernels", type=int, default=197)
    p.add_argument("--true-damage", type=float, default=4.82)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    original = Image.open(args.tray).convert("RGB")
    pipeline = Pipeline(args.localiser, args.classifier)

    rows = []
    print(f"{'factor':>7}{'long edge':>11}{'kernels':>9}{'count err%':>12}{'damage%':>9}")
    for factor in FACTORS:
        size = (round(original.width * factor), round(original.height * factor))
        scaled = original.resize(size, Image.LANCZOS)
        path = out / f"scale_{factor:.2f}.png"
        scaled.save(path)

        try:
            reading = pipeline.read(
                path, lot_id=path.stem, temperature_c=20.0, moisture_pct_wb=14.0
            )
            counted = reading.kernels_counted
            damage = reading.damage_mass_pct
            error = 100.0 * abs(counted - args.true_kernels) / args.true_kernels
        except ValueError as exc:
            counted, damage, error = 0, float("nan"), float("nan")
            print(f"{factor:>7.2f}{size[0]:>11}  failed: {exc}")
            rows.append(
                {"factor": factor, "long_edge_px": size[0], "kernels": 0,
                 "count_error_pct": None, "damage_mass_pct": None}
            )
            continue

        print(f"{factor:>7.2f}{size[0]:>11}{counted:>9}{error:>12.2f}{damage:>9.2f}")
        rows.append(
            {"factor": factor, "long_edge_px": size[0], "kernels": counted,
             "count_error_pct": error, "damage_mass_pct": damage}
        )

    usable = [r for r in rows if r["count_error_pct"] is not None
              and r["count_error_pct"] <= 5.0]
    summary = {
        "true_kernels": args.true_kernels,
        "true_damage_mass_pct": args.true_damage,
        "usable_factors": [r["factor"] for r in usable],
        "usable_long_edge_px": [r["long_edge_px"] for r in usable],
    }
    print("\n" + json.dumps(summary, indent=2))
    Path(args.results).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run it**

Run: `PYTHONPATH=$PWD .venv/bin/python -m experiments.scale_sensitivity`

Expected: a table across all eleven factors. Counting is expected to hold near 1.0 and to fail at the extremes. Note which factors keep count error under 5%.

- [ ] **Step 5: Record the result**

Write `docs/superpowers/plans/scale-experiment-result.md` containing: the table exactly as printed, the usable factor band, the chosen `WORKING_LONG_EDGE_PX` (the long edge at factor 1.0 if the band is centred there, otherwise the centre of the usable band), and one paragraph on what the band means for capture — whether a fixed resize suffices or the capture screen needs a framing guide.

If **no** factor other than 1.0 keeps count error under 5%, stop and report to the user before continuing. That result means a phone photograph must be framed to a tolerance the UI cannot enforce, and the plan needs revisiting rather than executing.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt src/app/__init__.py experiments/scale_sensitivity.py docs/superpowers/plans/scale-experiment-result.md
git commit -m "feat: measure how far photo scale can drift before counting fails"
```

---

### Task 2: The session store

Readings for one sitting, keyed by lot id, with ranking. Knows nothing about HTTP.

**Files:**
- Create: `src/app/store.py`
- Test: `tests/test_store.py`

**Interfaces:**
- Consumes: `src.pipeline.LotReading`, `src.pipeline.rank_lots`, `src.pipeline.RankedLot`.
- Produces:
  - `class SessionStore` with `add(reading: LotReading) -> None`, `get(lot_id: str) -> LotReading | None`, `remove(lot_id: str) -> bool`, `reset() -> None`, `readings() -> list[LotReading]`, `ranking() -> list[RankedLot]`, and `__contains__(lot_id: str) -> bool`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_store.py`:

```python
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
        for lot_id, damage in zip(order, [4.0, 12.0, 30.0]):
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=$PWD .venv/bin/python -m pytest tests/test_store.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'src.app.store'`.

- [ ] **Step 3: Write the implementation**

Create `src/app/store.py`:

```python
"""Readings for the current session.

A session is one sitting at one tray of lots. It lives in memory and dies with
the process, which is the honest lifetime: a stored reading outliving the
models that produced it would be a reading nobody could interpret later.
"""

from __future__ import annotations

from src.pipeline import LotReading, RankedLot, rank_lots


class SessionStore:
    def __init__(self) -> None:
        self._readings: dict[str, LotReading] = {}

    def add(self, reading: LotReading) -> None:
        """Store a reading, replacing any earlier one for the same lot.

        Replacement rather than accumulation because re-photographing is how a
        bad shot gets fixed, and two readings of one lot would both reach the
        ranking as if they were two lots.
        """
        self._readings[reading.lot_id] = reading

    def get(self, lot_id: str) -> LotReading | None:
        return self._readings.get(lot_id)

    def remove(self, lot_id: str) -> bool:
        """Drop a lot. Returns whether it was there, so a caller can 404."""
        return self._readings.pop(lot_id, None) is not None

    def reset(self) -> None:
        self._readings.clear()

    def readings(self) -> list[LotReading]:
        return list(self._readings.values())

    def ranking(self) -> list[RankedLot]:
        if not self._readings:
            return []
        return rank_lots(self.readings())

    def __contains__(self, lot_id: object) -> bool:
        return lot_id in self._readings
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=$PWD .venv/bin/python -m pytest tests/test_store.py -q`
Expected: 9 passed.

- [ ] **Step 5: Run the full suite**

Run: `PYTHONPATH=$PWD .venv/bin/python -m pytest tests/ -q`
Expected: 148 passed.

- [ ] **Step 6: Commit**

```bash
git add src/app/store.py tests/test_store.py
git commit -m "feat: session store for the capture app"
```

---

### Task 3: Upload handling and the working scale

Decode what the phone sent, reject what is not an image, and resize to the scale the localiser was trained at.

**Files:**
- Create: `src/app/images.py`
- Test: `tests/test_app_images.py`

**Interfaces:**
- Consumes: `WORKING_LONG_EDGE_PX` from Task 1's result document — substitute the recorded integer wherever the constant appears below.
- Produces:
  - `WORKING_LONG_EDGE_PX: int`
  - `class NotAnImage(Exception)`
  - `def decode(data: bytes) -> PIL.Image.Image` — raises `NotAnImage`
  - `def to_working_scale(image: Image.Image) -> Image.Image`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_app_images.py`:

```python
"""Tests for what arrives from a phone camera.

A phone photograph is nothing like the images the models were trained on: tens
of megapixels, arbitrary orientation, and occasionally not an image at all.
"""

import io

import pytest
from PIL import Image

from src.app.images import WORKING_LONG_EDGE_PX, NotAnImage, decode, to_working_scale


def _png_bytes(width, height, colour=(120, 90, 40)):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), colour).save(buffer, format="PNG")
    return buffer.getvalue()


def test_a_photograph_decodes():
    image = decode(_png_bytes(64, 48))
    assert image.size == (64, 48)


def test_something_that_is_not_an_image_is_refused():
    with pytest.raises(NotAnImage):
        decode(b"this is not a photograph")


def test_an_empty_upload_is_refused():
    with pytest.raises(NotAnImage):
        decode(b"")


def test_a_decoded_image_is_rgb_whatever_arrived():
    # Phones send JPEG, some tools send greyscale or RGBA PNGs. The pipeline
    # indexes three channels and would fail late and confusingly otherwise.
    buffer = io.BytesIO()
    Image.new("L", (32, 32), 128).save(buffer, format="PNG")
    assert decode(buffer.getvalue()).mode == "RGB"


def test_a_large_photograph_is_scaled_to_the_working_size():
    scaled = to_working_scale(Image.new("RGB", (8160, 6120)))
    assert max(scaled.size) == WORKING_LONG_EDGE_PX


def test_scaling_preserves_the_aspect_ratio():
    scaled = to_working_scale(Image.new("RGB", (4000, 3000)))
    assert scaled.width / scaled.height == pytest.approx(4000 / 3000, rel=1e-2)


def test_a_portrait_photograph_scales_on_its_long_edge():
    scaled = to_working_scale(Image.new("RGB", (3000, 4000)))
    assert scaled.height == WORKING_LONG_EDGE_PX


def test_a_small_image_is_left_alone():
    # Upscaling invents detail the localiser would read as kernel texture.
    small = Image.new("RGB", (200, 150))
    assert to_working_scale(small).size == (200, 150)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=$PWD .venv/bin/python -m pytest tests/test_app_images.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'src.app.images'`.

- [ ] **Step 3: Write the implementation**

Create `src/app/images.py`. Replace `1391` with the integer recorded in Task 1.

```python
"""Turning what a phone sent into what the models expect.

The localiser is fully convolutional and has no scale invariance: it learned
kernels at one size in pixels and a photograph presenting them at another is a
different problem. Every upload is therefore brought to one working scale
before it reaches the pipeline. The size, and the drift it tolerates, were
measured -- see docs/superpowers/plans/scale-experiment-result.md.
"""

from __future__ import annotations

import io

from PIL import Image, ImageOps, UnidentifiedImageError

#: Long edge every upload is resized to, matching the scale the localiser was
#: trained at. Measured, not guessed.
WORKING_LONG_EDGE_PX = 1391


class NotAnImage(Exception):
    """The upload could not be decoded as an image."""


def decode(data: bytes) -> Image.Image:
    """Decode an upload, honouring EXIF rotation and forcing RGB.

    Phones record orientation in EXIF rather than in the pixels, so a tray
    photographed in portrait arrives sideways unless it is transposed. The
    pipeline indexes three channels, so a greyscale or RGBA upload has to be
    converted here rather than failing deep inside inference.
    """
    if not data:
        raise NotAnImage("the upload was empty")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise NotAnImage(f"could not decode the upload as an image: {exc}") from exc
    return ImageOps.exif_transpose(image).convert("RGB")


def to_working_scale(image: Image.Image) -> Image.Image:
    """Resize so the long edge matches the trained scale, never upscaling.

    Enlarging a small photograph would invent detail that reads as kernel
    texture, so an image already below the working size is left as it is and
    allowed to fail honestly if its kernels are too small.
    """
    long_edge = max(image.size)
    if long_edge <= WORKING_LONG_EDGE_PX:
        return image
    factor = WORKING_LONG_EDGE_PX / long_edge
    size = (round(image.width * factor), round(image.height * factor))
    return image.resize(size, Image.LANCZOS)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=$PWD .venv/bin/python -m pytest tests/test_app_images.py -q`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/app/images.py tests/test_app_images.py
git commit -m "feat: bring phone uploads to the scale the localiser was trained at"
```

---

### Task 4: Expose per-instance detail from the pipeline

The overlay needs the instance array and the per-kernel predictions that `Pipeline.read` currently computes and discards. Split the internals; leave `read()`'s behaviour identical.

**Files:**
- Modify: `src/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class Detections` with `instances: np.ndarray` (int labels, 0 is background), `labels: list[int]` (instance labels kept after the debris filter, in prediction order), `predictions: list[str]` (project class name per kept instance, same order and length as `labels`).
  - `Pipeline.analyse(image_path, *, lot_id, temperature_c, moisture_pct_wb) -> tuple[LotReading, Detections]`
  - `Pipeline.read(...) -> LotReading` unchanged.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pipeline.py`:

```python
# --- Per-instance detail ---------------------------------------------------
# The overlay needs what the reading throws away. These pin the relationship
# between the two so the app can never draw a different measurement from the
# one the pipeline reported.

def test_detections_line_up_one_to_one_with_predictions():
    import numpy as np

    from src.pipeline import Detections

    detections = Detections(
        instances=np.array([[0, 1], [2, 2]]),
        labels=[1, 2],
        predictions=["sound", "fragment"],
    )
    assert len(detections.labels) == len(detections.predictions)


def test_detections_reject_a_mismatched_pairing():
    import numpy as np

    from src.pipeline import Detections

    with pytest.raises(ValueError, match="one prediction per instance"):
        Detections(
            instances=np.zeros((2, 2), dtype=int),
            labels=[1, 2],
            predictions=["sound"],
        )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=$PWD .venv/bin/python -m pytest tests/test_pipeline.py -q -k detections`
Expected: FAIL, `ImportError: cannot import name 'Detections' from 'src.pipeline'`.

- [ ] **Step 3: Add the dataclass**

In `src/pipeline.py`, after the `CorrectedDamage` dataclass, add:

```python
@dataclass(frozen=True)
class Detections:
    """Where the kernels were and what each one was called.

    The aggregate reading cannot be drawn on a photograph, and a second pass to
    recover the instances would be a second measurement that could disagree
    with the first. So the pipeline hands both out of one pass.
    """

    #: Instance labels per pixel; 0 is background.
    instances: np.ndarray

    #: Instance labels that survived the debris filter, in prediction order.
    labels: list[int]

    #: Project class name for each kept instance.
    predictions: list[str]

    def __post_init__(self) -> None:
        if len(self.labels) != len(self.predictions):
            raise ValueError(
                "one prediction per instance: "
                f"{len(self.labels)} labels, {len(self.predictions)} predictions"
            )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=$PWD .venv/bin/python -m pytest tests/test_pipeline.py -q -k detections`
Expected: 2 passed.

- [ ] **Step 5: Split `read` into `analyse` plus a wrapper**

In `src/pipeline.py`, rename the existing `read` method to `analyse`, change its return type to `tuple[LotReading, Detections]`, and return the detections alongside the reading. The body is unchanged except for the return. `kept` already holds the surviving instance labels and `predictions` already holds the per-crop class indices, so the detections are assembled from what is there:

```python
        names = [self.classes[index] for index in predictions]
        return reading, Detections(
            instances=instances, labels=kept, predictions=names
        )
```

Then add the wrapper immediately after it:

```python
    def read(
        self,
        image_path: str | Path,
        *,
        lot_id: str,
        temperature_c: float,
        moisture_pct_wb: float,
    ) -> LotReading:
        """The reading alone, for callers that do not draw the photograph."""
        reading, _ = self.analyse(
            image_path,
            lot_id=lot_id,
            temperature_c=temperature_c,
            moisture_pct_wb=moisture_pct_wb,
        )
        return reading
```

Keep the `@torch.no_grad()` decorator on `analyse`.

- [ ] **Step 6: Verify nothing that used `read` changed**

Run: `PYTHONPATH=$PWD .venv/bin/python -m pytest tests/ -q`
Expected: 150 passed.

Then confirm the real pipeline still produces the shipped result:

Run: `PYTHONPATH=$PWD .venv/bin/python -m src.validate_pipeline`
Expected: `mean_abs_error_pct_points` 1.108892385197364, `ranking_correct` true, `calibrated` false. Any other number means the split changed behaviour — stop and find out why before continuing.

- [ ] **Step 7: Commit**

```bash
git add src/pipeline.py tests/test_pipeline.py
git commit -m "feat: hand out per-instance detail alongside the reading"
```

---

### Task 5: The overlay

Draw kernel boundaries on the photograph, coloured by what each kernel was called.

**Files:**
- Create: `src/app/overlay.py`
- Test: `tests/test_overlay.py`

**Interfaces:**
- Consumes: `src.pipeline.Detections`, `src.mapping.classes.CLASSES` and `Admissibility`.
- Produces: `def draw(image: Image.Image, detections: Detections) -> Image.Image`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_overlay.py`:

```python
"""Tests for the annotated photograph.

The overlay is what a person actually looks at, so its job is to be faithful
to the measurement rather than attractive: a kernel the pipeline called
damaged must be marked as damaged, in the place the pipeline found it.
"""

import numpy as np
from PIL import Image

from src.app.overlay import DAMAGED_OUTLINE, SOUND_OUTLINE, draw
from src.pipeline import Detections


def _one_kernel_scene(class_name):
    # A 20x20 grey field with a 6x6 instance in the middle.
    instances = np.zeros((20, 20), dtype=int)
    instances[7:13, 7:13] = 1
    image = Image.new("RGB", (20, 20), (128, 128, 128))
    return image, Detections(instances=instances, labels=[1], predictions=[class_name])


def test_the_overlay_keeps_the_photograph_size():
    image, detections = _one_kernel_scene("sound")
    assert draw(image, detections).size == image.size


def test_the_original_photograph_is_not_modified():
    image, detections = _one_kernel_scene("sound")
    before = image.tobytes()
    draw(image, detections)
    assert image.tobytes() == before


def test_a_damaged_kernel_is_outlined_in_the_damage_colour():
    image, detections = _one_kernel_scene("fragment")
    pixels = np.array(draw(image, detections))
    assert (pixels == np.array(DAMAGED_OUTLINE)).all(axis=-1).any()


def test_a_sound_kernel_is_outlined_in_the_sound_colour():
    image, detections = _one_kernel_scene("sound")
    pixels = np.array(draw(image, detections))
    assert (pixels == np.array(SOUND_OUTLINE)).all(axis=-1).any()


def test_the_outline_lands_on_the_kernel_not_the_background():
    image, detections = _one_kernel_scene("fragment")
    pixels = np.array(draw(image, detections))
    marked = (pixels == np.array(DAMAGED_OUTLINE)).all(axis=-1)
    ys, xs = np.where(marked)
    # Every marked pixel sits within one pixel of the 7..12 instance box.
    assert ys.min() >= 6 and ys.max() <= 13
    assert xs.min() >= 6 and xs.max() <= 13


def test_an_empty_scene_returns_the_photograph_unchanged():
    image = Image.new("RGB", (10, 10), (200, 100, 50))
    detections = Detections(
        instances=np.zeros((10, 10), dtype=int), labels=[], predictions=[]
    )
    assert np.array_equal(np.array(draw(image, detections)), np.array(image))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=$PWD .venv/bin/python -m pytest tests/test_overlay.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'src.app.overlay'`.

- [ ] **Step 3: Write the implementation**

Create `src/app/overlay.py`:

```python
"""The photograph with the measurement drawn on it.

Two colours, not nine. The number that matters is the mechanical-damage
percentage, so the overlay answers the question that produces it -- which
kernels counted as damaged -- rather than displaying a class palette nobody
can hold in their head while looking at a tray.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from src.mapping.classes import CLASSES, Admissibility
from src.pipeline import Detections

#: Kernels that entered the damage term.
DAMAGED_OUTLINE = (220, 60, 40)

#: Kernels that did not. Biological deterioration is included here, because it
#: is measured and reported but never reaches the equation.
SOUND_OUTLINE = (60, 190, 120)


def _boundary(mask: np.ndarray) -> np.ndarray:
    """Pixels of the mask that touch something outside it.

    A one-pixel outline rather than a filled tint: the kernel itself has to
    stay visible, since the point of looking at the overlay is to disagree
    with it.
    """
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    interior = (
        padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
    )
    return mask & ~interior


def draw(image: Image.Image, detections: Detections) -> Image.Image:
    """Outline every detected kernel, coloured by whether it counted as damage."""
    pixels = np.array(image.convert("RGB"))

    for label, class_name in zip(detections.labels, detections.predictions):
        mask = detections.instances == label
        if not mask.any():
            continue
        damaged = CLASSES[class_name].admissibility is Admissibility.MECHANICAL
        pixels[_boundary(mask)] = DAMAGED_OUTLINE if damaged else SOUND_OUTLINE

    return Image.fromarray(pixels)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=$PWD .venv/bin/python -m pytest tests/test_overlay.py -q`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/app/overlay.py tests/test_overlay.py
git commit -m "feat: draw the measurement on the photograph"
```

---

### Task 6: The HTTP surface

FastAPI endpoints over the store, the pipeline, and the overlay.

**Files:**
- Create: `src/app/server.py`
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: everything from Tasks 2–5.
- Produces:
  - `def create_app(pipeline_factory=..., store=...) -> FastAPI` — the factory is called once on first use, so tests inject a fake and never load models.
  - `def serialise(reading: LotReading) -> dict`
  - Module-level `app = create_app()` for `uvicorn src.app.server:app`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_server.py`:

```python
"""Tests for the app's HTTP surface.

A fake pipeline stands in for the models: these tests are about what the
server does with a reading, not about inference, and loading 190 MB of weights
per test would make them useless to run.
"""

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.app.server import create_app
from src.physics.deterioration import assess
from src.pipeline import Detections, LotReading


class FakePipeline:
    """Returns a damage percentage taken from the lot id, so tests can order lots."""

    def __init__(self):
        self.calls = []

    def analyse(self, image_path, *, lot_id, temperature_c, moisture_pct_wb):
        self.calls.append(lot_id)
        damage = float(lot_id.split("_")[-1])
        reading = LotReading(
            lot_id=lot_id,
            kernels_counted=200,
            damage_mass_pct=damage,
            damage_count_pct=damage,
            biological_pct={"mould_suspect": 1.0},
            class_counts={"sound": 190, "fragment": 10},
            temperature_c=temperature_c,
            moisture_pct_wb=moisture_pct_wb,
            assessment=assess(damage, temperature_c, moisture_pct_wb),
        )
        instances = np.zeros((8, 8), dtype=int)
        instances[2:5, 2:5] = 1
        return reading, Detections(
            instances=instances, labels=[1], predictions=["fragment"]
        )


class FailingPipeline:
    def analyse(self, image_path, **kwargs):
        raise ValueError(f"no kernels found in {image_path}")


@pytest.fixture
def client():
    return TestClient(create_app(pipeline_factory=FakePipeline))


def _photo(width=40, height=30):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (120, 90, 40)).save(buffer, format="PNG")
    return buffer.getvalue()


def _upload(client, lot_id, temperature=20.0, moisture=14.0):
    return client.post(
        "/api/lots",
        files={"photo": (f"{lot_id}.png", _photo(), "image/png")},
        data={
            "lot_id": lot_id,
            "temperature_c": str(temperature),
            "moisture_pct_wb": str(moisture),
        },
    )


def test_a_photograph_produces_a_reading():
    client = TestClient(create_app(pipeline_factory=FakePipeline))
    response = _upload(client, "lot_5.0")
    assert response.status_code == 201
    body = response.json()
    assert body["lot_id"] == "lot_5.0"
    assert body["damage_mass_pct"] == pytest.approx(5.0)
    assert body["kernels_counted"] == 200


def test_a_reading_carries_its_resolution_limit(client):
    # Without this the interface can show an ordering the sample cannot support.
    body = _upload(client, "lot_5.0").json()
    assert body["resolvable_gap_pct"] > 0.0


def test_a_reading_says_absolute_days_are_gated(client):
    body = _upload(client, "lot_5.0").json()
    assert body["mode"] == "ranking"
    assert body["days_to_threshold"] is None
    assert body["suppression_reasons"]


def test_a_reading_says_whether_it_was_corrected(client):
    body = _upload(client, "lot_5.0").json()
    assert body["calibrated"] is False
    assert body["raw_damage_mass_pct"] == pytest.approx(body["damage_mass_pct"])


def test_lots_come_back_ranked_fastest_first(client):
    _upload(client, "lot_3.0")
    _upload(client, "lot_30.0")
    ranking = client.get("/api/lots").json()["ranking"]
    assert [entry["lot_id"] for entry in ranking] == ["lot_30.0", "lot_3.0"]
    assert ranking[0]["rank"] == 1


def test_lots_the_measurement_cannot_separate_are_reported_as_tied(client):
    # At 200 kernels a one-point gap at 5% damage is inside the sampling noise.
    _upload(client, "lot_5.0")
    _upload(client, "lot_6.0")
    ranking = client.get("/api/lots").json()["ranking"]
    assert ranking[0]["tied_with"] == ["lot_5.0"]


def test_rephotographing_a_lot_replaces_its_reading(client):
    _upload(client, "lot_5.0")
    _upload(client, "lot_5.0")
    assert len(client.get("/api/lots").json()["lots"]) == 1


def test_a_lot_can_be_deleted(client):
    _upload(client, "lot_5.0")
    assert client.delete("/api/lots/lot_5.0").status_code == 204
    assert client.get("/api/lots").json()["lots"] == []


def test_deleting_a_lot_that_was_never_there_is_a_404(client):
    assert client.delete("/api/lots/nobody").status_code == 404


def test_resetting_empties_the_session(client):
    _upload(client, "lot_5.0")
    assert client.post("/api/session/reset").status_code == 204
    assert client.get("/api/lots").json()["lots"] == []


def test_an_upload_that_is_not_an_image_is_refused(client):
    response = client.post(
        "/api/lots",
        files={"photo": ("notes.txt", b"not a photograph", "text/plain")},
        data={"lot_id": "lot_5.0", "temperature_c": "20", "moisture_pct_wb": "14"},
    )
    assert response.status_code == 415


def test_a_photograph_with_no_kernels_says_so_and_leaves_the_session_alone():
    client = TestClient(create_app(pipeline_factory=FailingPipeline))
    response = _upload(client, "lot_5.0")
    assert response.status_code == 422
    assert "no kernels" in response.json()["detail"]
    assert client.get("/api/lots").json()["lots"] == []


def test_the_overlay_is_served_for_a_known_lot(client):
    _upload(client, "lot_5.0")
    response = client.get("/api/lots/lot_5.0/overlay.png")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert Image.open(io.BytesIO(response.content)).size == (40, 30)


def test_the_overlay_of_an_unknown_lot_is_a_404(client):
    assert client.get("/api/lots/nobody/overlay.png").status_code == 404


def test_the_page_is_served_at_the_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_the_models_are_loaded_once_not_per_photograph():
    # Model load is 1.2 s. Per-request loading would make the app unusable and
    # would be invisible in a test that only checks the numbers.
    created = []

    def factory():
        created.append(1)
        return FakePipeline()

    client = TestClient(create_app(pipeline_factory=factory))
    _upload(client, "lot_5.0")
    _upload(client, "lot_9.0")
    assert len(created) == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=$PWD .venv/bin/python -m pytest tests/test_server.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'src.app.server'`.

- [ ] **Step 3: Write the implementation**

Create `src/app/server.py`:

```python
"""The HTTP surface: photograph in, reading out, lots ranked.

One process, one pipeline, one lock. There is a single set of weights and a
single MPS context, and a capstone demonstration has a single photographer, so
requests are serialised rather than made concurrent -- concurrency here would
add failure modes and buy nothing.

The pipeline is built lazily and once. Loading weights takes 1.2 seconds, which
is fine at startup and unusable per request, and deferring it means the tests
can inject a fake and never touch the models at all.
"""

from __future__ import annotations

import io
import threading
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from src.app.images import NotAnImage, decode, to_working_scale
from src.app.overlay import draw
from src.app.store import SessionStore
from src.pipeline import LotReading

STATIC = Path(__file__).parent / "static"


def serialise(reading: LotReading) -> dict:
    """One reading as JSON, carrying its own limits.

    The limits travel with the number deliberately. A damage percentage without
    its resolvable gap, its mode, and whether it was corrected is a figure
    someone will quote out of context -- most likely in a review.
    """
    assessment = reading.assessment
    return {
        "lot_id": reading.lot_id,
        "kernels_counted": reading.kernels_counted,
        "damage_mass_pct": reading.damage_mass_pct,
        "damage_count_pct": reading.damage_count_pct,
        "raw_damage_mass_pct": reading.raw_damage_mass_pct,
        "calibrated": reading.calibration is not None,
        "biological_pct": reading.biological_pct,
        "class_counts": reading.class_counts,
        "temperature_c": reading.temperature_c,
        "moisture_pct_wb": reading.moisture_pct_wb,
        "resolvable_gap_pct": reading.resolvable_gap_pct,
        "degradation_rate": assessment.degradation_rate,
        "mode": assessment.mode,
        "days_to_threshold": assessment.days_to_threshold,
        "suppression_reasons": list(assessment.suppression_reasons),
        "ranges_ok": assessment.ranges.all_ok,
        "range_notes": list(assessment.ranges.notes),
    }


def _default_pipeline():
    from src.pipeline import Pipeline

    return Pipeline()


def create_app(pipeline_factory=_default_pipeline, store: SessionStore | None = None):
    app = FastAPI(title="Maize damage capture")
    store = store if store is not None else SessionStore()

    state = {"pipeline": None}
    lock = threading.Lock()
    photographs: dict[str, bytes] = {}
    overlays: dict[str, bytes] = {}

    def pipeline():
        if state["pipeline"] is None:
            state["pipeline"] = pipeline_factory()
        return state["pipeline"]

    @app.post("/api/lots", status_code=201)
    async def add_lot(
        photo: UploadFile = File(...),
        lot_id: str = Form(...),
        temperature_c: float = Form(...),
        moisture_pct_wb: float = Form(...),
    ):
        raw = await photo.read()
        try:
            image = to_working_scale(decode(raw))
        except NotAnImage as exc:
            raise HTTPException(status_code=415, detail=str(exc)) from exc

        with lock, TemporaryDirectory() as tmp:
            path = Path(tmp) / "upload.png"
            image.save(path)
            try:
                reading, detections = pipeline().analyse(
                    path,
                    lot_id=lot_id,
                    temperature_c=temperature_c,
                    moisture_pct_wb=moisture_pct_wb,
                )
            except ValueError as exc:
                # A failed photograph never mutates the session.
                raise HTTPException(status_code=422, detail=str(exc)) from exc

            buffer = io.BytesIO()
            draw(image, detections).save(buffer, format="PNG")
            overlays[lot_id] = buffer.getvalue()
            photographs[lot_id] = raw
            store.add(reading)

        return serialise(reading)

    @app.get("/api/lots")
    def list_lots():
        return {
            "lots": [serialise(r) for r in store.readings()],
            "ranking": [
                {
                    "lot_id": entry.reading.lot_id,
                    "rank": entry.rank,
                    "tied_with": list(entry.tied_with),
                    "degradation_rate": entry.reading.assessment.degradation_rate,
                    "damage_mass_pct": entry.reading.damage_mass_pct,
                    "resolvable_gap_pct": entry.reading.resolvable_gap_pct,
                }
                for entry in store.ranking()
            ],
        }

    @app.get("/api/lots/{lot_id}/overlay.png")
    def overlay(lot_id: str):
        if lot_id not in overlays:
            raise HTTPException(status_code=404, detail=f"no lot {lot_id!r}")
        return Response(content=overlays[lot_id], media_type="image/png")

    @app.delete("/api/lots/{lot_id}", status_code=204)
    def delete_lot(lot_id: str):
        if not store.remove(lot_id):
            raise HTTPException(status_code=404, detail=f"no lot {lot_id!r}")
        overlays.pop(lot_id, None)
        photographs.pop(lot_id, None)
        return Response(status_code=204)

    @app.post("/api/session/reset", status_code=204)
    def reset():
        store.reset()
        overlays.clear()
        photographs.clear()
        return Response(status_code=204)

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


app = create_app()
```

- [ ] **Step 4: Create a placeholder page so the root route resolves**

The page itself is Task 7; the route needs a file to serve now.

```bash
mkdir -p src/app/static
printf '<!doctype html>\n<title>Maize damage capture</title>\n<p>Capture screen goes here.</p>\n' > src/app/static/index.html
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `PYTHONPATH=$PWD .venv/bin/python -m pytest tests/test_server.py -q`
Expected: 16 passed.

- [ ] **Step 6: Run the full suite**

Run: `PYTHONPATH=$PWD .venv/bin/python -m pytest tests/ -q`
Expected: 174 passed.

- [ ] **Step 7: Commit**

```bash
git add src/app/server.py src/app/static/index.html tests/test_server.py
git commit -m "feat: HTTP surface for the capture app"
```

---

### Task 7: The page

Two screens in one file: capture, and the ranking across lots.

**Files:**
- Create: `src/app/static/index.html`
- Create: `src/app/static/app.css`
- Create: `src/app/static/app.js`

**Interfaces:**
- Consumes: the endpoints from Task 6, exactly as serialised there.

- [ ] **Step 1: Write the page**

Replace `src/app/static/index.html`:

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Maize damage capture</title>
<link rel="stylesheet" href="/static/app.css">
</head>
<body>

<p class="provenance">
  These models were trained on synthetic trays composed from one laboratory
  imaging rig. <strong>No phone photograph has been validated against ground
  truth.</strong> Every number below is provisional.
</p>

<nav>
  <button id="tab-capture" class="active">Capture</button>
  <button id="tab-ranking">Ranking</button>
</nav>

<section id="capture">
  <form id="capture-form">
    <label>Lot name
      <input name="lot_id" required placeholder="brand A, bag 2">
    </label>
    <label>Temperature, &deg;C
      <input name="temperature_c" type="number" step="0.1" value="25" required>
    </label>
    <label>Moisture, % wet basis
      <input name="moisture_pct_wb" type="number" step="0.1" value="14" required>
    </label>
    <label class="camera">Photograph the tray
      <input name="photo" type="file" accept="image/*" capture="environment" required>
    </label>
    <button type="submit">Measure</button>
  </form>
  <p id="status" role="status"></p>
  <article id="result" hidden></article>
</section>

<section id="ranking" hidden>
  <p class="mode">Lots are ordered by degradation rate. Absolute days to
  threshold are withheld: Thompson&rsquo;s dry-matter-loss coefficients are
  unverified, so this system ranks and does not predict.</p>
  <div id="ranking-body"></div>
  <button id="reset">Clear session</button>
</section>

<script src="/static/app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Write the stylesheet**

Create `src/app/static/app.css`:

```css
:root {
  --ink: #1c1a17;
  --paper: #faf8f5;
  --line: #d9d2c8;
  --damage: #dc3c28;
  --sound: #3cbe78;
  --warn: #8a6d1f;
}

* { box-sizing: border-box; }

body {
  margin: 0;
  padding: 0 1rem 3rem;
  font: 16px/1.5 system-ui, sans-serif;
  color: var(--ink);
  background: var(--paper);
  max-width: 40rem;
  margin-inline: auto;
}

.provenance {
  margin: 0 -1rem 1rem;
  padding: 0.75rem 1rem;
  background: #2b2722;
  color: #f4efe7;
  font-size: 0.85rem;
}

nav { display: flex; gap: 0.5rem; margin-bottom: 1rem; }

nav button {
  flex: 1;
  padding: 0.6rem;
  border: 1px solid var(--line);
  background: transparent;
  font: inherit;
  border-radius: 0.3rem;
}

nav button.active { background: var(--ink); color: var(--paper); }

label { display: block; margin-bottom: 0.75rem; }

input {
  display: block;
  width: 100%;
  padding: 0.6rem;
  margin-top: 0.25rem;
  font: inherit;
  border: 1px solid var(--line);
  border-radius: 0.3rem;
  background: #fff;
}

.camera input { padding: 1.5rem 0.6rem; }

button[type="submit"], #reset {
  width: 100%;
  padding: 0.9rem;
  font: inherit;
  border: 0;
  border-radius: 0.3rem;
  background: var(--ink);
  color: var(--paper);
}

#status { min-height: 1.5rem; color: var(--warn); }

.headline { font-size: 2.5rem; margin: 0.5rem 0 0; }
.headline small { display: block; font-size: 0.9rem; color: #6b635a; }

table { width: 100%; border-collapse: collapse; margin-bottom: 1rem; }
th, td { text-align: left; padding: 0.4rem 0.3rem; border-bottom: 1px solid var(--line); }
th { font-weight: 600; font-size: 0.85rem; }

.tie { color: var(--warn); font-size: 0.85rem; }
.caveat { font-size: 0.85rem; color: var(--warn); }
img.overlay { width: 100%; border-radius: 0.3rem; }
.legend span::before {
  content: "\25CF";
  margin-right: 0.3rem;
}
.legend .damaged::before { color: var(--damage); }
.legend .sound::before { color: var(--sound); }
```

- [ ] **Step 3: Write the client script**

Create `src/app/static/app.js`:

```javascript
"use strict";

const status = document.getElementById("status");
const result = document.getElementById("result");

function show(name) {
  const capturing = name === "capture";
  document.getElementById("capture").hidden = !capturing;
  document.getElementById("ranking").hidden = capturing;
  document.getElementById("tab-capture").classList.toggle("active", capturing);
  document.getElementById("tab-ranking").classList.toggle("active", !capturing);
  if (!capturing) refreshRanking();
}

document.getElementById("tab-capture").onclick = () => show("capture");
document.getElementById("tab-ranking").onclick = () => show("ranking");

document.getElementById("capture-form").onsubmit = async (event) => {
  event.preventDefault();
  const form = event.target;
  status.textContent = "Measuring…";
  result.hidden = true;

  const response = await fetch("/api/lots", {
    method: "POST",
    body: new FormData(form),
  });

  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }));
    status.textContent = body.detail;
    return;
  }

  status.textContent = "";
  renderReading(await response.json());
  form.querySelector('input[name="photo"]').value = "";
};

function renderReading(lot) {
  // The gap travels with the number: a damage figure the sample cannot
  // support is a figure that should not be compared with another one.
  const bio = Object.entries(lot.biological_pct)
    .map(([name, pct]) => `${name.replace(/_/g, " ")} ${pct.toFixed(1)}%`)
    .join(", ") || "none detected";

  result.innerHTML = `
    <p class="headline">${lot.damage_mass_pct.toFixed(2)}%
      <small>mechanical damage by weight, from ${lot.kernels_counted} kernels</small>
    </p>
    <img class="overlay" alt="detected kernels"
         src="/api/lots/${encodeURIComponent(lot.lot_id)}/overlay.png?t=${Date.now()}">
    <p class="legend">
      <span class="damaged">counted as damage</span> &nbsp;
      <span class="sound">not counted</span>
    </p>
    <table>
      <tr><th>By count</th><td>${lot.damage_count_pct.toFixed(2)}%</td></tr>
      <tr><th>Smallest gap this sample can resolve</th>
          <td>${lot.resolvable_gap_pct.toFixed(2)} points</td></tr>
      <tr><th>Biological deterioration, withheld from the model</th><td>${bio}</td></tr>
      <tr><th>Mode</th><td>${lot.mode}</td></tr>
    </table>
    <p class="caveat">Uncorrected reading. The classifier errs in both
      directions, which overstates clean grain and understates heavily damaged
      grain; correction is off because it does not transfer between imaging
      sessions.</p>
    ${lot.ranges_ok ? "" : `<p class="caveat">Outside the published validity
      range: ${lot.range_notes.join("; ")}</p>`}
  `;
  result.hidden = false;
}

async function refreshRanking() {
  const body = document.getElementById("ranking-body");
  const data = await (await fetch("/api/lots")).json();

  if (!data.ranking.length) {
    body.innerHTML = "<p>No lots photographed yet.</p>";
    return;
  }

  const rows = data.ranking.map((entry) => `
    <tr>
      <td>${entry.rank}</td>
      <td>${entry.lot_id}
        ${entry.tied_with.length
          ? `<div class="tie">cannot be separated from
             ${entry.tied_with.join(", ")} at this sample size</div>`
          : ""}
      </td>
      <td>${entry.damage_mass_pct.toFixed(2)}%</td>
      <td>${entry.degradation_rate.toFixed(4)}</td>
    </tr>
  `).join("");

  body.innerHTML = `
    <table>
      <tr><th>Rank</th><th>Lot</th><th>Damage</th><th>Rate</th></tr>
      ${rows}
    </table>
  `;
}

document.getElementById("reset").onclick = async () => {
  await fetch("/api/session/reset", { method: "POST" });
  refreshRanking();
};
```

- [ ] **Step 4: Verify the server still serves the page**

Run: `PYTHONPATH=$PWD .venv/bin/python -m pytest tests/test_server.py -q`
Expected: 16 passed.

- [ ] **Step 5: Commit**

```bash
git add src/app/static/
git commit -m "feat: capture and ranking screens"
```

---

### Task 8: End-to-end against the real models

Two slow tests proving the app reports the pipeline's own measurement rather than a second implementation of it.

**Files:**
- Create: `tests/test_app_end_to_end.py`
- Modify: `pytest.ini` (create if absent)

**Interfaces:**
- Consumes: `create_app`, the real `Pipeline`, and the tray at `data/interim/validation/lot_10.png`.

- [ ] **Step 1: Register the marker**

Create `pytest.ini` if it does not exist, otherwise add the `markers` line:

```ini
[pytest]
markers =
    slow: loads real model weights and runs inference
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_app_end_to_end.py`:

```python
"""The app against the real models.

Slow and worth it: everything else in the app's tests uses a fake pipeline, so
without these the app could quietly become a second implementation of the
measurement that disagrees with the one that was validated.

Run with: PYTHONPATH=$PWD .venv/bin/python -m pytest tests/test_app_end_to_end.py -q
Skipped automatically when the models or the tray are not on disk.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.app.server import create_app

TRAY = Path("data/interim/validation/lot_10.png")
MODELS = (Path("models/localiser.pt"), Path("models/classifier.session.pt"))

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not TRAY.exists() or not all(m.exists() for m in MODELS),
        reason="needs the trained models and a synthetic tray on disk",
    ),
]


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


def test_the_app_reports_what_the_pipeline_reports(client):
    from src.pipeline import Pipeline

    direct = Pipeline().read(
        TRAY, lot_id="lot_10", temperature_c=25.0, moisture_pct_wb=14.0
    )

    response = client.post(
        "/api/lots",
        files={"photo": ("lot_10.png", TRAY.read_bytes(), "image/png")},
        data={"lot_id": "lot_10", "temperature_c": "25", "moisture_pct_wb": "14"},
    )
    assert response.status_code == 201
    body = response.json()

    assert body["kernels_counted"] == direct.kernels_counted
    assert body["damage_mass_pct"] == pytest.approx(
        direct.damage_mass_pct, abs=1e-6
    )


def test_the_measurement_lands_near_the_tray_s_known_damage(client):
    # lot_10 is 4.82% damaged by weight by construction.
    body = client.get("/api/lots").json()["lots"][0]
    assert body["damage_mass_pct"] == pytest.approx(4.82, abs=2.5)


def test_the_overlay_matches_the_photograph_s_dimensions(client):
    import io

    from PIL import Image

    response = client.get("/api/lots/lot_10/overlay.png")
    assert response.status_code == 200
    assert Image.open(io.BytesIO(response.content)).size == Image.open(TRAY).size
```

- [ ] **Step 3: Run them**

Run: `PYTHONPATH=$PWD .venv/bin/python -m pytest tests/test_app_end_to_end.py -q`
Expected: 3 passed, in roughly 10 seconds.

If `test_the_app_reports_what_the_pipeline_reports` fails, the app is not measuring what the pipeline measures — most likely the working-scale resize changed the image. That is a real defect, not a tolerance to widen. Report it rather than adjusting the assertion.

- [ ] **Step 4: Run the full suite excluding the slow tests**

Run: `PYTHONPATH=$PWD .venv/bin/python -m pytest tests/ -q -m "not slow"`
Expected: 174 passed, 3 deselected.

- [ ] **Step 5: Commit**

```bash
git add tests/test_app_end_to_end.py pytest.ini
git commit -m "test: the app reports the pipeline's own measurement"
```

---

### Task 9: Running it, and the README

How to start the server and reach it from the phone.

**Files:**
- Create: `src/app/README.md`
- Modify: `README.md`

- [ ] **Step 1: Write the app's README**

Create `src/app/README.md`:

```markdown
# Capture app

Photograph trays of maize from a phone, measure mechanical damage, rank lots.

## Running it

    PYTHONPATH=$PWD .venv/bin/python -m uvicorn src.app.server:app --host 0.0.0.0 --port 8000

`--host 0.0.0.0` is what makes the phone able to reach it. Find the Mac's
address on the network with `ipconfig getifaddr en0`, then open
`http://<that address>:8000` in Chrome on the phone. A personal hotspot from
the phone works and needs no wifi.

Models load on the first photograph, not at startup, so the first measurement
takes about three seconds and the rest about two.

## What it does not do

No persistence: a session lives in the process. No per-session calibration
mode; that becomes useful only once weighed mixtures exist to photograph.

## The numbers it shows

Every reading is uncorrected and comes from models trained on synthetic trays
from one laboratory rig. The interface says so, permanently, because the app
will be pointed at maize that the models have never seen.
```

- [ ] **Step 2: Add a pointer in the repository README**

In `README.md`, immediately before the `## Running the tests` heading, insert:

```markdown
## The capture app

`src/app/` serves a phone-facing page that photographs a tray and ranks lots.
See `src/app/README.md` for how to run it and reach it from a phone.

```

- [ ] **Step 3: Start the server and check it by hand**

Run: `PYTHONPATH=$PWD .venv/bin/python -m uvicorn src.app.server:app --host 0.0.0.0 --port 8000`

In a browser, open `http://localhost:8000`, upload `data/interim/validation/lot_10.png` through the form with any lot name, and confirm: a damage figure near 4.8% appears, the overlay renders with outlines, the provenance banner is visible, and the ranking tab lists the lot. Then stop the server.

- [ ] **Step 4: Commit**

```bash
git add src/app/README.md README.md
git commit -m "docs: how to run the capture app and reach it from a phone"
```

---

## Self-Review

**Spec coverage.** Server, store, overlay, images, static client: Tasks 2, 3, 5, 6, 7. The `Pipeline` split for per-instance detail: Task 4. All six endpoints: Task 6. The four things the interface must not hide — provenance banner, ranking mode with days gated, ties as ties, resolvable gap per lot — appear in the serialiser (Task 6), the page (Task 7), and are asserted in `test_a_reading_says_absolute_days_are_gated`, `test_lots_the_measurement_cannot_separate_are_reported_as_tied`, and `test_a_reading_carries_its_resolution_limit`. The uncorrected marker is `test_a_reading_says_whether_it_was_corrected`. The scale risk and its experiment: Task 1, which gates the rest. Error handling table: 415 and 422 are tested in Task 6; duplicate lot ids are tested as replacement rather than 409, which the spec allowed as "offering replace" — the store replaces and the test pins it. Out-of-range temperature and moisture are surfaced through `ranges_ok` and `range_notes` rather than rejected, as specified.

**One deliberate deviation from the spec.** The spec's error table lists a 422 for a median kernel size outside the trained band. That check is not implemented, because Task 1 measures the band and the sensible rejection threshold is not known until it has run. If the experiment shows a narrow band, add the check as a follow-up task against the measured numbers rather than a guessed one.

**Placeholders.** None. Every code step carries the code.

**Type consistency.** `Detections(instances, labels, predictions)` is defined in Task 4 and consumed with the same field names in Tasks 5 and 6. `SessionStore` methods used by `server.py` — `add`, `get`, `remove`, `reset`, `readings`, `ranking` — all exist in Task 2. `decode` and `to_working_scale` are used in Task 6 exactly as defined in Task 3. `serialise` key names match every key the client reads in Task 7: `damage_mass_pct`, `damage_count_pct`, `kernels_counted`, `resolvable_gap_pct`, `biological_pct`, `mode`, `ranges_ok`, `range_notes`, and the ranking's `lot_id`, `rank`, `tied_with`, `damage_mass_pct`, `degradation_rate`.

**Assumption to verify during Task 6.** `Assessment.mode` is read by `serialise`. It is used in `src/validate_pipeline.py` as `ranked[0].reading.assessment.mode`, so it exists; if it is a property rather than a field, the serialiser is unaffected.
