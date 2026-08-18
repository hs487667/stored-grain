# Capture App — Design

**Date:** 18 August 2026
**Status:** approved for planning

A phone-facing web app that photographs trays of shelled maize, measures the
percentage of kernels that are mechanically damaged, and ranks lots by how fast
they are degrading. It is the demonstrable surface of a pipeline that already
works: `src/pipeline.py` reads a photograph and returns a `LotReading`, and
`rank_lots` orders those readings. Nothing about the measurement changes here.

## Why this shape

Three decisions were taken before design, and they constrain everything below.

**Inference runs on the Mac; the phone is a camera.** The models stay in
PyTorch and the app calls the same `Pipeline` that produced the validated
1.11-point result. Exporting to ONNX or TFLite would buy offline operation and
cost days of quantisation work, and every numerical drift it introduced would
become something to explain in a review. Model load is 1.2 s and inference is
1.6 s for a 200-kernel tray, so a server on a laptop is fast enough to demo
live.

**The client is a mobile web page, not an Android app.** No install, no Play
Store, no Gradle loop; it runs in Chrome on the S24 Ultra and on an examiner's
phone. The native app is a better artefact and a worse use of the eleven weeks
before the November review.

**The app is multi-lot.** A single damage percentage demonstrates the
measurement; the ranking is the project's actual claim, and it only appears
when several lots exist at once.

## Architecture

    phone camera ──HTTP──> FastAPI (localhost / LAN)
                              │
                              ├── Pipeline.read()  ← existing, unchanged behaviour
                              ├── SessionStore     ← in-memory readings
                              └── overlay renderer

One process, one `Pipeline` instance built at startup and reused. Requests are
serialised behind a lock: there is one MPS context and one set of weights, and
a capstone demo has one photographer. Concurrency is not a requirement and
pretending otherwise would add failure modes for nothing.

State lives in memory for the life of the process. A session is one sitting at
one tray, and persistence would raise questions — which readings are still
valid, what happens when the models change — that the project does not need
answered yet. `POST /api/session/reset` and restarting the server are the same
operation.

### Components

| Unit | Responsibility | Depends on |
|---|---|---|
| `src/app/server.py` | HTTP surface, request validation, lock | FastAPI, store, pipeline |
| `src/app/store.py` | Readings for the current session, keyed by lot id | `src/pipeline.py` |
| `src/app/overlay.py` | Draw instance boundaries on the photo, coloured by class | Pillow, numpy |
| `src/app/images.py` | Decode upload, reject or resize to the working scale | Pillow |
| `src/app/static/` | The page: capture screen, ranking screen | nothing |

Each is independently testable. The store holds `LotReading` objects and knows
nothing about HTTP; the overlay takes an image plus an instance array and
returns an image; `images.py` is pure image arithmetic. Only `server.py` knows
about FastAPI.

### One change to existing code

`Pipeline.read` computes an instance array and a per-kernel prediction list and
discards both, returning only the aggregate `LotReading`. The overlay needs
them. The internals split into an `_analyse()` returning instances, kept
labels, and predictions, with `read()` wrapping it and returning exactly what it
returns today. Existing callers and the 139 passing tests are untouched.

## API

| Method | Path | Request | Response |
|---|---|---|---|
| `POST` | `/api/lots` | multipart: `photo`, `lot_id`, `temperature_c`, `moisture_pct_wb` | reading JSON, 201 |
| `GET` | `/api/lots` | — | readings + ranking with ties |
| `GET` | `/api/lots/{id}/overlay.png` | — | annotated PNG |
| `DELETE` | `/api/lots/{id}` | — | 204 |
| `POST` | `/api/session/reset` | — | 204 |
| `GET` | `/` | — | the page |

A reading serialises as: `lot_id`, `kernels_counted`, `damage_mass_pct`,
`damage_count_pct`, `raw_damage_mass_pct`, `biological_pct`, `class_counts`,
`temperature_c`, `moisture_pct_wb`, `resolvable_gap_pct`, `mode`,
`degradation_rate`, `days_to_threshold` (null only when an input is out of
range -- see the 18 August note below), and `calibrated`.

The ranking response carries, per lot, its rank, the lots it is tied with, and
the degradation rate it was ordered on.

## What the interface must not hide

The app will be pointed at real maize that the models have never seen, by
people who will read a number off a phone screen. Four things are therefore
part of the design rather than presentation polish:

1. **A permanent provenance banner.** Models are trained on synthetic trays
   composed from one laboratory imaging rig. No phone photograph has ever been
   validated against ground truth. Every number the app shows is provisional
   and the banner says so without being dismissible.
2. **Ranking is the output; days are secondary.** `mode` is displayed. Days to
   threshold shows with Steele's standard error inline, and standing caveats
   about the model render as a source line rather than as a warning. Where an
   out-of-range input withholds the figure, the app says which input and why.

   *Superseded 18 August:* this originally read "where absolute days would go,
   the app shows that `constants_verified` is false", because days were gated
   on Thompson (1972). They no longer are — Steele's own reference times reach
   the same answer from a source already on disk. See handoff §10d.
3. **Ties are shown as ties.** `rank_lots` already refuses to order lots closer
   than the sampling noise. The UI renders that refusal rather than presenting
   an arbitrary order as a result.
4. **Every reading carries its resolvable gap**, and a lot whose gap exceeds
   its distance from its neighbour is marked as unresolvable at this sample
   size, with the kernel count needed to separate it.

A reading is also marked **uncorrected**, because bias correction is off by
default and the known residual — clean grain overstated, heavy damage
understated — belongs next to the number rather than in a commit message.

## The scale risk

The localiser learned kernels at one size in pixels. Synthetic trays run about
1000 px across with roughly 200 kernels in frame. An S24 Ultra photograph is
12 MP by default and up to 200 MP, so a raw upload presents kernels several
times larger than anything the network has seen, and segmentation is expected
to fail — not degrade.

Mitigation, in order:

1. `images.py` resizes every upload so the long edge is a fixed working size,
   chosen to put kernel diameters in the trained range.
2. Before the app is trusted, an experiment: take an existing synthetic tray,
   rescale it across a range of factors, and measure count error at each. That
   curve says how tight the framing has to be and whether a single fixed resize
   is enough.
3. If the curve is narrow, the capture screen gains a framing guide — a fixed
   phone height and an on-screen box the tray must fill — and the reading is
   rejected when the median kernel size falls outside the trained band.

Step 2 gates the app's usefulness and is cheap. It runs before the capture
screen is built.

## Error handling

| Case | Behaviour |
|---|---|
| No kernels found | 422, "no kernels found — check framing and lighting" |
| Median kernel size outside the trained band | 422, naming the direction and the fix |
| Upload is not an image, or is corrupt | 415 |
| Duplicate `lot_id` | 409, offering replace |
| Temperature or moisture outside the model's validity range | Accepted; the physics layer already flags this and the flag is surfaced |
| Inference raises | 500, message logged, session left intact |

A failed photograph never mutates the session.

## Testing

Test-driven throughout, following the repository's existing style: each test
pins a specific way the thing can be silently wrong.

- **Endpoint tests** inject a fake pipeline through FastAPI's dependency
  override, so they run in milliseconds and cover: upload happy path, ranking
  with ties, deletion, reset, duplicate lot ids, non-image uploads, and the
  no-kernels path.
- **Store tests** cover replacement, ordering independence, and reset.
- **Overlay tests** assert the rendered image has the right dimensions and
  that instance boundaries land on the instances, not that it looks nice.
- **Image tests** cover the resize arithmetic and the size-band rejection.
- **Two slow tests** run the real models over an existing synthetic tray and
  assert the response matches the pipeline's own reading, which is what
  guarantees the app is not quietly a second implementation of the
  measurement.

## Out of scope

Persistence, user accounts, multiple concurrent sessions, on-device inference,
cloud deployment, and the per-session calibration mode. Calibration is
deliberately deferred: it becomes useful only when weighed mixtures exist to
photograph, and building it before then would be building against an imagined
workflow.
