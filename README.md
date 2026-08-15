# Hybrid AI-Based Maize Quality Assessment and Deterioration Risk Ranking

Measures mechanical damage in shelled maize from a photograph and uses it as the
input to a published deterioration model, ranking storage lots by how fast they
are degrading.

**Hybrid** because only the vision half learns anything. Temperature, moisture
and storage duration never touch a model — they go into Thompson's published
equation. That split is deliberate and is what lets the results be defended
without a storage trial.

The primary output is a **ranking**. Absolute days-to-threshold is secondary and
gated on `constants_verified` in `src/physics/constants.py`, which stays false
until Thompson (1972) is checked against the original.

## Layout

- `src/physics/` — deterioration model. Pure arithmetic over published
  constants, no learned parameters, no dependency on the vision pipeline.
- `tests/` — each test pins a specific way the model can be silently wrong.
- `data/raw/grainset/` — GrainSet maize subset and its data card (gitignored).

## Running the tests

    python3 -m venv .venv
    .venv/bin/pip install pytest
    .venv/bin/python -m pytest tests/ -q

Planning documents live one level up in `../`.
