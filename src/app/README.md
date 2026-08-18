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
