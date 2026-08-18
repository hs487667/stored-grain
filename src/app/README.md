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

## Reaching it from outside the network

A tunnel publishes the local server on an HTTPS URL without deploying
anything. The models, the weights and the inference stay on this machine, which
is the point: a real deploy would have to ship 120 MB of checkpoints and a
536 MB PyTorch install, and would then run on a CPU with no Metal backend.

    brew install cloudflared
    cloudflared tunnel --url http://localhost:8000

It prints a `https://<name>.trycloudflare.com` address that works from any
network. Free, and no account for a quick tunnel.

**Do not use a tunnel for the November review.** A lecture room demonstration
should not depend on Cloudflare and campus internet both working at the same
minute. Use the LAN address, or a personal hotspot with the Mac joined to it,
and test that path in the room beforehand.

Two things must be fixed before the tunnel is pointed at anyone else:
the session is process-wide, so every visitor shares one set of lots (see
`store.py`), and uploads are neither size-capped nor rate-limited, so a public
URL invites arbitrary images through a neural network on this machine.

## What it does not do

No persistence: a session lives in the process. No per-session calibration
mode; that becomes useful only once weighed mixtures exist to photograph.

## The numbers it shows

Every reading is uncorrected and comes from models trained on synthetic trays
from one laboratory rig. The interface says so, permanently, because the app
will be pointed at maize that the models have never seen.
