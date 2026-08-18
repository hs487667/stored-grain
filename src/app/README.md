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

After at least one lot is measured, the ranking panel offers PDF and CSV
downloads. Both reports are scoped to the current browser session and include
the rank, entered storage conditions, measured damage, kernel count,
measurement resolution, degradation result, and days-to-threshold when
available. The CSV retains the model-warning fields for analysis; the
presentation-focused PDF omits notes. Empty sessions return no report.

## Interface

The browser UI uses a field-instrument visual system shared with the PDF:
chlorophyll green, maize yellow, condensed technical headings, and a visible
capture-to-ranking process rail. Desktop keeps the capture station beside the
reading and ranking workspace; phone layouts collapse into the same sequence.
Motion is CSS-only and disabled through `prefers-reduced-motion`. No fonts,
images, or scripts are fetched from third parties.

## Reaching it from outside the network

A tunnel publishes the local server on an HTTPS URL without deploying
anything. The models, the weights and the inference stay on this machine, which
is the point: a real deploy would have to ship 120 MB of checkpoints and a
536 MB PyTorch install, and would then run on a CPU with no Metal backend.

    brew install cloudflared
    cloudflared tunnel --url http://localhost:8000

It prints a `https://<name>.trycloudflare.com` address that works from any
network. Free, and no account for a quick tunnel.

**Do not depend on a tunnel for the November review.** A lecture room demonstration
should not depend on Cloudflare and campus internet both working at the same
minute. Use the LAN address, or a personal hotspot with the Mac joined to it,
and test that path in the room beforehand.

Visitor data is isolated by an HTTP-only session cookie. Upload byte and decoded
pixel limits, rate limiting, and session expiry are enforced by the server.
This still is a local capstone demonstration, not a production deployment.

## What it does not do

No durable persistence: sessions live in the process and expire. No
per-session calibration mode; that becomes useful only once weighed mixtures
exist to photograph.

## The numbers it shows

Every reading is uncorrected and comes from models trained on synthetic trays
from one laboratory rig. The interface says so, permanently, because the app
will be pointed at maize that the models have never seen.
