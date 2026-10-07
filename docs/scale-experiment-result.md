# Scale sensitivity — result

**Run:** 18 August 2026 · `experiments/scale_sensitivity.py`
**Tray:** `data/interim/validation/lot_10.png`, 1391x1391, true damage 4.82% by mass
**Models:** `models/localiser.pt`, `models/classifier.session.pt`
**Raw results:** `models/scale-sensitivity.json`

## A correction to the ground truth

An earlier setting used `--true-kernels 197`. That is the number the synthesiser
*placed*, not the number a segmenter can *see*: overlap and edge clipping mean
32 of those 197 are not separable kernels in the rendered tray.
`models/pipeline-validation.json` records the countable truth for this tray as
**165**, against which the pipeline's own validated run counted 164.

Scored against 197, every factor including the native one fails — the native
scale appears to have 16.75% count error when it actually has 0.61%. The
default was corrected to 165 before the run below. Anyone re-running this must
use the countable count, not the placed count.

## The table

| factor | long edge px | kernels | count err % | damage % |
|---|---|---|---|---|
| 0.40 | 556 | 143 | 13.33 | 5.46 |
| 0.50 | 696 | 154 | 6.67 | 4.25 |
| 0.60 | 835 | 157 | 4.85 | 4.55 |
| 0.75 | 1043 | 158 | 4.24 | 4.79 |
| 0.90 | 1252 | 160 | 3.03 | 4.45 |
| **1.00** | **1391** | **164** | **0.61** | **4.85** |
| 1.10 | 1530 | 163 | 1.21 | 5.42 |
| 1.25 | 1739 | 161 | 2.42 | 5.76 |
| 1.50 | 2086 | 159 | 3.64 | 5.84 |
| 2.00 | 2782 | 132 | 20.00 | 4.49 |
| 3.00 | 4173 | 79 | 52.12 | 5.05 |

## The usable band

**Factors 0.60 to 1.50 hold count error under 5%** — long edges of 835 px to
2086 px, a 2.5x range. Outside it the localiser does not degrade gracefully:
counting falls off a cliff at 2.0 (20% error) and collapses at 3.0 (52%), and
at 0.40 kernels have shrunk below what the network resolves.

Damage percentage is the more forgiving quantity. Across the whole usable band
it stays within roughly one point of the 4.82% truth (4.45% to 5.84%), and the
error is one-directional at the upper end — upscaling past 1.25 splits kernels
and inflates the fragment count, which reads as damage. Counting fails loudly;
damage drifts quietly, so the count-error criterion is the right gate.

## `WORKING_LONG_EDGE_PX = 1391`

The geometric centre of the usable band is factor 0.95, which is within noise
of the native 1.0. The native long edge is therefore both the centre of the
tolerated range and the scale with the lowest observed error, and every upload
is resized so its long edge is **1391 px**.

## What this means for capture

**A fixed resize suffices. No framing guide is required.**

A 2.5x tolerance in apparent kernel size is wide enough that a person holding a
phone over a tray cannot easily leave it. To fall below 0.6 the tray would have
to occupy under a quarter of the frame; to exceed 1.5 the phone would have to
be close enough that the tray no longer fits. Both are visible mistakes that a
photographer corrects without instruction.

Two consequences for the capture app:

1. The server resizes on the **long edge**, not on kernel size, because the
   band is wide enough that a proxy is unnecessary.
2. The median-kernel-size rejection in the design remains worth keeping, but as
   a guard against the genuinely wrong photograph — a close-up of three
   kernels, a tray at arm's length across a room — rather than as routine
   framing enforcement. It should be set from the band measured here, not
   tighter.
