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
    # 165 is the count the localiser can see, recorded in
    # models/pipeline-validation.json. The synthesiser placed 197, but overlap
    # and edge clipping mean 32 of those are not separable kernels in the
    # rendered tray. Scoring against the placed count would condemn framing
    # that is in fact correct.
    p.add_argument("--true-kernels", type=int, default=165)
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
