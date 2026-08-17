"""Score the localiser on whole trays rather than crops.

Training evaluates on 384 px crops because that is what it trains on, but a
crop is a harder and unrepresentative problem: kernels sliced by the crop
boundary are partial objects that no real photograph contains. The number that
matters is the one measured on a whole scene, because a whole scene is what the
capture protocol produces.

Counting error is reported as the headline rather than mask quality. The damage
percentage is damaged kernels over total kernels, so a miscount moves the
number handed to the deterioration model. A tenth of a percent of boundary IoU
does not.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.vision.dataset import IMAGENET_MEAN, IMAGENET_STD
from src.vision.localise import UNet, count_error, instance_f1, instances_from_logits
from src.vision.train_localiser import device

#: Four pooling stages, so both sides must be divisible by 16.
STRIDE = 16


@torch.no_grad()
def predict_scene(model, image: np.ndarray, dev) -> torch.Tensor:
    """Run the whole image through the network in one pass.

    The network is fully convolutional, so it accepts any size that the
    pooling stages divide evenly. Padding by reflection rather than zeros
    avoids inventing a hard black edge that reads as a kernel boundary.
    """
    x = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    x = (x - torch.tensor(IMAGENET_MEAN)[:, None, None]) / torch.tensor(IMAGENET_STD)[
        :, None, None
    ]
    x = x.unsqueeze(0)

    _, _, h, w = x.shape
    pad_h = (STRIDE - h % STRIDE) % STRIDE
    pad_w = (STRIDE - w % STRIDE) % STRIDE
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

    logits = model(x.to(dev))[0]
    return logits[:, :h, :w]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenes", default="data/interim/scenes/val")
    p.add_argument("--checkpoint", default="models/localiser.pt")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--results", default="models/localiser-fullscene.json")
    args = p.parse_args()

    dev = device()
    model = UNet().to(dev)
    state = torch.load(args.checkpoint, map_location=dev)
    model.load_state_dict(state["model"])
    model.eval()

    from PIL import Image

    paths = sorted(Path(args.scenes).glob("scene_*.png"))[: args.limit]
    rows = []
    print(f"{'scene':>6}{'true':>7}{'pred':>7}{'diff':>7}{'err%':>8}{'F1':>8}", flush=True)
    for path in paths:
        image = np.array(Image.open(path).convert("RGB"))
        truth = np.load(path.with_name(path.stem + "_inst.npy"))

        predicted = instances_from_logits(predict_scene(model, image, dev))
        ce = count_error(predicted, truth)
        f1 = instance_f1(predicted, truth)["f1"]
        rows.append({**ce, "f1": f1})
        print(
            f"{path.stem[-4:]:>6}{ce['true']:>7}{ce['predicted']:>7}"
            f"{ce['error']:>+7}{ce['relative_error'] * 100:>7.1f}%{f1:>8.3f}",
            flush=True,
        )

    errors = np.array([r["relative_error"] for r in rows])
    summary = {
        "scenes": len(rows),
        "mean_abs_relative_count_error_pct": float(np.mean(np.abs(errors)) * 100),
        "median_abs_relative_count_error_pct": float(np.median(np.abs(errors)) * 100),
        # Signed, because a consistent bias can be corrected and random error
        # cannot. If the localiser always undercounts by 3%, the damage ratio
        # is barely affected -- both numerator and denominator shift together.
        "mean_signed_relative_count_error_pct": float(np.mean(errors) * 100),
        "mean_instance_f1": float(np.mean([r["f1"] for r in rows])),
        "mean_abs_count_error_kernels": float(
            np.mean([abs(r["error"]) for r in rows])
        ),
    }
    print("\n" + json.dumps(summary, indent=2), flush=True)
    Path(args.results).write_text(json.dumps({"summary": summary, "scenes": rows}, indent=2))


if __name__ == "__main__":
    main()
