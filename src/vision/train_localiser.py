"""Train the kernel localiser on synthetic bulk scenes.

Scenes are pre-generated rather than composed on the fly: composition costs
about a third of a second, which would starve the GPU, and a fixed scene set
makes runs comparable to each other.

Split discipline carries over from the classifier. A scene is built only from
kernels belonging to one split, so a kernel whose image trained the localiser
never reappears inside a validation scene.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.data.manifest import build, load_grainset
from src.vision.dataset import IMAGENET_MEAN, IMAGENET_STD
from src.vision.localise import (
    UNet,
    count_error,
    instance_f1,
    instances_from_logits,
    instance_targets,
)
from src.vision.synth import build_cutout_pool, compose, pool_for_damage


def generate(pool, n_scenes: int, n_kernels: int, seed: int, out: Path) -> list[Path]:
    """Write scenes and their instance maps to disk, skipping any already there."""
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    rng = random.Random(seed)
    for i in range(n_scenes):
        image_path = out / f"scene_{i:04d}.png"
        inst_path = out / f"scene_{i:04d}_inst.npy"
        if not (image_path.exists() and inst_path.exists()):
            damage = rng.choice([0, 2, 5, 10, 20, 40])
            selection = pool_for_damage(pool, damage, n_kernels, seed=seed * 1000 + i)
            scene = compose(selection, seed=seed * 1000 + i)
            scene.save(image_path)
            np.save(inst_path, scene.instances)
        paths.append(image_path)
    return paths


class SceneDataset(Dataset):
    """Random crops from pre-generated scenes."""

    def __init__(self, paths: list[Path], crop: int = 384, augment: bool = True):
        self.paths = paths
        self.crop = crop
        self.augment = augment

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        from PIL import Image

        image = np.array(Image.open(self.paths[i]).convert("RGB"))
        instances = np.load(self.paths[i].with_name(self.paths[i].stem + "_inst.npy"))

        h, w = instances.shape
        c = min(self.crop, h, w)
        top = random.randint(0, h - c)
        left = random.randint(0, w - c)
        image = image[top : top + c, left : left + c]
        instances = instances[top : top + c, left : left + c]

        if self.augment:
            k = random.randint(0, 3)
            image = np.rot90(image, k, axes=(0, 1)).copy()
            instances = np.rot90(instances, k).copy()
            if random.random() < 0.5:
                image = image[:, ::-1].copy()
                instances = instances[:, ::-1].copy()

        target = instance_targets(instances)

        x = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        x = (x - torch.tensor(IMAGENET_MEAN)[:, None, None]) / torch.tensor(
            IMAGENET_STD
        )[:, None, None]
        return x, torch.from_numpy(target), torch.from_numpy(instances.astype(np.int32))


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, dataset, dev, limit: int = 24) -> dict:
    model.eval()
    counts, f1s, relative = [], [], []
    for i in range(min(limit, len(dataset))):
        x, _, instances = dataset[i]
        logits = model(x.unsqueeze(0).to(dev))[0]
        predicted = instances_from_logits(logits)
        truth = instances.numpy()
        ce = count_error(predicted, truth)
        counts.append(abs(ce["error"]))
        relative.append(abs(ce["relative_error"]))
        f1s.append(instance_f1(predicted, truth)["f1"])
    return {
        "mean_abs_count_error": float(np.mean(counts)),
        "mean_abs_relative_count_error": float(np.mean(relative)),
        "instance_f1": float(np.mean(f1s)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grainset", default="data/raw/grainset/maize")
    p.add_argument("--scenes", default="data/interim/scenes")
    p.add_argument("--train-scenes", type=int, default=300)
    p.add_argument("--val-scenes", type=int, default=40)
    p.add_argument("--kernels", type=int, default=200)
    p.add_argument("--pool", type=int, default=600)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--crop", type=int, default=384)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--checkpoint", default="models/localiser.pt")
    p.add_argument("--results", default="models/localiser-results.json")
    args = p.parse_args()

    manifest = build(load_grainset(Path(args.grainset)))
    root = Path(args.scenes)

    print("building cutout pools", flush=True)
    train_pool = build_cutout_pool(manifest.split("train"), limit=args.pool, seed=1)
    val_pool = build_cutout_pool(manifest.split("val"), limit=args.pool // 3, seed=2)
    print(f"pools: train {len(train_pool)} | val {len(val_pool)}", flush=True)

    started = time.time()
    train_paths = generate(train_pool, args.train_scenes, args.kernels, 1, root / "train")
    val_paths = generate(val_pool, args.val_scenes, args.kernels, 2, root / "val")
    print(
        f"scenes: train {len(train_paths)} | val {len(val_paths)} "
        f"({time.time() - started:.0f}s)",
        flush=True,
    )

    train_set = SceneDataset(train_paths, args.crop, augment=True)
    val_set = SceneDataset(val_paths, args.crop, augment=False)
    loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )

    dev = device()
    model = UNet().to(dev)
    # The border class is a thin rim and therefore rare in pixel terms, but it
    # is the entire mechanism for separating touching kernels. Left unweighted,
    # the network learns to skip it and merges neighbours.
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0, 3.0]).to(dev))
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)

    best = {"instance_f1": -1.0}
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running, seen = time.time(), 0.0, 0
        for x, target, _ in loader:
            x, target = x.to(dev), target.to(dev)
            optimiser.zero_grad()
            loss = criterion(model(x), target)
            loss.backward()
            optimiser.step()
            running += loss.item() * x.size(0)
            seen += x.size(0)
        schedule.step()

        metrics = evaluate(model, val_set, dev)
        print(
            f"epoch {epoch}/{args.epochs} loss {running / seen:.4f} "
            f"instance-F1 {metrics['instance_f1']:.4f} "
            f"count error {metrics['mean_abs_count_error']:.1f} kernels "
            f"({metrics['mean_abs_relative_count_error'] * 100:.1f}%) "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )
        if metrics["instance_f1"] > best["instance_f1"]:
            best = {**metrics, "epoch": epoch}
            Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), **best}, args.checkpoint)

    Path(args.results).write_text(json.dumps(best, indent=2))
    print(f"best: {best}", flush=True)


if __name__ == "__main__":
    main()
