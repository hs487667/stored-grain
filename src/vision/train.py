"""Train the per-kernel classifier, and report the number honestly.

Two rules this module exists to enforce.

**Macro-F1, never plain accuracy.** ``sound`` is about half the corpus. A model
that predicts ``sound`` for everything scores near 50% accuracy and is worth
nothing, because the quantity the project needs is the damaged fraction.

**Report the session split and the random split side by side.** The random
number is the one most of the literature quotes. It is also the optimistic one,
and the gap between the two is a result worth publishing rather than an
embarrassment worth hiding.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from src.data.manifest import (
    build,
    by_day,
    by_session,
    load_cornseeds,
    load_grainset,
    random_split_baseline,
)
from src.vision.dataset import (
    CLASS_NAMES,
    KernelDataset,
    class_weights,
    eval_transform,
    train_transform,
)


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(n_classes: int = len(CLASS_NAMES)) -> nn.Module:
    """ResNet50, ImageNet-initialised.

    Chosen because Kumari et al. (2026) found ResNet50V2 best of the three
    architectures they compared on the closest published task, which makes it
    the defensible baseline rather than the interesting one.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model


def macro_f1(confusion: torch.Tensor) -> tuple[float, dict[str, float]]:
    """Macro-F1 and the per-class F1 behind it."""
    per_class: dict[str, float] = {}
    scores = []
    for i, name in enumerate(CLASS_NAMES):
        tp = confusion[i, i].item()
        fp = confusion[:, i].sum().item() - tp
        fn = confusion[i, :].sum().item() - tp
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        per_class[name] = f1
        # A class absent from this split cannot be scored; averaging a zero in
        # would understate the model rather than measure it.
        if confusion[i, :].sum().item() > 0:
            scores.append(f1)
    return (sum(scores) / len(scores) if scores else 0.0), per_class


@torch.no_grad()
def evaluate(model, loader, dev) -> dict:
    model.eval()
    n = len(CLASS_NAMES)
    confusion = torch.zeros(n, n, dtype=torch.long)
    per_dataset = {0: [0, 0], 1: [0, 0]}   # dataset -> [correct, total]

    for images, labels, source in loader:
        preds = model(images.to(dev)).argmax(1).cpu()
        for t, p in zip(labels, preds):
            confusion[t, p] += 1
        for s, t, p in zip(source, labels, preds):
            per_dataset[int(s)][1] += 1
            per_dataset[int(s)][0] += int(t == p)

    f1, per_class = macro_f1(confusion)
    total = confusion.sum().item()
    return {
        "macro_f1": f1,
        "accuracy": confusion.diag().sum().item() / total if total else 0.0,
        "per_class_f1": per_class,
        "grainset_accuracy": (
            per_dataset[0][0] / per_dataset[0][1] if per_dataset[0][1] else None
        ),
        "cornseeds_accuracy": (
            per_dataset[1][0] / per_dataset[1][1] if per_dataset[1][1] else None
        ),
        "n": total,
    }


def run(strategy: str, args) -> dict:
    samples = load_grainset(Path(args.grainset)) + load_cornseeds(Path(args.cornseeds))
    if strategy == "session":
        manifest = build(samples, key=by_session)
    elif strategy == "day":
        manifest = build(samples, key=by_day)
    elif strategy == "random":
        manifest = random_split_baseline(samples)
    else:
        raise ValueError(f"unknown split strategy {strategy!r}")

    train_samples = manifest.split("train")
    val_samples = manifest.split("val")
    print(
        f"[{strategy}] train {len(train_samples)} | val {len(val_samples)} | "
        f"leakage {manifest.leakage_pct():.1f}%",
        flush=True,
    )

    train_loader = DataLoader(
        KernelDataset(train_samples, train_transform(args.size)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        KernelDataset(val_samples, eval_transform(args.size)),
        batch_size=args.batch_size,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )

    dev = device()
    model = build_model().to(dev)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_samples).to(dev))
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)

    best = {"macro_f1": -1.0}
    for epoch in range(1, args.epochs + 1):
        model.train()
        started, running = time.time(), 0.0
        for images, labels, _ in train_loader:
            images, labels = images.to(dev), labels.to(dev)
            optimiser.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimiser.step()
            running += loss.item() * images.size(0)
        schedule.step()

        metrics = evaluate(model, val_loader, dev)
        print(
            f"[{strategy}] epoch {epoch}/{args.epochs} "
            f"loss {running / len(train_samples):.4f} "
            f"macro-F1 {metrics['macro_f1']:.4f} acc {metrics['accuracy']:.4f} "
            f"({time.time() - started:.0f}s)",
            flush=True,
        )
        if metrics["macro_f1"] > best["macro_f1"]:
            best = {**metrics, "epoch": epoch}
            if args.checkpoint:
                Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {"model": model.state_dict(), "classes": CLASS_NAMES, **best},
                    f"{args.checkpoint}.{strategy}.pt",
                )

    return best


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grainset", default="data/raw/grainset/maize")
    p.add_argument("--cornseeds", default="data/raw/kaggle/cornseeds/corn")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--checkpoint", default="models/classifier")
    p.add_argument(
        "--strategies",
        default="session,random",
        help="Which splits to train under. Both, so the inflation is visible.",
    )
    p.add_argument("--results", default="models/baseline-results.json")
    args = p.parse_args()

    results = {s: run(s, args) for s in args.strategies.split(",")}

    baseline = results.get("random")
    if baseline:
        for strategy in ("session", "day"):
            if strategy in results:
                gap = baseline["macro_f1"] - results[strategy]["macro_f1"]
                results[f"inflation_vs_{strategy}"] = gap
                print(
                    f"\nRandom split scores {gap:+.4f} macro-F1 against the "
                    f"{strategy} split. Positive means the random figure is "
                    f"the optimistic one.",
                    flush=True,
                )

    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results).write_text(json.dumps(results, indent=2))
    print(f"wrote {args.results}", flush=True)


if __name__ == "__main__":
    main()
