"""Put the labelled real kernels onto other surfaces and other light.

``localiser-real.pt`` was fine-tuned only on trays shot on a matte black card
under even light. On anything else it fails in a specific way: a lamp's bright
patch on a grey mousepad reads as kernel, so the app counts the mat between
kernels and loses the kernels inside the patch, and the total still looks
plausible.

The auto-labelled trays already hold every kernel's exact pixels, and on black
card the kernel/background split is clean. So the kernels are cut out and
pasted onto photographs of real surfaces (CC0 textures from Poly Haven), then
the scene is relit: an uneven illumination field over everything, a lamp
hotspot added to the surface, and a soft shadow under each kernel. The
instance map is carried over unchanged, so every composite is labelled.

Half the scenes keep the tray's own layout, touching kernels included; half
scatter the kernels apart, as a spread on a mat looks. Textures and trays are
split before composing, so validation composites use surfaces and kernels the
training composites never saw.
"""

from __future__ import annotations

import argparse
import io
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance
from scipy import ndimage

from src.vision.train_localiser import load_instances, real_scene_paths

#: Scenes per folder. ``real_scene_paths`` treats each folder as one group.
GROUP_SIZE = 25


def background(texture: Path, size: tuple[int, int], rng: random.Random) -> np.ndarray:
    """A random crop of a texture, recoloured, at ``size`` (width, height)."""
    image = Image.open(texture).convert("RGB")
    if rng.random() < 0.5:
        image = image.transpose(Image.Transpose.ROTATE_90)
    side = int(min(image.size) * rng.uniform(0.25, 1.0))
    aspect = size[0] / size[1]
    w, h = (side, int(side / aspect)) if aspect >= 1 else (int(side * aspect), side)
    left = rng.randint(0, image.width - w)
    top = rng.randint(0, image.height - h)
    image = image.crop((left, top, left + w, top + h)).resize(size, Image.BICUBIC)
    image = ImageEnhance.Color(image).enhance(rng.uniform(0.2, 1.3))
    # Skewed dark: mats, cloths and sacks are the realistic case, and a pale
    # surface is the hardest for a bright-kernel prior.
    image = ImageEnhance.Brightness(image).enhance(rng.choice([rng.uniform(0.15, 0.6), rng.uniform(0.6, 1.4)]))
    return np.asarray(image, dtype=np.float32)


def scatter(image: np.ndarray, instances: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    """The same kernels, moved apart to random non-overlapping places."""
    h, w = instances.shape
    out_img = np.zeros_like(image)
    out_inst = np.zeros_like(instances)
    taken = np.zeros((h, w), bool)
    labels = [l for l in np.unique(instances) if l]
    rng.shuffle(labels)
    keep = int(len(labels) * rng.uniform(0.5, 1.0))
    slices = ndimage.find_objects(instances)
    placed = 0
    for label in labels[:keep]:
        sl = slices[label - 1]
        mask = instances[sl] == label
        patch = image[sl]
        k = rng.randint(0, 3)
        mask, patch = np.rot90(mask, k), np.rot90(patch, k)
        mh, mw = mask.shape
        grown = ndimage.binary_dilation(mask, iterations=3)
        for _ in range(20):
            y, x = rng.randint(0, h - mh), rng.randint(0, w - mw)
            if not (taken[y : y + mh, x : x + mw] & grown).any():
                taken[y : y + mh, x : x + mw] |= grown
                placed += 1
                out_inst[y : y + mh, x : x + mw][mask] = placed
                out_img[y : y + mh, x : x + mw][mask] = patch[mask]
                break
    return out_img, out_inst


def relight(scene: np.ndarray, bg: np.ndarray, mask: np.ndarray, rng: random.Random) -> np.ndarray:
    """Shadows, uneven light and a lamp hotspot over a pasted scene."""
    h, w = mask.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    # Soft shadow under the kernels, falling away from the light.
    dy, dx = rng.randint(-6, 6), rng.randint(-6, 6)
    shadow = ndimage.gaussian_filter(np.roll(mask, (dy, dx), (0, 1)).astype(np.float32), rng.uniform(2, 5))
    bg = bg * (1 - rng.uniform(0.2, 0.55) * shadow)[..., None]

    # Edges feathered by under a pixel: the cut-outs carry a rim of black card.
    alpha = ndimage.gaussian_filter(mask.astype(np.float32), 0.6)[..., None]
    alpha = np.where(mask[..., None], np.maximum(alpha, 0.85), alpha * 0.5)
    out = alpha * scene + (1 - alpha) * bg

    # Illumination falling off across the frame, applied to everything.
    gx, gy = rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5)
    field = 1 + gx * (xx / w - 0.5) + gy * (yy / h - 0.5)
    out = out * (field * rng.uniform(0.75, 1.2))[..., None]

    # The lamp's hotspot: mostly on the surface, which reflects it, a little on
    # the kernels, and close to white rather than the surface's colour.
    if rng.random() < 0.7:
        cy, cx = rng.uniform(0.1, 0.9) * h, rng.uniform(0.1, 0.9) * w
        sigma = rng.uniform(0.12, 0.45) * max(h, w)
        spot = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2))
        strength = rng.uniform(40, 150)
        weight = np.where(mask, 0.25, 1.0)
        out = out + (strength * spot * weight)[..., None] * np.array([1.0, 0.98, 0.95])

    out = np.clip(out, 0, 255)
    if rng.random() < 0.4:
        out = ndimage.gaussian_filter(out, (rng.uniform(0.3, 1.0),) * 2 + (0,))
    out = out + np.random.default_rng(rng.randint(0, 2**31)).normal(0, rng.uniform(1, 5), out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def rescale(image: np.ndarray, instances: np.ndarray, factor: float,
            rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    """Resize the kernels by ``factor`` inside a frame of the same size.

    Larger kernels are cropped to a random window; smaller ones are placed at a
    random offset with empty surface around them.
    """
    h, w = instances.shape
    size = (max(1, round(w * factor)), max(1, round(h * factor)))
    big = np.asarray(Image.fromarray(image.astype(np.uint8)).resize(size, Image.LANCZOS), dtype=np.float32)
    big_inst = np.asarray(Image.fromarray(instances.astype(np.int32)).resize(size, Image.NEAREST))
    out = np.zeros_like(image)
    out_inst = np.zeros_like(instances)
    sh, sw = big_inst.shape
    if factor >= 1:
        y, x = rng.randint(0, sh - h), rng.randint(0, sw - w)
        out, out_inst = big[y : y + h, x : x + w], big_inst[y : y + h, x : x + w]
    else:
        y, x = rng.randint(0, h - sh), rng.randint(0, w - sw)
        out[y : y + sh, x : x + sw] = big
        out_inst[y : y + sh, x : x + sw] = big_inst
    return out, out_inst


def compose_one(source: Path, texture: Path, rng: random.Random,
                scale: tuple[float, float] = (1.0, 1.0)) -> tuple[np.ndarray, np.ndarray]:
    image = np.asarray(Image.open(source).convert("RGB"), dtype=np.float32)
    instances = load_instances(source)
    factor = rng.uniform(*scale)
    if factor != 1.0:
        image, instances = rescale(image, instances, factor, rng)
    if rng.random() < 0.5:
        image, instances = scatter(image, instances, rng)
    bg = background(texture, (instances.shape[1], instances.shape[0]), rng)
    scene = relight(image, bg, instances > 0, rng)
    # JPEG round trip: phones never hand over clean pixels.
    buffer = io.BytesIO()
    Image.fromarray(scene).save(buffer, "JPEG", quality=rng.randint(70, 95))
    return np.asarray(Image.open(buffer).convert("RGB")), instances


def write_split(sources: list[Path], textures: list[Path], n: int, out: Path, prefix: str, seed: int,
                scale: tuple[float, float] = (1.0, 1.0)) -> None:
    rng = random.Random(seed)
    for i in range(n):
        image, instances = compose_one(rng.choice(sources), rng.choice(textures), rng, scale)
        folder = out / f"{prefix}{i // GROUP_SIZE:03d}"
        folder.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(folder / f"{i:05d}.png")
        Image.fromarray(instances.astype(np.uint16)).save(folder / f"{i:05d}_inst.png")
        if (i + 1) % 50 == 0:
            print(f"{prefix}: {i + 1}/{n}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--real", help="auto-labelled trays, one folder per tray")
    p.add_argument("--synthetic", help="GrainSet scenes with train/ and val/ folders, used instead of --real; "
                   "they carry the damaged and discoloured kernels the trays lack")
    p.add_argument("--scale", default="1,1", help="range of kernel resizing, e.g. 0.35,1.1")
    p.add_argument("--prefix", default="bg")
    p.add_argument("--textures", default="data/raw/backgrounds/polyhaven")
    p.add_argument("--out", default="data/interim/backgrounds")
    p.add_argument("--val-groups", default="")
    p.add_argument("--test-groups", default="")
    p.add_argument("--train", type=int, default=1200)
    p.add_argument("--val", type=int, default=100)
    p.add_argument("--val-texture-fraction", type=float, default=0.15)
    args = p.parse_args()

    val_groups = {g for g in args.val_groups.split(",") if g}
    test_groups = {g for g in args.test_groups.split(",") if g}
    if args.synthetic:
        scenes = lambda d: sorted(p for p in (Path(args.synthetic) / d).glob("*.png") if not p.stem.endswith("_inst"))
        train_sources, val_sources = scenes("train"), scenes("val")
    else:
        real = Path(args.real)
        train_sources = real_scene_paths(real, exclude=val_groups | test_groups)
        val_sources = real_scene_paths(real, groups=val_groups)
    scale = tuple(float(v) for v in args.scale.split(","))

    textures = sorted(Path(args.textures).glob("*.jpg"))
    random.Random(0).shuffle(textures)
    n_val = max(1, int(len(textures) * args.val_texture_fraction))
    val_textures, train_textures = textures[:n_val], textures[n_val:]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.prefix}-split.json").write_text(json.dumps({
        "train_trays": sorted({s.parent.name for s in train_sources}),
        "val_trays": sorted(val_groups), "test_trays": sorted(test_groups),
        "train_textures": [t.stem for t in train_textures],
        "val_textures": [t.stem for t in val_textures],
    }, indent=1))
    print(f"sources: train {len(train_sources)} | val {len(val_sources)}; "
          f"textures: train {len(train_textures)} | val {len(val_textures)}", flush=True)

    write_split(val_sources, val_textures, args.val, out, f"{args.prefix}val", seed=2, scale=scale)
    write_split(train_sources, train_textures, args.train, out, f"{args.prefix}train", seed=1, scale=scale)


if __name__ == "__main__":
    main()
