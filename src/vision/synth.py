"""Build bulk-maize scenes out of single-kernel cutouts.

Every public dataset in this field photographs one kernel at a time on a clean
background -- GrainSet, GrainSpace and Corn Seeds alike, and GrainSpace states
outright that extra matter is removed by hand. The photograph this project
actually takes is two hundred kernels touching each other on a black card. So
there is no public training data for the one thing the localiser has to do:
decide where one kernel ends and the next begins.

GrainSet ships a binary mask for all 19,000 maize images, and the capture
protocol puts real kernels on a matte black card. That combination makes
compositing unusually honest here: the cutouts are exact, and the synthetic
background is the same colour as the real one.

What a scene gives that a real photograph does not:

* **Exact instance masks**, so the localiser has supervision for touching
  kernels that no dataset provides.
* **An exactly known damage percentage**, mass basis included, because the
  scene is assembled from kernels of known class. That lets the whole pipeline
  -- localise, classify, map, rank -- be tested end to end before any maize has
  been bought.

Two things a synthetic scene cannot teach, stated here so they are not
forgotten when the results look good: kernels composited onto a background cast
no shadows on each other and sit in a single plane, and every cutout was lit by
the same laboratory rig. Real kernels shadow their neighbours and a phone under
a desk lamp does not look like a P600. Synthetic scenes are for teaching
separation, not for claiming performance.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

from src.data.manifest import Sample
from src.mapping.classes import CLASSES, Admissibility, measure

#: The capture protocol's matte black card, as an RGB value. Not pure black --
#: a real card photographs as a dark grey and a model trained against 0,0,0
#: would learn an edge that does not exist in the real images.
BACKGROUND = (18, 18, 20)

#: Derived from the mapping layer rather than restated, so that a change to
#: what counts as mechanical damage cannot silently disagree with the scenes
#: generated to test it.
MECHANICAL = frozenset(
    name
    for name, cls in CLASSES.items()
    if cls.admissibility is Admissibility.MECHANICAL
)


@dataclass(frozen=True)
class Cutout:
    """One kernel lifted off its background, ready to be placed."""

    rgb: np.ndarray          # (h, w, 3) uint8
    alpha: np.ndarray        # (h, w) bool
    class_name: str
    relative_mass: float
    source: Path

    @property
    def area(self) -> int:
        return int(self.alpha.sum())


#: A connected region smaller than this share of the largest one is debris
#: rather than a kernel view.
MIN_VIEW_SHARE = 0.2


def load_cutouts(sample: Sample) -> list[Cutout]:
    """Lift each kernel view out of its image using the dataset's mask.

    **A GrainSet file holds one kernel photographed from both sides, laid out
    side by side in a single image.** Confirmed visually and numerically: every
    maize mask splits into exactly two components of near-equal area, roughly
    50/50. The data card's note that "paired kernel images may not align
    perfectly" refers to these two halves, not to two separate files.

    Treating the pair as one object is wrong for compositing -- it places a
    double-width blob that no tray ever contains -- so each view is returned
    separately. Both views of a kernel share a source file and therefore a
    split, so using both cannot leak across the train/test boundary.

    Returns an empty list when the sample has no mask or the mask is empty,
    which is the caller's signal to skip it rather than an error: Corn Seeds
    ships no masks at all and is a legitimate part of the same manifest.
    """
    if sample.mask is None or not sample.mask.exists():
        return []

    rgb = np.array(Image.open(sample.path).convert("RGB"))
    mask = np.array(Image.open(sample.mask).convert("L")) > 127
    if mask.shape != rgb.shape[:2] or not mask.any():
        return []

    labelled, n = ndimage.label(mask)
    if n == 0:
        return []
    areas = ndimage.sum(np.ones_like(labelled), labelled, range(1, n + 1))
    largest = areas.max()

    cutouts: list[Cutout] = []
    for index, area in enumerate(areas, start=1):
        if area < largest * MIN_VIEW_SHARE:
            continue
        view = labelled == index
        ys, xs = np.where(view)
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        cutouts.append(
            Cutout(
                rgb=rgb[y0:y1, x0:x1],
                alpha=view[y0:y1, x0:x1],
                class_name=sample.class_name,
                relative_mass=sample.kernel_class.relative_mass or 1.0,
                source=sample.path,
            )
        )
    return cutouts


@dataclass
class Scene:
    """A composited bulk-maize image and everything known to be true about it."""

    image: np.ndarray                    # (H, W, 3) uint8
    instances: np.ndarray                # (H, W) uint16, 0 = background
    class_names: list[str] = field(default_factory=list)
    placed: int = 0
    requested: int = 0

    @property
    def truth(self):
        """The damage measurement this scene is, by construction."""
        counts: dict[str, int] = {}
        for name in self.class_names:
            counts[name] = counts.get(name, 0) + 1
        return measure(counts)

    def save(self, image_path: Path, instances_path: Path | None = None) -> None:
        Image.fromarray(self.image).save(image_path)
        if instances_path is not None:
            # uint16 so more than 255 kernels remain distinguishable.
            Image.fromarray(self.instances.astype(np.uint16)).save(instances_path)


def _prepare(
    cutout: Cutout, degrees: float, kernel_px: int | None
) -> tuple[np.ndarray, np.ndarray]:
    """Scale a cutout to the scene's kernel size, then rotate it.

    Scaling matters for more than fitting kernels on a canvas. A GrainSet
    cutout is roughly 600 px along its long axis because the P600 rig
    photographs one kernel at a time. In a real protocol frame of ~200 kernels
    a kernel spans a far smaller share of the image, so composing at native
    size would train the localiser on a magnification it will never see.
    """
    rgb = Image.fromarray(cutout.rgb)
    alpha = Image.fromarray(cutout.alpha.astype(np.uint8) * 255)

    # Erode one pixel before anything else. The dataset masks sit a shade
    # outside the kernel, so compositing them raw leaves a pale fringe of the
    # original rig's background around every kernel. On a black card that
    # fringe is a bright outline -- a perfect edge cue that exists only in
    # synthetic scenes, which is exactly the kind of shortcut a localiser will
    # learn and then lose the moment it sees a real photograph.
    alpha = alpha.filter(ImageFilter.MinFilter(3))

    if kernel_px is not None:
        long_axis = max(rgb.size)
        if long_axis > 0:
            scale = kernel_px / long_axis
            new = (max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale)))
            rgb = rgb.resize(new, Image.LANCZOS)
            alpha = alpha.resize(new, Image.NEAREST)

    rgb = rgb.rotate(degrees, resample=Image.BILINEAR, expand=True)
    alpha = alpha.rotate(degrees, resample=Image.NEAREST, expand=True)
    return np.array(rgb), np.array(alpha) > 127


def canvas_for(n_kernels: int, kernel_px: int, *, packing: float = 1.6) -> tuple[int, int]:
    """A square canvas that fits ``n_kernels`` at ``kernel_px`` without crowding.

    A kernel occupies roughly half its bounding box, and ``packing`` leaves the
    gaps a single flat layer actually has.
    """
    area = n_kernels * kernel_px * (kernel_px / 2.0) * packing
    side = int(round(area**0.5))
    return side, side


def compose(
    cutouts: list[Cutout],
    *,
    size: tuple[int, int] | None = None,
    n_kernels: int | None = None,
    kernel_px: int = 110,
    max_overlap: float = 0.15,
    seed: int = 0,
    max_attempts_per_kernel: int = 60,
) -> Scene:
    """Place kernels on a background until the scene is full.

    With ``n_kernels`` set, kernels are drawn from ``cutouts`` with replacement.
    Leave it ``None`` to place exactly the multiset given -- which is what
    :func:`pool_for_damage` returns, and the only way a damage target survives
    into the finished scene.

    ``max_overlap`` is the largest fraction of a kernel's own area that may be
    hidden by kernels placed after it. Zero produces a tidy grid that teaches
    the localiser nothing about contact; too high produces a pile, which the
    capture protocol forbids for a different reason -- kernels underneath are
    invisible, so the visible damage fraction stops matching the weighed one.
    The protocol's single-flat-layer rule corresponds to a small but non-zero
    value here, which is the default.

    Placement is by rejection sampling, and a scene may end up with fewer
    kernels than requested. That is reported in ``Scene.placed`` rather than
    silently accepted, because a scene that is 30% emptier than intended is a
    different scene -- and because dropped kernels shift the damage percentage
    away from the target, which is why the truth is read back off the finished
    scene rather than assumed.
    """
    if not cutouts:
        raise ValueError("cannot compose a scene from no cutouts")
    if n_kernels is not None and n_kernels <= 0:
        raise ValueError(f"n_kernels must be positive, got {n_kernels}")
    if not 0.0 <= max_overlap < 1.0:
        raise ValueError(f"max_overlap must be in [0, 1), got {max_overlap}")

    rng = random.Random(seed)

    if n_kernels is None:
        queue = list(cutouts)
        rng.shuffle(queue)
    else:
        queue = [rng.choice(cutouts) for _ in range(n_kernels)]

    height, width = size if size is not None else canvas_for(len(queue), kernel_px)

    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = BACKGROUND
    instances = np.zeros((height, width), dtype=np.uint16)

    scene = Scene(image=image, instances=instances, requested=len(queue))

    for cutout in queue:
        rgb, alpha = _prepare(cutout, rng.uniform(0, 360), kernel_px)
        h, w = alpha.shape
        if h >= height or w >= width:
            continue

        for _attempt in range(max_attempts_per_kernel):
            top = rng.randint(0, height - h)
            left = rng.randint(0, width - w)

            window = instances[top : top + h, left : left + w]
            covered = np.count_nonzero(window[alpha])
            if covered / alpha.sum() > max_overlap:
                continue

            # The visible part of this kernel must stay in one piece. A kernel
            # already on the card can lie across the middle of where this one
            # would go, leaving two separated slivers -- one instance, two
            # blobs, and a counting target that cannot be learned. Rejecting
            # the position is cheaper than repairing the scene afterwards.
            free = alpha & (window == 0)
            if not free.any():
                continue
            pieces, n_pieces = ndimage.label(free)
            if n_pieces > 1:
                # Keep only the largest visible piece and require it to still
                # be most of the kernel. Demanding exactly one piece rejects
                # every placement whose mask carries any stray speck, which
                # silently produced empty scenes.
                sizes = ndimage.sum(np.ones_like(pieces), pieces, range(1, n_pieces + 1))
                free = pieces == (int(np.argmax(sizes)) + 1)
            if free.sum() < alpha.sum() * (1.0 - max_overlap):
                continue

            # Paint only into pixels no kernel has claimed. Kernels abut, they
            # never stack -- which is what the capture protocol demands of a
            # real tray ("kernels may touch; they must not stack", since a
            # kernel hidden underneath is invisible and breaks the match
            # between the weighed damage fraction and the visible one).
            #
            # This is not a cosmetic choice. Letting a later kernel overwrite
            # an earlier one caps how much the *new* kernel is covered but not
            # how much it covers its neighbours, so at realistic density an
            # early kernel is nibbled by many later ones until its instance is
            # shattered into disconnected fragments. Measured before this fix:
            # every instance in a scene averaged 2.55 visible pieces, which
            # made the counting target unlearnable.
            scene.placed += 1
            scene.class_names.append(cutout.class_name)
            region = scene.image[top : top + h, left : left + w]
            region[free] = rgb[free]
            window[free] = scene.placed
            break

    return scene


def build_cutout_pool(
    samples: list[Sample], *, limit: int | None = None, seed: int = 0
) -> list[Cutout]:
    """Load cutouts for the samples that have masks.

    ``limit`` caps how many are loaded, since a pool of a few hundred distinct
    kernels already produces varied scenes and loading all 19,000 is slow.
    """
    usable = [s for s in samples if s.mask is not None]
    if not usable:
        raise ValueError("none of these samples carry a mask")
    rng = random.Random(seed)
    rng.shuffle(usable)

    pool: list[Cutout] = []
    for sample in usable:
        pool.extend(load_cutouts(sample))
        if limit is not None and len(pool) >= limit:
            break
    if not pool:
        raise ValueError("no usable cutouts -- every mask was empty or mismatched")
    return pool


def pool_for_damage(
    pool: list[Cutout], damage_pct: float, n_kernels: int, *, seed: int = 0
) -> list[Cutout]:
    """Draw a kernel multiset whose mechanical damage is near ``damage_pct``.

    The scene's exact truth still comes from :attr:`Scene.truth` after
    composition, because rejection sampling may drop kernels. This only sets
    the target.
    """
    if not 0.0 <= damage_pct <= 100.0:
        raise ValueError(f"damage must be in [0, 100] %, got {damage_pct}")

    damaged = [c for c in pool if c.class_name in MECHANICAL]
    sound = [c for c in pool if c.class_name not in MECHANICAL]
    if not damaged and damage_pct > 0:
        raise ValueError("pool contains no mechanically damaged kernels")
    if not sound and damage_pct < 100:
        raise ValueError("pool contains no undamaged kernels")

    rng = random.Random(seed)
    n_damaged = round(n_kernels * damage_pct / 100.0)
    return [rng.choice(damaged) for _ in range(n_damaged)] + [
        rng.choice(sound) for _ in range(n_kernels - n_damaged)
    ]
