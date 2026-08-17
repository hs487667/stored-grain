"""Photograph in, ranked lots out.

The whole system in one place: localise every kernel, classify each one, map
the classes onto the mechanical-damage term, run the published deterioration
model, and order the lots. Each stage already exists and is tested on its own;
this is where the seams are, and the seams are where a plausible-looking wrong
number comes from.

Three properties this module is responsible for:

**The damage percentage reaching the model is mass basis.** The classifier
counts kernels. Steele's damage term is percent by weight. The mapping layer
converts, and the pipeline must pass the converted figure -- passing the count
would overstate damage wherever fragments are present.

**Biological deterioration never enters the equation.** Mould, insect damage
and sprouting are measured, reported to the user, and withheld from the model.
The multiplier was fitted on mechanical damage alone.

**The ranking declines to order what it cannot separate.** Counting is a finite
sample, so two lots whose damage differs by less than the sampling noise cannot
honestly be ordered. Ties are reported as ties rather than invented.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.mapping.classes import DamageMeasurement, measure
from src.measurement.sampling import DamageCalibration, resolution
from src.physics.deterioration import Assessment, assess
from src.vision.dataset import CLASS_NAMES, eval_transform
from src.vision.evaluate_localiser import predict_scene
from src.vision.localise import UNet, instances_from_logits

#: A predicted kernel smaller than this fraction of the median is debris.
MIN_RELATIVE_AREA = 0.15


@dataclass(frozen=True)
class CorrectedDamage:
    """Damage after the classifier's own error rates are divided out."""

    mass_pct: float
    count_pct: float


def correct_measurement(
    measurement: DamageMeasurement, calibration: DamageCalibration | None
) -> CorrectedDamage:
    """Remove the classifier's systematic bias from a damage measurement.

    The error rates are counted per kernel, so the correction belongs on the
    count basis; the mass figure follows by the same count-to-mass ratio the
    measurement already carries. Correcting the mass percentage directly would
    subtract a count-basis false-positive floor from a mass-basis number, and
    where the damaged population is mostly fragments those two live on
    different scales.

    Without a calibration the measurement passes through untouched. An
    uncorrected reading is honest -- it is simply the raw one -- and a wrong
    correction is not, so the pipeline runs without one rather than assuming.
    """
    if calibration is None:
        return CorrectedDamage(
            mass_pct=measurement.mechanical_damage_mass_pct,
            count_pct=measurement.mechanical_damage_count_pct,
        )
    if calibration.basis == "mass":
        mass_pct = calibration.correct(measurement.mechanical_damage_mass_pct)
        return CorrectedDamage(
            mass_pct=mass_pct,
            count_pct=mass_pct / measurement.count_to_mass_ratio,
        )
    count_pct = calibration.correct(measurement.mechanical_damage_count_pct)
    return CorrectedDamage(
        mass_pct=count_pct * measurement.count_to_mass_ratio,
        count_pct=count_pct,
    )


@dataclass
class LotReading:
    """Everything one photograph says about one lot."""

    lot_id: str
    kernels_counted: int
    damage_mass_pct: float
    damage_count_pct: float
    biological_pct: dict[str, float]
    class_counts: dict[str, int]

    temperature_c: float
    moisture_pct_wb: float

    assessment: Assessment = field(repr=False)

    #: What the classifier reported before its bias was divided out, and the
    #: rates that were divided out. Both ``None`` when the pipeline ran
    #: uncalibrated, which is also when ``damage_mass_pct`` is the raw figure.
    raw_damage_mass_pct: float | None = None
    raw_damage_count_pct: float | None = None
    calibration: DamageCalibration | None = None

    @property
    def resolvable_gap_pct(self) -> float:
        """Smallest damage difference this reading can honestly distinguish."""
        return resolution(
            self.damage_mass_pct, max(self.kernels_counted, 1)
        ).minimum_resolvable_gap_pct


def _crop(image: np.ndarray, instances: np.ndarray, label: int, pad: int = 4):
    ys, xs = np.where(instances == label)
    if len(ys) == 0:
        return None
    y0 = max(0, ys.min() - pad)
    y1 = min(image.shape[0], ys.max() + 1 + pad)
    x0 = max(0, xs.min() - pad)
    x1 = min(image.shape[1], xs.max() + 1 + pad)
    return Image.fromarray(image[y0:y1, x0:x1])


class Pipeline:
    def __init__(
        self,
        localiser_checkpoint: str | Path = "models/localiser.pt",
        classifier_checkpoint: str | Path = "models/classifier.session.pt",
        device: torch.device | None = None,
        batch_size: int = 128,
        calibration: DamageCalibration | str | Path | None = None,
    ):
        from torchvision import models as tv

        if isinstance(calibration, (str, Path)):
            calibration = DamageCalibration.from_dict(
                json.loads(Path(calibration).read_text())
            )
        self.calibration = calibration

        self.device = device or (
            torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        )
        self.batch_size = batch_size

        self.localiser = UNet().to(self.device)
        self.localiser.load_state_dict(
            torch.load(localiser_checkpoint, map_location=self.device)["model"]
        )
        self.localiser.eval()

        state = torch.load(classifier_checkpoint, map_location=self.device)
        classifier = tv.resnet50()
        classifier.fc = torch.nn.Linear(classifier.fc.in_features, len(CLASS_NAMES))
        classifier.load_state_dict(state["model"])
        self.classifier = classifier.to(self.device).eval()
        self.classes = state.get("classes", CLASS_NAMES)
        self.transform = eval_transform()

    @torch.no_grad()
    def read(
        self,
        image_path: str | Path,
        *,
        lot_id: str,
        temperature_c: float,
        moisture_pct_wb: float,
    ) -> LotReading:
        image = np.array(Image.open(image_path).convert("RGB"))

        instances = instances_from_logits(
            predict_scene(self.localiser, image, self.device)
        )
        labels = [int(l) for l in np.unique(instances) if l != 0]
        if not labels:
            raise ValueError(f"no kernels found in {image_path}")

        # Drop debris relative to the kernels actually present rather than to a
        # fixed pixel count, so the same threshold holds at any magnification.
        areas = {l: int((instances == l).sum()) for l in labels}
        floor = np.median(list(areas.values())) * MIN_RELATIVE_AREA
        labels = [l for l in labels if areas[l] >= floor]

        crops, kept = [], []
        for label in labels:
            crop = _crop(image, instances, label)
            if crop is not None:
                crops.append(self.transform(crop))
                kept.append(label)

        predictions: list[int] = []
        for start in range(0, len(crops), self.batch_size):
            batch = torch.stack(crops[start : start + self.batch_size]).to(self.device)
            predictions.extend(self.classifier(batch).argmax(1).cpu().tolist())

        counts: dict[str, int] = {}
        for index in predictions:
            name = self.classes[index]
            counts[name] = counts.get(name, 0) + 1

        measurement = measure(counts)
        corrected = correct_measurement(measurement, self.calibration)
        assessment = assess(corrected.mass_pct, temperature_c, moisture_pct_wb)

        return LotReading(
            lot_id=lot_id,
            kernels_counted=measurement.kernels_counted,
            damage_mass_pct=corrected.mass_pct,
            damage_count_pct=corrected.count_pct,
            biological_pct=measurement.biological_count_pct,
            class_counts=counts,
            temperature_c=temperature_c,
            moisture_pct_wb=moisture_pct_wb,
            assessment=assessment,
            raw_damage_mass_pct=measurement.mechanical_damage_mass_pct,
            raw_damage_count_pct=measurement.mechanical_damage_count_pct,
            calibration=self.calibration,
        )


@dataclass
class RankedLot:
    reading: LotReading
    rank: int

    #: Lots this one could not be separated from. A tie is a result: it says
    #: the damage difference is inside the sampling noise of the measurement.
    tied_with: tuple[str, ...] = ()


def rank_lots(readings: list[LotReading]) -> list[RankedLot]:
    """Order lots fastest-degrading first, refusing to split unresolvable pairs.

    Ordering is by degradation rate from the deterioration model, not by damage
    directly, because temperature and moisture belong in the comparison too.
    Ties are decided on the damage measurement, since that is the quantity
    carrying sampling error -- the environmental inputs are typed in by the user
    and carry no counting noise.
    """
    ordered = sorted(
        readings, key=lambda r: r.assessment.degradation_rate, reverse=True
    )
    ranked: list[RankedLot] = []
    for i, reading in enumerate(ordered):
        tied = []
        for other in ordered:
            if other is reading:
                continue
            gap = abs(other.damage_mass_pct - reading.damage_mass_pct)
            if gap < max(reading.resolvable_gap_pct, other.resolvable_gap_pct):
                tied.append(other.lot_id)
        ranked.append(RankedLot(reading=reading, rank=i + 1, tied_with=tuple(tied)))
    return ranked
