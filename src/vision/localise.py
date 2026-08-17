"""Find every kernel in a tray photograph, including the ones that touch.

The damage percentage is a ratio of counts, so the localiser's job is not to
draw pretty outlines -- it is to get the count right. Two kernels merged into
one, or one kernel split into two, moves the denominator and therefore the
number the deterioration model is handed. Boundary accuracy matters only in so
far as it produces the right number of objects.

The approach is boundary-aware segmentation rather than a detector. A network
predicts three things per pixel -- background, kernel interior, kernel border --
and the interiors become watershed seeds that grow back out to the full mask.
Predicting the border explicitly is what separates touching kernels: the seam
between two kernels is exactly the region the network learns to mark, and
eroding it away leaves two distinct seeds where a plain foreground mask would
leave one blob.

This is trainable at all only because the scenes are synthetic. GrainSet gives
19,000 kernels with exact masks, and compositing them produces bulk images
whose instance labels are perfect and free -- which no public dataset of
touching kernels provides. See ``src.vision.synth`` for what those scenes can
and cannot teach.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BACKGROUND, INTERIOR, BORDER = 0, 1, 2


# --- Targets ---------------------------------------------------------------

def instance_targets(instances: np.ndarray, border_width: int = 2) -> np.ndarray:
    """Turn an instance map into per-pixel background / interior / border.

    The border is the rim of each instance rather than only the seams where two
    kernels meet. Marking every rim keeps the target consistent whether or not
    a kernel happens to have a neighbour, so the network learns "edge of
    kernel" instead of "place where two kernels collide" -- and an isolated
    kernel in a sparse corner of a real tray still gets segmented correctly.
    """
    from scipy import ndimage

    target = np.zeros(instances.shape, dtype=np.int64)
    target[instances > 0] = INTERIOR

    for label in np.unique(instances):
        if label == 0:
            continue
        mask = instances == label
        eroded = ndimage.binary_erosion(mask, iterations=border_width)
        target[mask & ~eroded] = BORDER

    return target


# --- Model -----------------------------------------------------------------

def _block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """A small U-Net. Deliberately small.

    The task is texture-and-edge shaped rather than semantic -- there is one
    object class and it is always a maize kernel on a dark card -- so capacity
    is better spent on resolution than on depth. A larger backbone would also
    overfit synthetic compositing artefacts faster than it would learn kernel
    shape.
    """

    def __init__(self, channels=(32, 64, 128, 256), n_classes: int = 3):
        super().__init__()
        self.downs = nn.ModuleList()
        prev = 3
        for ch in channels:
            self.downs.append(_block(prev, ch))
            prev = ch

        self.bottleneck = _block(prev, prev * 2)

        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        prev = prev * 2
        for ch in reversed(channels):
            self.up_convs.append(nn.ConvTranspose2d(prev, ch, 2, stride=2))
            self.ups.append(_block(ch * 2, ch))
            prev = ch

        self.head = nn.Conv2d(prev, n_classes, 1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)

        x = self.bottleneck(x)

        for up_conv, up, skip in zip(self.up_convs, self.ups, reversed(skips)):
            x = up_conv(x)
            x = up(torch.cat([x, skip], dim=1))

        return self.head(x)


# --- Turning a prediction into kernels -------------------------------------

def instances_from_logits(
    logits: torch.Tensor, *, min_area: int = 60, interior_threshold: float = 0.5
) -> np.ndarray:
    """Watershed the predicted interiors back out to whole kernels.

    ``min_area`` discards specks. It is a real decision, not a tidy-up: a
    fragment is a legitimate kernel object and is *smaller* than a sound one,
    so setting this too high deletes exactly the class the damage term counts.
    """
    from scipy import ndimage
    from skimage.segmentation import watershed

    probs = torch.softmax(logits, dim=0).cpu().numpy()
    interior = probs[INTERIOR] > interior_threshold
    foreground = (probs[INTERIOR] + probs[BORDER]) > interior_threshold

    markers, _ = ndimage.label(interior)
    if markers.max() == 0:
        return np.zeros_like(markers)

    labels = watershed(-probs[INTERIOR], markers, mask=foreground)

    for label in np.unique(labels):
        if label == 0:
            continue
        area = int((labels == label).sum())
        if area < min_area:
            labels[labels == label] = 0

    return labels


# --- Metrics ---------------------------------------------------------------

def count_error(predicted: np.ndarray, truth: np.ndarray) -> dict:
    """How far off the kernel count is -- the number that moves the damage ratio."""
    n_pred = len(set(np.unique(predicted)) - {0})
    n_true = len(set(np.unique(truth)) - {0})
    return {
        "predicted": n_pred,
        "true": n_true,
        "error": n_pred - n_true,
        "relative_error": (n_pred - n_true) / n_true if n_true else 0.0,
    }


def instance_f1(predicted: np.ndarray, truth: np.ndarray, iou_threshold: float = 0.5) -> dict:
    """Match predicted kernels to true ones by intersection over union.

    A predicted kernel counts as correct when it overlaps a true kernel by more
    than ``iou_threshold`` and no other prediction has already claimed it.
    """
    true_labels = [l for l in np.unique(truth) if l != 0]
    pred_labels = [l for l in np.unique(predicted) if l != 0]
    if not true_labels:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "matched": 0}

    claimed: set[int] = set()
    matched = 0
    for pred in pred_labels:
        pred_mask = predicted == pred
        overlapping = np.unique(truth[pred_mask])
        best_iou, best_label = 0.0, None
        for label in overlapping:
            if label == 0 or label in claimed:
                continue
            true_mask = truth == label
            union = np.count_nonzero(pred_mask | true_mask)
            iou = np.count_nonzero(pred_mask & true_mask) / union if union else 0.0
            if iou > best_iou:
                best_iou, best_label = iou, label
        if best_iou >= iou_threshold and best_label is not None:
            claimed.add(int(best_label))
            matched += 1

    precision = matched / len(pred_labels) if pred_labels else 0.0
    recall = matched / len(true_labels)
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1, "matched": matched}
