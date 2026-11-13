"""Evaluation metrics for flood water segmentation.

Strictly excludes no-data / invalid pixels (-1) from all computations.
Calculates Water IoU, F1-Score, Precision, and Recall.

Two aggregation schemes are supported and must not be mixed up:
* **Pooled (micro) IoU** – confusion counts are summed over all chips, then IoU is
  computed once. This is the headline metric (matches Bonafilia et al., 2020).
* **Chip-mean (macro) IoU** – IoU is computed per chip, then averaged.

Bootstrap confidence intervals are computed by resampling *chips* (the independent
unit) and recomputing the same aggregate, so the CI always brackets its own
point estimate.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor]


def _squeeze_channel(x: ArrayLike) -> ArrayLike:
    """Drop a singleton channel dim: [B, 1, H, W] -> [B, H, W]."""
    if x.ndim == 4 and x.shape[1] == 1:
        return x[:, 0]
    return x


def confusion_counts(
    pred: ArrayLike, target: ArrayLike, ignore_index: int = -1
) -> Tuple[int, int, int, int]:
    """Return (tp, fp, fn, tn) over valid pixels. Works on numpy or torch (any device)."""
    pred = _squeeze_channel(pred)
    target = _squeeze_channel(target)
    if pred.shape != target.shape:
        raise ValueError(f"pred shape {tuple(pred.shape)} != target shape {tuple(target.shape)}")

    if isinstance(pred, torch.Tensor) or isinstance(target, torch.Tensor):
        pred_t = torch.as_tensor(pred)
        target_t = torch.as_tensor(target, device=pred_t.device)
        valid = target_t != ignore_index
        p = (pred_t == 1) & valid
        t = (target_t == 1) & valid
        tp = int((p & t).sum().item())
        fp = int((p & ~t).sum().item())
        fn = int((~p & t & valid).sum().item())
        tn = int(valid.sum().item()) - tp - fp - fn
        return tp, fp, fn, tn

    valid = target != ignore_index
    p = (pred == 1) & valid
    t = (target == 1) & valid
    tp = int(np.count_nonzero(p & t))
    fp = int(np.count_nonzero(p & ~t))
    fn = int(np.count_nonzero(~p & t))
    tn = int(np.count_nonzero(valid)) - tp - fp - fn
    return tp, fp, fn, tn


def metrics_from_counts(tp: float, fp: float, fn: float, tn: float = 0.0) -> Dict[str, float]:
    """Compute IoU / F1 / precision / recall / accuracy from confusion counts."""

    def _div(a: float, b: float) -> float:
        return float(a / b) if b > 0 else 0.0

    total = tp + fp + fn + tn
    return {
        "iou": _div(tp, tp + fp + fn),
        "f1": _div(2 * tp, 2 * tp + fp + fn),
        "precision": _div(tp, tp + fp),
        "recall": _div(tp, tp + fn),
        "accuracy": _div(tp + tn, total),
    }


class SegmentationMetrics:
    """Accumulator for binary segmentation metrics with no-data masking.

    Also keeps per-chip confusion counts so pooled-IoU bootstrap CIs and
    chip-mean IoU can be derived from a single pass.
    """

    def __init__(self, ignore_index: int = -1, track_per_chip: bool = False):
        self.ignore_index = ignore_index
        self.track_per_chip = track_per_chip
        self.reset()

    def reset(self) -> None:
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        self.total_valid_pixels = 0
        self.per_chip: List[Tuple[int, int, int, int]] = []

    def update(self, pred: ArrayLike, target: ArrayLike) -> None:
        """Update confusion counts.

        Args:
            pred: Binary predictions (0/1), shape [H, W], [B, H, W] or [B, 1, H, W].
            target: Ground truth (-1/0/1), same spatial shape as ``pred``.
        """
        pred = _squeeze_channel(pred)
        target = _squeeze_channel(target)

        if self.track_per_chip and pred.ndim == 3:
            for p, t in zip(pred, target):
                self._add(*confusion_counts(p, t, self.ignore_index))
        else:
            self._add(*confusion_counts(pred, target, self.ignore_index))

    def _add(self, tp: int, fp: int, fn: int, tn: int) -> None:
        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.tn += tn
        self.total_valid_pixels += tp + fp + fn + tn
        if self.track_per_chip:
            self.per_chip.append((tp, fp, fn, tn))

    def compute(self) -> Dict[str, float]:
        """Compute aggregate (pooled) metrics over all accumulated samples."""
        res: Dict[str, float] = metrics_from_counts(self.tp, self.fp, self.fn, self.tn)
        res.update(
            {
                "tp": self.tp,
                "fp": self.fp,
                "fn": self.fn,
                "tn": self.tn,
                "total_pixels": self.total_valid_pixels,
            }
        )
        return res

    def per_chip_counts(self) -> np.ndarray:
        """Per-chip confusion counts as an [N, 4] int64 array (tp, fp, fn, tn)."""
        return np.asarray(self.per_chip, dtype=np.int64).reshape(-1, 4)


def iou_from_counts(tp: int, fp: int, fn: int) -> float:
    """IoU for a single chip.

    A chip with no water in the label and no water predicted is a perfect
    prediction (IoU = 1.0); this convention only affects chip-mean IoU.
    """
    union = tp + fp + fn
    if union == 0:
        return 1.0
    return float(tp / union)


def compute_sample_iou(
    pred: np.ndarray,
    target: np.ndarray,
    ignore_index: int = -1,
) -> float:
    """Compute Water class IoU for a single chip, ignoring -1."""
    tp, fp, fn, _ = confusion_counts(pred, target, ignore_index)
    return iou_from_counts(tp, fp, fn)


def summarize_with_ci(
    per_chip_counts: np.ndarray,
    n_bootstraps: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Pooled metrics, chip-mean IoU, and chip-level bootstrap CIs for both.

    Args:
        per_chip_counts: [N, 4] array of (tp, fp, fn, tn) per chip.
    """
    counts = np.asarray(per_chip_counts, dtype=np.int64).reshape(-1, 4)
    n = len(counts)
    tot = counts.sum(axis=0)
    res: Dict[str, float] = metrics_from_counts(*tot.tolist())
    res.update({k: int(v) for k, v in zip(("tp", "fp", "fn", "tn"), tot.tolist())})

    chip_ious = np.array([iou_from_counts(tp, fp, fn) for tp, fp, fn, _ in counts])
    res["chip_mean_iou"] = float(chip_ious.mean()) if n else 0.0
    res["num_chips"] = n

    alpha = (1.0 - confidence_level) / 2.0
    if n == 0:
        res.update(ci_95_lower=0.0, ci_95_upper=0.0, chip_mean_ci_95_lower=0.0, chip_mean_ci_95_upper=0.0)
        return res

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_bootstraps, n))
    boot = counts[idx].sum(axis=1)  # [B, 4]
    denom = boot[:, 0] + boot[:, 1] + boot[:, 2]
    boot_iou = np.where(denom > 0, boot[:, 0] / np.maximum(denom, 1), 0.0)
    boot_chip_mean = chip_ious[idx].mean(axis=1)

    res["ci_95_lower"] = float(np.quantile(boot_iou, alpha))
    res["ci_95_upper"] = float(np.quantile(boot_iou, 1.0 - alpha))
    res["chip_mean_ci_95_lower"] = float(np.quantile(boot_chip_mean, alpha))
    res["chip_mean_ci_95_upper"] = float(np.quantile(boot_chip_mean, 1.0 - alpha))
    return res


def bootstrap_ci(
    scores: Sequence[float],
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap mean and CI of a list of chip-level scores (chip-mean aggregate)."""
    scores_arr = np.asarray(scores, dtype=np.float64)
    if len(scores_arr) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(scores_arr), size=(n_bootstraps, len(scores_arr)))
    boot_means = scores_arr[idx].mean(axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    return (
        float(scores_arr.mean()),
        float(np.quantile(boot_means, alpha)),
        float(np.quantile(boot_means, 1.0 - alpha)),
    )
