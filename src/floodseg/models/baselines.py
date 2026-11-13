"""Non-learned remote sensing baselines for optical and SAR flood segmentation.

Includes:
1. NDWI  (Green - NIR)  / (Green + NIR),  water if index > tau (tau tuned on train).
2. MNDWI (Green - SWIR1)/ (Green + SWIR1), water if index > tau (tau tuned on train).
3. SAR global threshold on Sentinel-1 VV (dB), water if VV < tau (tau tuned on train).
4. Otsu scene-adaptive threshold on VV (dB), no tuning (per-chip bimodal split).

Threshold tuning maximises **pooled** water IoU over the training split, matching
the headline evaluation metric. Tuning is streaming: each chip contributes exact
(tp, fp, fn) counts for every candidate threshold, so the full split never has
to be held in memory.
"""

from __future__ import annotations

import logging
from typing import Iterable, Literal, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Sen1Floods11 S2Hand band order: B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B10, B11, B12
S2_GREEN, S2_NIR, S2_SWIR1 = 2, 7, 11
S1_VV = 0


def compute_otsu_threshold(image: np.ndarray, nbins: int = 256) -> float:
    """Pure NumPy implementation of Otsu's bimodal thresholding.

    Maximizes between-class variance without requiring scikit-image dependency.
    """
    flat = image.ravel()
    flat = flat[np.isfinite(flat)]
    if len(flat) == 0 or np.ptp(flat) < 1e-4:
        return float(np.mean(flat)) if len(flat) > 0 else 0.0

    counts, bin_edges = np.histogram(flat, bins=nbins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    p = counts / counts.sum()
    weight1 = np.cumsum(p)
    weight2 = 1.0 - weight1
    cum_mean = np.cumsum(p * bin_centers)
    total_mean = cum_mean[-1]
    mean1 = cum_mean / np.maximum(weight1, 1e-12)
    mean2 = (total_mean - cum_mean) / np.maximum(weight2, 1e-12)

    variance = weight1 * weight2 * ((mean1 - mean2) ** 2)
    return float(bin_centers[int(np.argmax(variance))])


def _band(cube: np.ndarray, idx: int, n_bands: int) -> np.ndarray:
    """Extract band ``idx`` from a [H, W, C] or [C, H, W] cube with C == n_bands."""
    if cube.ndim == 3 and cube.shape[-1] == n_bands:
        return cube[:, :, idx].astype(np.float32)
    if cube.ndim == 3 and cube.shape[0] == n_bands:
        return cube[idx].astype(np.float32)
    raise ValueError(f"Expected a {n_bands}-band cube, got shape {cube.shape}")


class ThresholdBaseline:
    """A scalar index image plus a single global decision threshold."""

    name: str = "threshold"
    water_if: Literal["above", "below"] = "above"
    default_grid: np.ndarray = np.round(np.linspace(-0.5, 0.5, 101), 4)

    def __init__(self, threshold: float = 0.0):
        self.threshold = float(threshold)

    def compute_index(self, cube: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def _decide(self, index: np.ndarray, threshold: float) -> np.ndarray:
        with np.errstate(invalid="ignore"):
            water = index > threshold if self.water_if == "above" else index < threshold
        # Non-finite index values (e.g. SAR NaNs) are never water.
        return (water & np.isfinite(index)).astype(np.int64)

    def predict(self, cube: np.ndarray) -> np.ndarray:
        """Predict binary water mask (1: water, 0: not water)."""
        return self._decide(self.compute_index(cube), self.threshold)

    def threshold_counts(
        self, cube: np.ndarray, label: np.ndarray, thresholds: np.ndarray
    ) -> np.ndarray:
        """Exact (tp, fp, fn) for every candidate threshold on one chip -> [T, 3]."""
        index = self.compute_index(cube)
        valid = label != -1
        idx_v = index[valid]
        lbl_v = label[valid]
        finite = np.isfinite(idx_v)
        water_vals = np.sort(idx_v[(lbl_v == 1) & finite])
        land_vals = np.sort(idx_v[(lbl_v == 0) & finite])
        n_water_total = int(np.count_nonzero(lbl_v == 1))

        if self.water_if == "above":
            tp = len(water_vals) - np.searchsorted(water_vals, thresholds, side="right")
            fp = len(land_vals) - np.searchsorted(land_vals, thresholds, side="right")
        else:
            tp = np.searchsorted(water_vals, thresholds, side="left")
            fp = np.searchsorted(land_vals, thresholds, side="left")
        fn = n_water_total - tp
        return np.stack([tp, fp, fn], axis=1).astype(np.int64)

    def tune_threshold(
        self,
        images: Iterable[np.ndarray],
        labels: Iterable[np.ndarray],
        thresholds: Optional[np.ndarray] = None,
    ) -> float:
        """Pick the threshold maximising pooled water IoU over (image, label) pairs."""
        grid = self.default_grid if thresholds is None else np.asarray(thresholds, dtype=np.float64)
        acc = np.zeros((len(grid), 3), dtype=np.int64)
        for img, lbl in zip(images, labels):
            acc += self.threshold_counts(img, lbl, grid)
        return self.set_threshold_from_counts(acc, grid)

    def set_threshold_from_counts(self, acc: np.ndarray, grid: np.ndarray) -> float:
        denom = acc.sum(axis=1)
        ious = np.where(denom > 0, acc[:, 0] / np.maximum(denom, 1), 0.0)
        best = int(np.argmax(ious))
        self.threshold = float(grid[best])
        logger.info(
            "Tuned %s threshold: %.4f (pooled train IoU: %.4f)", self.name, self.threshold, ious[best]
        )
        return self.threshold


class NDWIBaseline(ThresholdBaseline):
    """Normalized Difference Water Index: (Green - NIR) / (Green + NIR) on S2 B3, B8."""

    name = "NDWI"
    water_if = "above"

    def __init__(self, threshold: float = 0.0, eps: float = 1e-6):
        super().__init__(threshold)
        self.eps = eps

    def compute_index(self, s2_cube: np.ndarray) -> np.ndarray:
        green = _band(s2_cube, S2_GREEN, 13)
        nir = _band(s2_cube, S2_NIR, 13)
        ndwi = (green - nir) / (green + nir + self.eps)
        return np.nan_to_num(ndwi, nan=-1.0, posinf=1.0, neginf=-1.0)


class MNDWIBaseline(ThresholdBaseline):
    """Modified NDWI: (Green - SWIR1) / (Green + SWIR1) on S2 B3, B11."""

    name = "MNDWI"
    water_if = "above"

    def __init__(self, threshold: float = 0.0, eps: float = 1e-6):
        super().__init__(threshold)
        self.eps = eps

    def compute_index(self, s2_cube: np.ndarray) -> np.ndarray:
        green = _band(s2_cube, S2_GREEN, 13)
        swir = _band(s2_cube, S2_SWIR1, 13)
        mndwi = (green - swir) / (green + swir + self.eps)
        return np.nan_to_num(mndwi, nan=-1.0, posinf=1.0, neginf=-1.0)


class SARThresholdBaseline(ThresholdBaseline):
    """Single global VV (dB) threshold tuned on train: water if VV < tau.

    Calm water is a specular reflector, so it appears dark (low backscatter).
    """

    name = "SAR_VV_Threshold"
    water_if = "below"
    default_grid = np.round(np.linspace(-25.0, -8.0, 171), 4)

    def __init__(self, threshold: float = -16.0):
        super().__init__(threshold)

    def compute_index(self, s1_cube: np.ndarray) -> np.ndarray:
        return _band(s1_cube, S1_VV, 2)


class OtsuSARBaseline:
    """Scene-adaptive Otsu threshold on Sentinel-1 VV backscatter (dB). No tuning."""

    name = "Otsu_SAR_Adaptive"

    def __init__(self, fallback_threshold: float = -16.0, clip_range=(-25.0, -8.0)):
        self.fallback_threshold = fallback_threshold
        self.clip_range = clip_range

    def chip_threshold(self, vv: np.ndarray) -> float:
        valid_vv = vv[np.isfinite(vv)]
        if len(valid_vv) < 100 or np.ptp(valid_vv) < 1e-3:
            return self.fallback_threshold
        thresh = compute_otsu_threshold(valid_vv)
        # Bound threshold to a physically plausible range for C-band SAR
        return float(np.clip(thresh, *self.clip_range))

    def predict(self, s1_cube: np.ndarray) -> np.ndarray:
        vv = _band(s1_cube, S1_VV, 2)
        thresh = self.chip_threshold(vv)
        with np.errstate(invalid="ignore"):
            return ((vv < thresh) & np.isfinite(vv)).astype(np.int64)

    # Backwards-compatible alias
    predict_adaptive = predict
