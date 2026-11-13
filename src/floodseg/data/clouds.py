"""Cloud handling for Sentinel-2: a simple cloud proxy and physically plausible synthetic clouds.

Why synthetic clouds?  Sen1Floods11 hand labels mark clouded pixels as no-data (-1),
so real chips cannot measure how a model behaves *under* cloud. Instead we occlude
the optical bands of labelled pixels with synthetic thick cloud and keep the true
label, which directly tests the core question: how much does SAR help when optical
imagery is blocked?

Clouds are applied to raw Sentinel-2 L1C TOA values (reflectance x 10,000) *before*
normalisation, and only to S2 channels. SAR channels are untouched because C-band
radar penetrates cloud.
"""

from __future__ import annotations

import zlib
from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

# Median TOA reflectance of thick cloud (pixels with B2, B3, B4 > 0.35) over the
# official train split (93 chips, 30,584 sampled pixels). Order: B1 ... B12.
THICK_CLOUD_TOA = np.array(
    [0.458, 0.458, 0.449, 0.469, 0.462, 0.529, 0.579, 0.530, 0.608, 0.086, 0.004, 0.484, 0.357],
    dtype=np.float32,
) * 1e4

# Band positions in the 13-band S2 cube
_B2, _B3, _B4, _B10 = 1, 2, 3, 10


def cloud_proxy_mask(s2_dn: np.ndarray) -> np.ndarray:
    """Heuristic cloud mask on a [H, W, 13] L1C TOA cube (DN = reflectance x 1e4).

    Cloud if all of blue/green/red > 0.18 (bright, spectrally flat) or cirrus B10 > 0.012.
    On the official train split it agrees only moderately with the labellers' no-data
    mask (P(no-data | proxy) = 0.40, P(proxy | no-data) = 0.58), so it is used only
    to stratify chips, never as ground truth.
    """
    r = s2_dn.astype(np.float32) / 1e4
    bright = (r[..., _B2] > 0.18) & (r[..., _B3] > 0.18) & (r[..., _B4] > 0.18)
    return bright | (r[..., _B10] > 0.012)


def _smooth_field(shape: tuple, sigma: float, rng: np.random.Generator, factor: int = 4) -> np.ndarray:
    """Standardised Gaussian random field. Generated at 1/``factor`` resolution and
    bilinearly upsampled: visually identical for large sigma and ~factor^2 cheaper."""
    h, w = shape
    lo = (max(2, -(-h // factor)), max(2, -(-w // factor)))
    f = gaussian_filter(rng.standard_normal(lo).astype(np.float32), sigma=sigma / factor, mode="wrap")
    f = zoom(f, (h / lo[0], w / lo[1]), order=1)[:h, :w]
    return (f - f.mean()) / (f.std() + 1e-8)


def synthetic_cloud_alpha(
    shape: tuple, coverage: float, rng: np.random.Generator, sigma: float = 24.0, edge: float = 0.15
) -> np.ndarray:
    """Cloud opacity field in [0, 1] with ~``coverage`` of pixels at opacity >= 0.5.

    A Gaussian-smoothed random field is thresholded at its (1 - coverage) quantile,
    with a soft ramp of width ``edge`` (in field std units) for semi-transparent borders.
    """
    if coverage <= 0:
        return np.zeros(shape, dtype=np.float32)
    if coverage >= 1:
        return np.ones(shape, dtype=np.float32)
    field = _smooth_field(shape, sigma, rng)
    thr = np.quantile(field, 1.0 - coverage)
    return np.clip(0.5 + (field - thr) / (2 * edge), 0.0, 1.0).astype(np.float32)


def apply_synthetic_clouds(
    image: np.ndarray, alpha: np.ndarray, s2_start: int, rng: np.random.Generator
) -> np.ndarray:
    """Blend thick-cloud spectra into the S2 channels [s2_start : s2_start + 13] of a raw
    [H, W, C] image. Brightness varies smoothly (+-20%) to mimic cloud texture."""
    if not np.any(alpha > 0):
        return image
    out = image.astype(np.float32, copy=True)
    texture = 1.0 + 0.2 * np.tanh(_smooth_field(alpha.shape, 8.0, rng))
    cloud = THICK_CLOUD_TOA[None, None, :] * texture[..., None]
    a = alpha[..., None]
    s2 = out[..., s2_start : s2_start + 13]
    out[..., s2_start : s2_start + 13] = (1.0 - a) * s2 + a * cloud
    return out


def s2_start_for_modality(modality: str) -> Optional[int]:
    """Index of the first S2 channel in a modality's stack (None for SAR-only)."""
    return {"s1": None, "s2": 0, "fusion": 2}[modality]


class FixedCoverageClouds:
    """Deterministic evaluation clouds: same field for a given (chip, coverage, seed)."""

    def __init__(self, coverage: float, seed: int = 0):
        self.coverage = float(coverage)
        self.seed = seed

    def __call__(self, image: np.ndarray, modality: str, chip_id: str) -> np.ndarray:
        start = s2_start_for_modality(modality)
        if start is None or self.coverage <= 0:
            return image
        key = zlib.crc32(f"{chip_id}|{self.coverage:.3f}|{self.seed}".encode())
        rng = np.random.default_rng(key)
        alpha = synthetic_cloud_alpha(image.shape[:2], self.coverage, rng)
        return apply_synthetic_clouds(image, alpha, start, rng)


class RandomCloudAugmentation:
    """Training augmentation: with probability ``p``, add clouds of coverage ~ U(0, max_coverage).

    Teaches optical/fusion models what to do when the optical signal is blocked
    (for fusion: fall back on SAR). Randomness comes from torch's per-worker RNG so
    DataLoader workers draw different clouds.
    """

    def __init__(self, p: float = 0.5, max_coverage: float = 1.0):
        self.p = p
        self.max_coverage = max_coverage

    def __call__(self, image: np.ndarray, modality: str, chip_id: str) -> np.ndarray:
        import torch

        start = s2_start_for_modality(modality)
        if start is None or self.p <= 0:
            return image
        rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,))))
        if rng.random() >= self.p:
            return image
        alpha = synthetic_cloud_alpha(image.shape[:2], rng.uniform(0.0, self.max_coverage), rng)
        return apply_synthetic_clouds(image, alpha, start, rng)
