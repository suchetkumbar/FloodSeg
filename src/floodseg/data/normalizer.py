"""Per-band normalization computed strictly on the training split.

Prevents data leakage across validation, test, and out-of-distribution events.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch


class ChannelNormalizer:
    """Normalizes multi-band satellite imagery using precomputed train-set stats."""

    def __init__(
        self,
        mean: Union[List[float], np.ndarray, torch.Tensor],
        std: Union[List[float], np.ndarray, torch.Tensor],
        eps: float = 1e-6,
    ):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.eps = eps

        # Check for zero std
        self.std = np.where(self.std < self.eps, 1.0, self.std)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Normalize an image array of shape [H, W, C]."""
        normalized = (image - self.mean) / (self.std + self.eps)
        # Replace any remaining NaNs or Infs with 0.0
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        return normalized.astype(np.float32)

    def normalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize a tensor of shape [C, H, W] or [B, C, H, W]."""
        mean = torch.as_tensor(self.mean, dtype=tensor.dtype, device=tensor.device)
        std = torch.as_tensor(self.std, dtype=tensor.dtype, device=tensor.device)

        if tensor.dim() == 3:
            # [C, H, W]
            mean = mean[:, None, None]
            std = std[:, None, None]
        elif tensor.dim() == 4:
            # [B, C, H, W]
            mean = mean[None, :, None, None]
            std = std[None, :, None, None]

        normalized = (tensor - mean) / (std + self.eps)
        return torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)

    def to_dict(self) -> Dict[str, List[float]]:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "eps": self.eps,
        }

    def save_json(self, path: Union[str, Path], metadata: Optional[Dict] = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        if metadata:
            payload["metadata"] = metadata
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> ChannelNormalizer:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(mean=data["mean"], std=data["std"], eps=data.get("eps", 1e-6))


    @classmethod
    def concat(cls, *normalizers: "ChannelNormalizer") -> "ChannelNormalizer":
        """Stack per-channel stats, e.g. S1 (2) + S2 (13) -> fusion (15)."""
        mean = np.concatenate([n.mean for n in normalizers])
        std = np.concatenate([n.std for n in normalizers])
        return cls(mean=mean, std=std, eps=normalizers[0].eps)


class RunningChannelStats:
    """Streaming per-channel mean/std (Chan et al. parallel Welford merge).

    Non-finite values are excluded per channel, so a NaN in one SAR band does not
    discard the co-located valid values of the other bands.
    """

    def __init__(self, num_channels: int):
        self.num_channels = num_channels
        self.count = np.zeros(num_channels, dtype=np.float64)
        self.mean = np.zeros(num_channels, dtype=np.float64)
        self.m2 = np.zeros(num_channels, dtype=np.float64)

    def update(self, image: np.ndarray) -> None:
        """Add an [H, W, C] image."""
        flat = image.reshape(-1, self.num_channels).astype(np.float64)
        finite = np.isfinite(flat)
        n_b = finite.sum(axis=0).astype(np.float64)
        safe = np.where(finite, flat, 0.0)
        sum_b = safe.sum(axis=0)
        mean_b = np.divide(sum_b, n_b, out=np.zeros_like(sum_b), where=n_b > 0)
        m2_b = (np.where(finite, flat - mean_b, 0.0) ** 2).sum(axis=0)

        n_a = self.count
        n = n_a + n_b
        delta = mean_b - self.mean
        with np.errstate(invalid="ignore", divide="ignore"):
            self.mean = np.where(n > 0, self.mean + delta * n_b / np.maximum(n, 1), self.mean)
            self.m2 = self.m2 + m2_b + np.where(n > 0, delta**2 * n_a * n_b / np.maximum(n, 1), 0.0)
        self.count = n

    def to_normalizer(self) -> ChannelNormalizer:
        if np.any(self.count == 0):
            raise ValueError("A channel had no finite pixels; cannot compute statistics.")
        std = np.sqrt(np.maximum(self.m2 / self.count, 1e-8))
        return ChannelNormalizer(mean=self.mean, std=std)


def compute_normalization_stats(
    image_paths: List[Path],
    read_fn,
    num_channels: int,
    max_samples: Optional[int] = None,
) -> ChannelNormalizer:
    """Compute per-band mean and std in one streaming pass.

    Args:
        image_paths: List of filepaths in the training split.
        read_fn: Function to load image as [H, W, C] float32 numpy array.
        num_channels: Number of input channels.
        max_samples: Optional limit on number of chips used to compute stats.
    """
    if max_samples is not None:
        image_paths = image_paths[:max_samples]
    running = RunningChannelStats(num_channels)
    for p in image_paths:
        running.update(read_fn(p))
    return running.to_normalizer()
