"""PyTorch Dataset implementation for Sen1Floods11.

Supports S1-only (2 bands), S2-only (13 bands), and fused S1+S2 (15 bands).
Handles no-data masking (-1) and train-only band normalization.
"""

from __future__ import annotations

import csv
import logging
import os
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np

try:
    import rasterio

    HAS_RASTERIO = True
except (ImportError, OSError):
    HAS_RASTERIO = False

import tifffile
import torch
from torch.utils.data import Dataset

from floodseg.data.normalizer import ChannelNormalizer
from floodseg.data.transforms import apply_transforms

logger = logging.getLogger(__name__)

ModalityType = Literal["s1", "s2", "fusion"]


class Sen1Floods11Dataset(Dataset):
    """Dataset class for Sen1Floods11 hand-labeled flood chips."""

    # Default Sentinel-2 band wavelengths & names
    S2_BANDS = [
        "B1_Coastal",
        "B2_Blue",
        "B3_Green",
        "B4_Red",
        "B5_RedEdge1",
        "B6_RedEdge2",
        "B7_RedEdge3",
        "B8_NIR",
        "B8A_NarrowNIR",
        "B9_WaterVapor",
        "B10_Cirrus",
        "B11_SWIR1",
        "B12_SWIR2",
    ]
    S1_BANDS = ["VV", "VH"]

    def __init__(
        self,
        root_dir: Union[str, Path],
        split_file: Union[str, Path],
        modality: ModalityType = "fusion",
        transform: Optional[Callable] = None,
        normalizer: Optional[ChannelNormalizer] = None,
        s2_band_indices: Optional[List[int]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        crop_size: Optional[Tuple[int, int]] = None,
    ):
        """
        Args:
            root_dir: Root directory containing raw/sen1floods11.
            split_file: Path to split CSV file.
            modality: One of 's1' (2ch), 's2' (13ch), or 'fusion' (15ch).
            transform: Albumentations compose transform pipeline.
            normalizer: Pre-computed ChannelNormalizer (from train split only).
            s2_band_indices: Optional subset of S2 bands (0-indexed). Defaults to all 13.
            cache_dir: Optional directory of decoded chips (``{chip_id}.npy`` = raw
                [H, W, 15] S1+S2 stack in float16, ``{chip_id}_label.npy`` = int8 label).
                Missing entries are written on first access. Float16 keeps ~3 significant
                digits (<0.05% relative error), far below sensor noise, and makes data
                loading much faster than GeoTIFF decoding.
            crop_size: Optional (h, w) random crop applied before augmentation (training).
        """
        self.root_dir = Path(root_dir)
        self.split_file = Path(split_file)
        self.modality = modality
        self.transform = transform
        self.normalizer = normalizer
        self.s2_band_indices = s2_band_indices or list(range(13))
        self.crop_size = tuple(crop_size) if crop_size is not None else None
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.samples = self._load_samples()
        if len(self.samples) == 0:
            logger.warning("No samples found in split file: %s", self.split_file)

    def _load_samples(self) -> List[Dict[str, str]]:
        samples = []
        if not self.split_file.exists():
            raise FileNotFoundError(f"Split file does not exist: {self.split_file}")

        with open(self.split_file, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or len(row) < 2:
                    continue
                s1_filename = row[0].strip()
                label_filename = row[1].strip()
                s2_filename = (
                    row[2].strip()
                    if len(row) >= 3
                    else s1_filename.replace("S1Hand", "S2Hand").replace("_S1.tif", "_S2Hand.tif")
                )

                base_id = s1_filename.replace("_S1Hand.tif", "").replace("_S1.tif", "")
                country = base_id.split("_")[0]

                samples.append(
                    {
                        "chip_id": base_id,
                        "event": country,
                        "s1_path": str(
                            self.root_dir / "raw" / "sen1floods11" / "S1Hand" / s1_filename
                        ),
                        "s2_path": str(
                            self.root_dir / "raw" / "sen1floods11" / "S2Hand" / s2_filename
                        ),
                        "label_path": str(
                            self.root_dir / "raw" / "sen1floods11" / "LabelHand" / label_filename
                        ),
                    }
                )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _read_geotiff(path: str) -> np.ndarray:
        """Read a GeoTIFF using rasterio or tifffile into a [H, W, C] float32 array."""
        if HAS_RASTERIO:
            try:
                with rasterio.open(path) as src:
                    data = src.read()  # [C, H, W]
                    data = np.moveaxis(data, 0, -1)  # [H, W, C]
                    return data.astype(np.float32)
            except Exception:
                pass

        # Resilient fallback to tifffile
        data = tifffile.imread(path)
        if data.ndim == 2:
            return data[:, :, np.newaxis].astype(np.float32)
        elif data.ndim == 3:
            # If shape is [C, H, W], convert to [H, W, C]
            if data.shape[0] < data.shape[-1] and data.shape[0] in [1, 2, 3, 4, 12, 13, 15]:
                data = np.moveaxis(data, 0, -1)
            return data.astype(np.float32)
        return data.astype(np.float32)

    @staticmethod
    def _read_label(path: str) -> np.ndarray:
        """Read QC label GeoTIFF into a [H, W] int64 array with values {-1, 0, 1}."""
        if HAS_RASTERIO:
            try:
                with rasterio.open(path) as src:
                    data = src.read(1)  # [H, W]
                    return data.astype(np.int64)
            except Exception:
                pass

        data = tifffile.imread(path)
        if data.ndim == 3:
            data = data[0] if data.shape[0] == 1 else data[:, :, 0]
        return data.astype(np.int64)

    @staticmethod
    def _atomic_save(path: Path, arr: np.ndarray) -> None:
        tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npy")
        np.save(tmp, arr)
        os.replace(tmp, path)

    def _cached_stack(self, idx: int) -> np.ndarray:
        """Raw channel-first [15, H, W] S1+S2 stack (float16, memory-mapped) from cache.

        GeoTIFFs are decoded and the cache entry written on a miss.
        """
        sample = self.samples[idx]
        path = self.cache_dir / f"{sample['chip_id']}.npy"
        if not path.exists():
            stack = np.concatenate(
                [self._read_geotiff(sample["s1_path"]), self._read_geotiff(sample["s2_path"])], axis=-1
            )
            self._atomic_save(path, np.ascontiguousarray(np.moveaxis(stack, -1, 0)).astype(np.float16))
        return np.load(path, mmap_mode="r")

    def _channel_indices(self) -> List[int]:
        s2 = [2 + i for i in self.s2_band_indices]
        return {"s1": [0, 1], "s2": s2, "fusion": [0, 1] + s2}[self.modality]

    def _read_cached_window(self, idx: int, window: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
        """Read only the needed channels (and crop window) from the cache -> [h, w, C] float32."""
        stack = self._cached_stack(idx)
        ch = self._channel_indices()
        if window is not None:
            y, x, h, w = window
            stack = stack[:, y : y + h, x : x + w]
        if ch == list(range(ch[0], ch[-1] + 1)):
            data = stack[ch[0] : ch[-1] + 1]
        else:
            data = stack[ch]
        return np.ascontiguousarray(np.moveaxis(np.asarray(data, dtype=np.float32), 0, -1))

    def get_label(self, idx: int) -> np.ndarray:
        """Label [H, W] int64 in {-1, 0, 1} (cached if ``cache_dir`` is set)."""
        sample = self.samples[idx]
        if self.cache_dir is None:
            return self._read_label(sample["label_path"])
        path = self.cache_dir / f"{sample['chip_id']}_label.npy"
        if path.exists():
            return np.load(path).astype(np.int64)
        label = self._read_label(sample["label_path"])
        self._atomic_save(path, label.astype(np.int8))
        return label

    def get_raw_image(self, idx: int) -> np.ndarray:
        """Load raw unnormalized image [H, W, C] for inspection or baselines."""
        sample = self.samples[idx]

        if self.cache_dir is not None:
            return self._read_cached_window(idx, None)

        if self.modality == "s1":
            # 2 bands: VV, VH (dB)
            return self._read_geotiff(sample["s1_path"])
        elif self.modality == "s2":
            # 13 bands: TOA reflectance (unscaled or scaled by 10,000)
            s2_data = self._read_geotiff(sample["s2_path"])
            if self.s2_band_indices:
                s2_data = s2_data[:, :, self.s2_band_indices]
            return s2_data
        elif self.modality == "fusion":
            s1_data = self._read_geotiff(sample["s1_path"])
            s2_data = self._read_geotiff(sample["s2_path"])
            if self.s2_band_indices:
                s2_data = s2_data[:, :, self.s2_band_indices]
            # Stack S1 and S2 along channel dimension: [H, W, C1 + C2]
            return np.concatenate([s1_data, s2_data], axis=-1)
        else:
            raise ValueError(f"Unknown modality: {self.modality}")

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        sample = self.samples[idx]

        # 1. Read image and mask. A random crop (training) is taken *before* reading
        #    from the memory-mapped cache, so only the window's bytes hit the disk.
        mask = self.get_label(idx)  # [H, W], values in {-1, 0, 1}
        window = None
        if self.crop_size is not None:
            ch, cw = self.crop_size
            H, W = mask.shape
            y = int(torch.randint(0, H - ch + 1, (1,)))
            x = int(torch.randint(0, W - cw + 1, (1,)))
            window = (y, x, ch, cw)
            mask = mask[y : y + ch, x : x + cw]
        if self.cache_dir is not None:
            raw_image = self._read_cached_window(idx, window)
        else:
            raw_image = self.get_raw_image(idx)
            if window is not None:
                y, x, ch, cw = window
                raw_image = raw_image[y : y + ch, x : x + cw]

        # 2. Apply normalization (if normalizer provided)
        if self.normalizer is not None:
            image = self.normalizer(raw_image)
        else:
            image = raw_image

        # 3. Apply geometric augmentations (flips, rotations, crops)
        # Note: Mask is kept intact, {-1, 0, 1} preserved
        image_tensor, mask_tensor = apply_transforms(image, mask, self.transform)

        # 4. Generate boolean valid pixel mask: True for valid {0, 1}, False for no-data {-1}
        valid_mask = mask_tensor >= 0

        # Ensure no NaNs or Infs exist in image tensor
        image_tensor = torch.nan_to_num(image_tensor, nan=0.0, posinf=0.0, neginf=0.0)

        return {
            "image": image_tensor,  # FloatTensor [C, H, W]
            "mask": mask_tensor,  # LongTensor [H, W] (-1: invalid, 0: not water, 1: water)
            "valid_mask": valid_mask,  # BoolTensor [H, W]
            "chip_id": sample["chip_id"],
            "event": sample["event"],
        }
