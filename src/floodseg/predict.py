"""Scene-level flood prediction to georeferenced GeoTIFFs.

Input rasters are expected in Sen1Floods11 conventions:
* S1: 2 bands (VV, VH) backscatter in **dB**;
* S2: 13 bands (B1 ... B12) TOA reflectance x 10,000.
If both are given, S2 is resampled onto the S1 grid. Several checkpoints (e.g. the
three seeds of one configuration) are averaged into an ensemble probability.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import rasterio
import torch
from rasterio.warp import Resampling, reproject

from floodseg.inference import load_model_from_checkpoint, predict_tiled

logger = logging.getLogger(__name__)


def read_on_grid(path: Path, ref_profile: Optional[Dict] = None) -> tuple:
    """Read a raster as [H, W, C] float32 (nodata -> NaN), optionally reprojected onto
    ``ref_profile``'s grid (bilinear). Returns (array, profile)."""
    with rasterio.open(path) as src:
        if ref_profile is None:
            data = src.read(out_dtype="float32", masked=True).filled(np.nan)
            return np.moveaxis(data, 0, -1), src.profile.copy()
        out = np.full((src.count, ref_profile["height"], ref_profile["width"]), np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src, list(range(1, src.count + 1))),
            destination=out,
            src_nodata=src.nodata,
            dst_nodata=np.nan,
            dst_transform=ref_profile["transform"],
            dst_crs=ref_profile["crs"],
            resampling=Resampling.bilinear,
        )
        return np.moveaxis(out, 0, -1), ref_profile


def write_geotiff(path: Path, array: np.ndarray, profile: Dict, dtype: str, nodata) -> None:
    prof = profile.copy()
    prof.update(driver="GTiff", count=1, dtype=dtype, nodata=nodata, compress="deflate", tiled=True,
                blockxsize=256, blockysize=256)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(array.astype(dtype), 1)


def predict_scene(
    checkpoints: Sequence[Path],
    s1_path: Optional[Path],
    s2_path: Optional[Path],
    out_prefix: Path,
    threshold: float = 0.5,
    tile: int = 512,
    overlap: int = 128,
    device: Optional[torch.device] = None,
) -> Dict[str, Path]:
    """Predict a scene and write ``<prefix>_prob.tif`` (float32 0-1) and ``<prefix>_water.tif``
    (uint8: 1 water, 0 land, 255 no-data)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = [load_model_from_checkpoint(p, device) for p in checkpoints]
    modalities = {meta["modality"] for _, _, meta in loaded}
    if len(modalities) != 1:
        raise ValueError(f"All checkpoints must share one modality, got {modalities}")
    modality = modalities.pop()

    parts: List[np.ndarray] = []
    profile = None
    if modality in ("s1", "fusion"):
        if s1_path is None:
            raise ValueError(f"Modality {modality!r} needs --s1")
        s1, profile = read_on_grid(Path(s1_path))
        if s1.shape[-1] != 2:
            raise ValueError(f"S1 raster must have 2 bands (VV, VH dB), got {s1.shape[-1]}")
        parts.append(s1)
    if modality in ("s2", "fusion"):
        if s2_path is None:
            raise ValueError(f"Modality {modality!r} needs --s2")
        s2, profile = read_on_grid(Path(s2_path), profile)
        if s2.shape[-1] != 13:
            raise ValueError(f"S2 raster must have 13 bands (B1..B12), got {s2.shape[-1]}")
        parts.append(s2)
    image = np.concatenate(parts, axis=-1)
    invalid = ~np.all(np.isfinite(image), axis=-1)

    logger.info("Predicting %s scene %s with %d model(s) on %s", modality, image.shape[:2], len(loaded), device)
    prob = predict_tiled(
        [m for m, _, _ in loaded], image, [n for _, n, _ in loaded], tile=tile, overlap=overlap, device=device
    )
    prob[invalid] = np.nan
    water = np.where(invalid, 255, (prob > threshold).astype(np.uint8))

    out_prefix = Path(out_prefix)
    paths = {"prob": out_prefix.with_name(out_prefix.name + "_prob.tif"),
             "water": out_prefix.with_name(out_prefix.name + "_water.tif")}
    write_geotiff(paths["prob"], prob, profile, "float32", np.nan)
    write_geotiff(paths["water"], water, profile, "uint8", 255)
    logger.info("Wrote %s and %s", paths["prob"], paths["water"])
    return paths
