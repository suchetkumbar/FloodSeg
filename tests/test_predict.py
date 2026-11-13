"""End-to-end test of scene prediction to georeferenced GeoTIFFs."""

from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.transform import from_origin

from floodseg.data.normalizer import ChannelNormalizer
from floodseg.models.unet import FloodUNet
from floodseg.predict import predict_scene


def _write_s1(path: Path, h: int = 300, w: int = 420) -> rasterio.Affine:
    transform = from_origin(76.3, 9.6, 8.983e-5, 8.983e-5)
    data = np.random.default_rng(0).uniform(-25, 0, (2, h, w)).astype(np.float32)
    data[:, :5, :7] = np.nan  # no-data corner
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=2, dtype="float32",
                       crs="EPSG:4326", transform=transform, nodata=np.nan) as dst:
        dst.write(data)
    return transform


def _checkpoint(path: Path) -> None:
    model = FloodUNet(in_channels=2, encoder_weights=None)
    torch.save({
        "model_state_dict": model.state_dict(), "modality": "s1", "in_channels": 2,
        "encoder_name": "resnet34", "rgb_indices": None,
        "normalizer": ChannelNormalizer([-10.0, -17.0], [4.0, 4.7]).to_dict(),
    }, path)


def test_predict_scene_writes_aligned_geotiffs(tmp_path: Path):
    s1 = tmp_path / "s1.tif"
    transform = _write_s1(s1)
    ckpt = tmp_path / "m.pt"
    _checkpoint(ckpt)

    paths = predict_scene([ckpt, ckpt], s1, None, tmp_path / "out" / "scene", tile=256, overlap=64,
                          device=torch.device("cpu"))
    with rasterio.open(paths["prob"]) as p, rasterio.open(paths["water"]) as w:
        assert p.shape == w.shape == (300, 420)
        assert p.crs.to_epsg() == 4326 and p.transform == transform
        prob, water = p.read(1), w.read(1)
    assert np.isnan(prob[:5, :7]).all() and (water[:5, :7] == 255).all()
    finite = np.isfinite(prob)
    assert ((prob[finite] >= 0) & (prob[finite] <= 1)).all()
    assert set(np.unique(water[finite])).issubset({0, 1})
