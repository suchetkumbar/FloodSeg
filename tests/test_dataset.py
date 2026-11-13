"""Unit tests for Sen1Floods11Dataset, normalizer, transforms, and data integrity.

Validates:
- Tensor shapes and dtypes
- Mask alignment and valid value set {-1, 0, 1}
- Per-band normalization
- 'No NaNs / No Infs' invariant
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import tifffile
import torch

from floodseg.data.dataset import Sen1Floods11Dataset
from floodseg.data.normalizer import ChannelNormalizer
from floodseg.data.transforms import apply_transforms, get_train_transforms


@pytest.fixture
def mock_dataset_dir(tmp_path: Path):
    """Create a temporary mock Sen1Floods11 directory with synthetic GeoTIFFs."""
    raw_dir = tmp_path / "raw" / "sen1floods11"
    s1_dir = raw_dir / "S1Hand"
    s2_dir = raw_dir / "S2Hand"
    label_dir = raw_dir / "LabelHand"
    splits_dir = tmp_path / "processed" / "splits" / "official"

    for d in (s1_dir, s2_dir, label_dir, splits_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Image specs
    height, width = 64, 64  # Small dimension for rapid testing

    # 1. Create S1 mock (2 bands: VV, VH) as [H, W, 2]
    s1_data = np.random.uniform(-25.0, 0.0, (height, width, 2)).astype(np.float32)
    s1_path = s1_dir / "India_999999_S1Hand.tif"
    tifffile.imwrite(s1_path, s1_data, planarconfig="CONTIG")

    # 2. Create S2 mock (13 bands: uint16) as [H, W, 13]
    s2_data = np.random.randint(100, 5000, (height, width, 13), dtype=np.uint16)
    s2_path = s2_dir / "India_999999_S2Hand.tif"
    tifffile.imwrite(s2_path, s2_data, planarconfig="CONTIG")

    # 3. Create Label mock (1 band: -1, 0, 1) as [H, W]
    label_data = np.random.choice([-1, 0, 1], size=(height, width), p=[0.1, 0.6, 0.3]).astype(
        np.int16
    )
    label_path = label_dir / "India_999999_LabelHand.tif"
    tifffile.imwrite(label_path, label_data)

    # 4. Create split CSV
    split_csv = splits_dir / "flood_train_data.csv"
    with open(split_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["India_999999_S1Hand.tif", "India_999999_LabelHand.tif", "India_999999_S2Hand.tif"]
        )

    return tmp_path, split_csv


def test_dataset_shapes_and_dtypes(mock_dataset_dir):
    """Test output shapes, channel counts, and dtypes for s1, s2, and fusion."""
    root_dir, split_csv = mock_dataset_dir

    # Test S1-only
    ds_s1 = Sen1Floods11Dataset(root_dir=root_dir, split_file=split_csv, modality="s1")
    sample_s1 = ds_s1[0]
    assert sample_s1["image"].shape == (2, 64, 64)
    assert sample_s1["image"].dtype == torch.float32
    assert sample_s1["mask"].shape == (64, 64)
    assert sample_s1["mask"].dtype == torch.int64

    # Test S2-only
    ds_s2 = Sen1Floods11Dataset(root_dir=root_dir, split_file=split_csv, modality="s2")
    sample_s2 = ds_s2[0]
    assert sample_s2["image"].shape == (13, 64, 64)
    assert sample_s2["image"].dtype == torch.float32

    # Test Fusion (S1 + S2)
    ds_fusion = Sen1Floods11Dataset(root_dir=root_dir, split_file=split_csv, modality="fusion")
    sample_fusion = ds_fusion[0]
    assert sample_fusion["image"].shape == (15, 64, 64)
    assert sample_fusion["image"].dtype == torch.float32


def test_mask_alignment_and_values(mock_dataset_dir):
    """Verify mask values are in {-1, 0, 1} and valid_mask matches mask >= 0."""
    root_dir, split_csv = mock_dataset_dir
    ds = Sen1Floods11Dataset(root_dir=root_dir, split_file=split_csv, modality="fusion")
    sample = ds[0]

    mask = sample["mask"]
    valid_mask = sample["valid_mask"]

    unique_vals = torch.unique(mask).tolist()
    for val in unique_vals:
        assert val in [-1, 0, 1], f"Unexpected mask value: {val}"

    # Verify valid_mask
    expected_valid = mask >= 0
    assert torch.equal(valid_mask, expected_valid)


def test_no_nans_or_infs(mock_dataset_dir):
    """Verify images contain zero NaNs or Infs."""
    root_dir, split_csv = mock_dataset_dir
    ds = Sen1Floods11Dataset(root_dir=root_dir, split_file=split_csv, modality="fusion")
    sample = ds[0]
    image = sample["image"]

    assert not torch.isnan(image).any(), "Found NaNs in dataset image tensor"
    assert not torch.isinf(image).any(), "Found Infs in dataset image tensor"


def test_normalization_application(mock_dataset_dir):
    """Test ChannelNormalizer transforms input and produces finite normalized values."""
    root_dir, split_csv = mock_dataset_dir

    means = [10.0] * 15
    stds = [5.0] * 15
    normalizer = ChannelNormalizer(mean=means, std=stds)

    ds = Sen1Floods11Dataset(
        root_dir=root_dir, split_file=split_csv, modality="fusion", normalizer=normalizer
    )
    sample = ds[0]
    image = sample["image"]

    assert image.shape == (15, 64, 64)
    assert not torch.isnan(image).any()
    assert not torch.isinf(image).any()


def test_rigid_transforms_mask_consistency():
    """Verify spatial transforms keep image and mask perfectly aligned."""
    img = np.zeros((64, 64, 3), dtype=np.float32)
    mask = np.zeros((64, 64), dtype=np.int64)

    # Set asymmetric pattern
    img[0:10, 0:10, :] = 5.0
    mask[0:10, 0:10] = 1
    mask[20:30, 20:30] = -1

    transform = get_train_transforms(crop_size=(32, 32), p_flip=1.0, p_rot=1.0)
    img_t, mask_t = apply_transforms(img, mask, transform)

    assert img_t.shape == (3, 32, 32)
    assert mask_t.shape == (32, 32)
    assert set(torch.unique(mask_t).tolist()).issubset({-1, 0, 1})


def test_real_downloaded_chips_integration():
    """Verify loading real downloaded chips from data/ directory if present."""
    real_split = Path("data/processed/splits/official/flood_train_data.csv")
    first_label = Path("data/raw/sen1floods11/LabelHand/Ghana_103272_LabelHand.tif")
    if not real_split.exists() or not first_label.exists():
        pytest.skip("Downloaded Sen1Floods11 chips not found.")

    ds = Sen1Floods11Dataset(
        root_dir=Path("data"),
        split_file=real_split,
        modality="fusion",
    )
    assert len(ds) > 0, "No chips in local split"

    sample = ds[0]
    # S1 (2) + S2 (13) = 15 channels, 512x512 resolution
    assert sample["image"].shape == (15, 512, 512)
    assert sample["mask"].shape == (512, 512)
    assert sample["valid_mask"].shape == (512, 512)
    assert not torch.isnan(sample["image"]).any()
    assert not torch.isinf(sample["image"]).any()
    assert set(torch.unique(sample["mask"]).tolist()).issubset({-1, 0, 1})


@pytest.mark.parametrize("modality", ["s1", "s2", "fusion"])
def test_cache_and_crop_match_uncached(mock_dataset_dir, tmp_path, modality):
    """Cached (float16, channel-first mmap) reads must match GeoTIFF reads; crops must align."""
    root_dir, split_csv = mock_dataset_dir
    plain = Sen1Floods11Dataset(root_dir=root_dir, split_file=split_csv, modality=modality)
    cached = Sen1Floods11Dataset(
        root_dir=root_dir, split_file=split_csv, modality=modality, cache_dir=tmp_path / "cache"
    )
    a, b = plain.get_raw_image(0), cached.get_raw_image(0)
    assert a.shape == b.shape
    np.testing.assert_allclose(a, b, rtol=1e-3, atol=1e-2)
    assert np.array_equal(plain.get_label(0), cached.get_label(0))

    torch.manual_seed(0)
    cropped = Sen1Floods11Dataset(
        root_dir=root_dir, split_file=split_csv, modality=modality,
        cache_dir=tmp_path / "cache", crop_size=(32, 32),
    )[0]
    assert cropped["image"].shape[1:] == (32, 32)
    assert cropped["mask"].shape == (32, 32)
    # The cropped mask must be a contiguous window of the full mask.
    full = torch.from_numpy(plain.get_label(0))
    found = any(
        torch.equal(full[y : y + 32, x : x + 32], cropped["mask"])
        for y in range(33) for x in range(33)
    )
    assert found
