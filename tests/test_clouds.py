"""Tests for the cloud proxy and synthetic cloud occlusion."""

import numpy as np
import pytest

from floodseg.data.clouds import (
    THICK_CLOUD_TOA,
    FixedCoverageClouds,
    RandomCloudAugmentation,
    apply_synthetic_clouds,
    cloud_proxy_mask,
    synthetic_cloud_alpha,
)


@pytest.mark.parametrize("coverage", [0.1, 0.5, 0.9])
def test_alpha_hits_requested_coverage(coverage):
    alpha = synthetic_cloud_alpha((256, 256), coverage, np.random.default_rng(0))
    assert abs((alpha >= 0.5).mean() - coverage) < 0.02
    assert alpha.min() >= 0 and alpha.max() <= 1


def test_alpha_extremes():
    rng = np.random.default_rng(0)
    assert not synthetic_cloud_alpha((64, 64), 0.0, rng).any()
    assert synthetic_cloud_alpha((64, 64), 1.0, rng).min() == 1.0


def test_clouds_only_touch_optical_channels_of_fusion():
    img = np.random.default_rng(1).uniform(100, 3000, (64, 64, 15)).astype(np.float32)
    img[..., :2] = -12.0  # SAR dB
    out = FixedCoverageClouds(1.0)(img, "fusion", "chip_1")
    assert np.array_equal(out[..., :2], img[..., :2]), "SAR must be untouched"
    np.testing.assert_allclose(out[..., 2:].mean(axis=(0, 1)) / THICK_CLOUD_TOA, 1.0, atol=0.2)


def test_fixed_clouds_are_deterministic_and_sar_only_passthrough():
    img = np.ones((64, 64, 13), np.float32) * 500
    f = FixedCoverageClouds(0.5, seed=3)
    assert np.array_equal(f(img, "s2", "a"), f(img, "s2", "a"))
    assert not np.array_equal(f(img, "s2", "a"), f(img, "s2", "b"))
    s1 = np.ones((64, 64, 2), np.float32)
    assert FixedCoverageClouds(1.0)(s1, "s1", "a") is s1
    assert RandomCloudAugmentation(p=1.0)(s1, "s1", "a") is s1


def test_proxy_flags_synthetic_cloud_but_not_dark_water():
    water = np.full((32, 32, 13), 300.0, np.float32)
    water[..., 10] = 20.0  # cirrus band: clear sky is ~0.002 reflectance
    clouded = apply_synthetic_clouds(water, np.ones((32, 32), np.float32), 0, np.random.default_rng(0))
    assert not cloud_proxy_mask(water).any()
    assert cloud_proxy_mask(clouded).mean() > 0.95
