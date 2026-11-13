"""Unit tests for non-learned remote sensing baselines."""

import numpy as np
import pytest

from floodseg.metrics.evaluator import SegmentationMetrics, compute_sample_iou
from floodseg.models.baselines import (
    MNDWIBaseline,
    NDWIBaseline,
    OtsuSARBaseline,
    SARThresholdBaseline,
)


@pytest.fixture
def synthetic_optical_data():
    """Create synthetic S2 optical chip with clear water and land signatures."""
    h, w = 64, 64
    cube = np.zeros((h, w, 13), dtype=np.float32)
    label = np.zeros((h, w), dtype=np.int64)

    # Water region: Green high, NIR low
    cube[:32, :, 2] = 2000.0   # Band 3 (Green)
    cube[:32, :, 7] = 200.0    # Band 8 (NIR)
    cube[:32, :, 11] = 100.0   # Band 11 (SWIR1)
    label[:32, :] = 1          # Water

    # Land region: Green moderate, NIR very high (vegetation)
    cube[32:, :, 2] = 800.0
    cube[32:, :, 7] = 4000.0
    cube[32:, :, 11] = 2500.0
    label[32:, :] = 0          # Land

    # Invalid stripe
    label[0:5, :] = -1

    return cube, label


@pytest.fixture
def synthetic_sar_data():
    """Create synthetic S1 SAR chip with dark water and bright rough land."""
    h, w = 64, 64
    cube = np.zeros((h, w, 2), dtype=np.float32)
    label = np.zeros((h, w), dtype=np.int64)

    rng = np.random.default_rng(0)

    # Water region: Specular reflection -> very dark VV (around -20 dB)
    cube[:32, :, 0] = rng.normal(-20.0, 1.0, (32, w))
    cube[:32, :, 1] = rng.normal(-25.0, 1.0, (32, w))
    label[:32, :] = 1

    # Land region: Diffuse reflection -> brighter VV (around -8 dB)
    cube[32:, :, 0] = rng.normal(-8.0, 1.5, (32, w))
    cube[32:, :, 1] = rng.normal(-14.0, 1.5, (32, w))
    label[32:, :] = 0

    return cube, label


def test_ndwi_baseline_detection(synthetic_optical_data):
    """Test NDWI water detection on synthetic multispectral cube."""
    cube, label = synthetic_optical_data
    baseline = NDWIBaseline(threshold=0.0)

    pred = baseline.predict(cube)
    assert pred.shape == (64, 64)

    metrics = SegmentationMetrics(ignore_index=-1)
    metrics.update(pred, label)
    res = metrics.compute()

    assert res["iou"] > 0.85, f"Expected high water IoU, got {res['iou']}"
    assert res["precision"] > 0.85
    assert res["recall"] > 0.85


def test_mndwi_baseline_tuning(synthetic_optical_data):
    """Test MNDWI threshold grid search on synthetic training data."""
    cube, label = synthetic_optical_data
    baseline = MNDWIBaseline(threshold=0.0)

    tuned_thresh = baseline.tune_threshold([cube], [label])
    assert -0.5 <= tuned_thresh <= 0.5

    pred = baseline.predict(cube)
    iou = compute_sample_iou(pred, label, ignore_index=-1)
    assert iou > 0.85


def test_otsu_sar_baseline(synthetic_sar_data):
    """Test Otsu adaptive thresholding on bimodal SAR VV distribution."""
    cube, label = synthetic_sar_data
    baseline = OtsuSARBaseline()

    pred = baseline.predict_adaptive(cube)
    assert pred.shape == (64, 64)

    metrics = SegmentationMetrics(ignore_index=-1)
    metrics.update(pred, label)
    res = metrics.compute()

    assert res["iou"] > 0.80, f"Expected high SAR IoU, got {res['iou']}"


def test_sar_threshold_tuning_recovers_separating_value(synthetic_sar_data):
    """Tuned global VV threshold must fall between the water (-20 dB) and land (-8 dB) modes."""
    cube, label = synthetic_sar_data
    baseline = SARThresholdBaseline()
    tau = baseline.tune_threshold([cube], [label])
    # Any threshold in the gap between the brightest water and darkest land pixel is optimal
    gap_lo, gap_hi = cube[:32, :, 0].max(), cube[32:, :, 0].min()
    assert gap_lo - 0.5 < tau < gap_hi + 0.5
    assert compute_sample_iou(baseline.predict(cube), label) > 0.95


def test_threshold_counts_match_predict(synthetic_optical_data):
    """Streaming per-threshold counts must equal counts from an explicit predict()."""
    from floodseg.metrics.evaluator import confusion_counts

    cube, label = synthetic_optical_data
    rng = np.random.default_rng(0)
    cube = cube + rng.normal(0, 300, cube.shape).astype(np.float32)
    for baseline in (NDWIBaseline(), MNDWIBaseline()):
        grid = np.array([-0.3, 0.0, 0.2])
        counts = baseline.threshold_counts(cube, label, grid)
        for t, row in zip(grid, counts):
            baseline.threshold = float(t)
            tp, fp, fn, _ = confusion_counts(baseline.predict(cube), label)
            assert tuple(row) == (tp, fp, fn)


def test_sar_nan_pixels_are_never_water(synthetic_sar_data):
    cube, label = synthetic_sar_data
    cube = cube.copy()
    cube[:4, :, 0] = np.nan
    assert SARThresholdBaseline(-15.0).predict(cube)[:4].sum() == 0
    assert OtsuSARBaseline().predict(cube)[:4].sum() == 0
