"""Unit tests for confusion accumulation, pooled IoU, and bootstrap CIs."""

import numpy as np
import torch

from floodseg.data.normalizer import RunningChannelStats
from floodseg.metrics.evaluator import SegmentationMetrics, confusion_counts, summarize_with_ci


def test_update_accepts_channel_dim_logit_shape():
    """Trainer passes preds as [B, 1, H, W] and masks as [B, H, W]."""
    pred = torch.zeros(4, 1, 8, 8, dtype=torch.long)
    pred[:, :, :4] = 1
    target = torch.zeros(4, 8, 8, dtype=torch.long)
    target[:, :4] = 1
    target[:, 7] = -1
    m = SegmentationMetrics()
    m.update(pred, target)
    res = m.compute()
    assert res["iou"] == 1.0
    assert res["total_pixels"] == 4 * 7 * 8


def test_torch_and_numpy_counts_agree():
    rng = np.random.default_rng(0)
    pred = rng.integers(0, 2, (3, 16, 16))
    target = rng.integers(-1, 2, (3, 16, 16))
    assert confusion_counts(pred, target) == confusion_counts(torch.from_numpy(pred), torch.from_numpy(target))


def test_pooled_ci_brackets_point_estimate():
    rng = np.random.default_rng(1)
    counts = np.stack(
        [rng.integers(0, 5000, 60), rng.integers(0, 2000, 60), rng.integers(0, 2000, 60), rng.integers(0, 9e4, 60)],
        axis=1,
    )
    res = summarize_with_ci(counts)
    assert res["ci_95_lower"] <= res["iou"] <= res["ci_95_upper"]
    assert res["chip_mean_ci_95_lower"] <= res["chip_mean_iou"] <= res["chip_mean_ci_95_upper"]


def test_per_chip_tracking_matches_pooled():
    rng = np.random.default_rng(2)
    pred = rng.integers(0, 2, (5, 8, 8))
    target = rng.integers(-1, 2, (5, 8, 8))
    m = SegmentationMetrics(track_per_chip=True)
    m.update(pred, target)
    per_chip = m.per_chip_counts()
    assert per_chip.shape == (5, 4)
    assert per_chip.sum(axis=0).tolist() == [m.tp, m.fp, m.fn, m.tn]


def test_running_stats_match_numpy_and_skip_nans_per_channel():
    rng = np.random.default_rng(3)
    imgs = [rng.normal(5, 2, (32, 32, 3)) for _ in range(4)]
    imgs[1][0, 0, 0] = np.nan  # must only drop that value from channel 0
    running = RunningChannelStats(3)
    for im in imgs:
        running.update(im)
    stacked = np.concatenate([im.reshape(-1, 3) for im in imgs])
    norm = running.to_normalizer()
    np.testing.assert_allclose(norm.mean, np.nanmean(stacked, axis=0), rtol=1e-5)
    np.testing.assert_allclose(norm.std, np.nanstd(stacked, axis=0), rtol=1e-5)
