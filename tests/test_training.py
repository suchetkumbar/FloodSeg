"""Unit tests for the LR schedule, calibration (ECE) and split evaluation."""

import csv
from pathlib import Path

import numpy as np
import pytest
import tifffile
import torch

from floodseg.evaluation import CalibrationHistogram, evaluate_model_on_split
from floodseg.models.train_engine import warmup_cosine_scheduler


def test_warmup_then_cosine_schedule():
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    sched = warmup_cosine_scheduler(opt, epochs=10, warmup_epochs=3, min_lr=0.01)
    lrs = []
    for _ in range(10):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] < lrs[1] < lrs[2] < lrs[3] == pytest.approx(1.0)
    assert all(a >= b for a, b in zip(lrs[3:], lrs[4:]))
    assert lrs[-1] >= 0.01


def test_ece_zero_for_calibrated_and_large_for_overconfident():
    rng = torch.Generator().manual_seed(0)
    probs = torch.rand(200_000, generator=rng)
    labels = (torch.rand(200_000, generator=rng) < probs).long()
    h = CalibrationHistogram()
    h.update(probs, labels)
    assert h.ece() < 0.01

    h2 = CalibrationHistogram()
    h2.update(torch.full((1000,), 0.99), torch.zeros(1000, dtype=torch.long))
    assert h2.ece() > 0.9


def test_ece_ignores_no_data():
    h = CalibrationHistogram()
    h.update(torch.tensor([0.99, 0.99]), torch.tensor([1, -1]))
    assert h.count.sum() == 1


class _OracleFromChannel(torch.nn.Module):
    """Predicts water where input channel 0 > 0 (logit = 10 * x0)."""

    def forward(self, x):
        return 10.0 * x[:, :1]


def test_evaluate_model_on_split_per_event(tmp_path: Path):
    raw = tmp_path / "raw" / "sen1floods11"
    for d in ("S1Hand", "S2Hand", "LabelHand"):
        (raw / d).mkdir(parents=True)
    rows = []
    for name in ("India_1", "India_2", "Bolivia_1"):
        s1 = np.random.uniform(-1, 1, (32, 32, 2)).astype(np.float32)
        label = (s1[:, :, 0] > 0).astype(np.int16)
        label[:2] = -1
        tifffile.imwrite(raw / "S1Hand" / f"{name}_S1Hand.tif", s1, planarconfig="CONTIG")
        tifffile.imwrite(raw / "S2Hand" / f"{name}_S2Hand.tif", np.ones((32, 32, 13), np.uint16), planarconfig="CONTIG")
        tifffile.imwrite(raw / "LabelHand" / f"{name}_LabelHand.tif", label)
        rows.append([f"{name}_S1Hand.tif", f"{name}_LabelHand.tif", f"{name}_S2Hand.tif"])
    split = tmp_path / "split.csv"
    with open(split, "w", newline="") as f:
        csv.writer(f).writerows(rows)

    res = evaluate_model_on_split(
        _OracleFromChannel(), split, tmp_path, "s1", None, torch.device("cpu"), batch_size=2
    )
    assert res["num_chips"] == 3
    assert res["iou"] == pytest.approx(1.0)
    assert set(res["per_event"]) == {"India", "Bolivia"}
    assert res["per_event"]["India"]["num_chips"] == 2
    assert res["per_chip"]["Bolivia_1"]["iou"] == pytest.approx(1.0)


def test_predict_tiled_matches_single_tile_and_handles_odd_sizes():
    from floodseg.data.normalizer import ChannelNormalizer
    from floodseg.inference import predict_tiled

    class Px(torch.nn.Module):  # per-pixel model: tiling must not change its output
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.ones(1))

        def forward(self, x):
            return x[:, :1] * self.w

    norm = ChannelNormalizer([0.0, 0.0], [1.0, 1.0])
    img = np.random.default_rng(0).normal(0, 1, (700, 900, 2)).astype(np.float32)
    prob = predict_tiled(Px(), img, norm, tile=256, overlap=64, device=torch.device("cpu"))
    assert prob.shape == (700, 900)
    np.testing.assert_allclose(prob, 1 / (1 + np.exp(-img[..., 0])), atol=1e-4)
    small = predict_tiled(Px(), img[:100, :120], norm, tile=256, overlap=64, device=torch.device("cpu"))
    assert small.shape == (100, 120)
