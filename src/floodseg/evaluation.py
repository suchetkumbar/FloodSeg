"""Model evaluation on a split: pooled + per-event metrics, bootstrap CIs, calibration.

A single inference pass collects per-chip confusion counts and a reliability
histogram, so downstream analysis (per-event, per-chip, ECE) never needs to
re-run the model.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from floodseg.data.dataset import Sen1Floods11Dataset
from floodseg.data.normalizer import ChannelNormalizer
from floodseg.metrics.evaluator import confusion_counts, iou_from_counts, summarize_with_ci


class CalibrationHistogram:
    """Accumulates (count, sum prob, sum label) per probability bin over valid pixels."""

    def __init__(self, n_bins: int = 15):
        self.n_bins = n_bins
        self.count = np.zeros(n_bins, dtype=np.float64)
        self.sum_prob = np.zeros(n_bins, dtype=np.float64)
        self.sum_label = np.zeros(n_bins, dtype=np.float64)

    def update(self, probs: torch.Tensor, target: torch.Tensor) -> None:
        valid = target != -1
        p = probs[valid].float()
        y = (target[valid] == 1).float()
        b = torch.clamp((p * self.n_bins).long(), max=self.n_bins - 1)
        self.count += torch.bincount(b, minlength=self.n_bins).double().cpu().numpy()
        self.sum_prob += torch.bincount(b, weights=p, minlength=self.n_bins).double().cpu().numpy()
        self.sum_label += torch.bincount(b, weights=y, minlength=self.n_bins).double().cpu().numpy()

    def ece(self) -> float:
        total = self.count.sum()
        if total == 0:
            return 0.0
        nz = self.count > 0
        gap = np.abs(self.sum_prob[nz] / self.count[nz] - self.sum_label[nz] / self.count[nz])
        return float((self.count[nz] / total * gap).sum())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_bins": self.n_bins,
            "count": self.count.tolist(),
            "mean_prob": np.divide(self.sum_prob, self.count, out=np.zeros_like(self.count), where=self.count > 0).tolist(),
            "frac_water": np.divide(self.sum_label, self.count, out=np.zeros_like(self.count), where=self.count > 0).tolist(),
        }


@torch.no_grad()
def evaluate_model_on_split(
    model: nn.Module,
    split_path: Path,
    data_dir: Path,
    modality: str,
    normalizer: Optional[ChannelNormalizer],
    device: torch.device,
    batch_size: int = 4,
    num_workers: int = 0,
    amp: bool = True,
    cache_dir: Optional[Path] = None,
    cloud_fn=None,
) -> Dict[str, Any]:
    """Evaluate on full 512x512 chips. Returns pooled metrics + CIs, per-event metrics,
    per-chip counts and a calibration histogram."""
    ds = Sen1Floods11Dataset(root_dir=data_dir, split_file=split_path, modality=modality, normalizer=normalizer, cache_dir=cache_dir,
        cloud_fn=cloud_fn,
    )
    if len(ds) == 0:
        return {}
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model.eval()
    use_amp = amp and device.type == "cuda"

    calib = CalibrationHistogram()
    per_chip: Dict[str, Tuple[int, int, int, int]] = {}
    chip_event: Dict[str, str] = {}
    for batch in loader:
        imgs = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(imgs)
        probs = torch.sigmoid(logits.float()).squeeze(1)
        preds = (probs > 0.5).long()
        calib.update(probs, masks)
        for p, m, cid, ev in zip(preds, masks, batch["chip_id"], batch["event"]):
            per_chip[cid] = confusion_counts(p, m)
            chip_event[cid] = ev

    ids = list(per_chip)
    counts = np.array([per_chip[c] for c in ids], dtype=np.int64)
    res = summarize_with_ci(counts)
    res["ece"] = calib.ece()
    res["calibration"] = calib.to_dict()

    by_event: Dict[str, list] = defaultdict(list)
    for cid in ids:
        by_event[chip_event[cid]].append(per_chip[cid])
    res["per_event"] = {
        ev: {k: v for k, v in summarize_with_ci(np.array(c), n_bootstraps=1000).items()}
        for ev, c in sorted(by_event.items())
    }
    res["per_chip"] = {
        cid: {"event": chip_event[cid], "tp": int(per_chip[cid][0]), "fp": int(per_chip[cid][1]),
              "fn": int(per_chip[cid][2]), "tn": int(per_chip[cid][3]),
              "iou": iou_from_counts(*per_chip[cid][:3])}
        for cid in ids
    }
    return res
