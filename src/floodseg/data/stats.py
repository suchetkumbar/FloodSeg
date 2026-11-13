"""Compute and persist train-split-only normalization statistics."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Union

from tqdm import tqdm

from floodseg.data.dataset import Sen1Floods11Dataset
from floodseg.data.normalizer import ChannelNormalizer, RunningChannelStats
from floodseg.data.splits import get_split_paths, get_stats_path

logger = logging.getLogger(__name__)


def compute_split_stats(
    data_dir: Union[str, Path], split_type: str, write: bool = True
) -> Dict[str, ChannelNormalizer]:
    """Compute per-band stats over the *train* split of ``split_type``.

    One streaming pass computes S1 and S2 statistics; fusion stats are their
    concatenation (identical to computing them on the 15-band stack, since
    statistics are per channel).
    """
    data_dir = Path(data_dir)
    train_csv = get_split_paths(data_dir, split_type)["train"]
    ds = Sen1Floods11Dataset(root_dir=data_dir, split_file=train_csv, modality="fusion")

    s1_stats = RunningChannelStats(2)
    s2_stats = RunningChannelStats(13)
    for i in tqdm(range(len(ds)), desc=f"Stats [{split_type} train]"):
        raw = ds.get_raw_image(i)
        s1_stats.update(raw[:, :, :2])
        s2_stats.update(raw[:, :, 2:])

    s1 = s1_stats.to_normalizer()
    s2 = s2_stats.to_normalizer()
    out = {"s1": s1, "s2": s2, "fusion": ChannelNormalizer.concat(s1, s2)}

    if write:
        for modality, norm in out.items():
            path = get_stats_path(data_dir, split_type, modality)
            norm.save_json(
                path,
                metadata={"split_type": split_type, "split": "train", "num_chips": len(ds)},
            )
            logger.info("Wrote %s/%s stats (%d train chips) -> %s", split_type, modality, len(ds), path)
    return out
