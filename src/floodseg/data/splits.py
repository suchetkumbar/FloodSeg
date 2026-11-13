"""Single source of truth for split CSV locations and normalization-stat paths."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Literal, Union

SplitType = Literal["official", "loeo"]
SPLIT_TYPES = ("official", "loeo")
MODALITY_CHANNELS = {"s1": 2, "s2": 13, "fusion": 15}


def get_split_paths(data_dir: Union[str, Path], split_type: str) -> Dict[str, Path]:
    """Return {'train', 'val', 'test'} split CSV paths for a split regime.

    For ``loeo`` the ``test`` split is the held-out events (India, Bolivia), which are
    never seen during training, validation, threshold tuning or normalization.
    """
    root = Path(data_dir) / "processed" / "splits"
    if split_type == "official":
        d = root / "official"
        return {
            "train": d / "flood_train_data.csv",
            "val": d / "flood_val_data.csv",
            "test": d / "flood_test_data.csv",
        }
    if split_type == "loeo":
        d = root / "loeo"
        return {
            "train": d / "loeo_train_data.csv",
            "val": d / "loeo_val_data.csv",
            "test": d / "loeo_held_out_test_data.csv",
        }
    raise ValueError(f"Unknown split_type {split_type!r}; expected one of {SPLIT_TYPES}")


def get_stats_path(data_dir: Union[str, Path], split_type: str, modality: str) -> Path:
    """Path of the train-only normalization stats for a (split regime, modality) pair."""
    return Path(data_dir) / "stats" / f"train_stats_{split_type}_{modality}.json"
