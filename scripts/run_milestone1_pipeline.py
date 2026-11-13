"""Verification and execution script for Milestone 1: Data Pipeline.

1. Downloads split CSVs and all hand-labeled chips (existing files are skipped).
2. Generates checksum manifest and LOEO splits.
3. Computes per-band normalization statistics on the *train* split of each
   split regime (official, LOEO) - never on val/test/held-out chips.
4. Generates exploratory visualization plot.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from floodseg.data.dataset import Sen1Floods11Dataset
from floodseg.data.download import download_sen1floods11
from floodseg.data.splits import SPLIT_TYPES, get_split_paths
from floodseg.data.stats import compute_split_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Milestone1")


def main():
    data_dir = Path("data")

    logger.info("=== Step 1: Downloading Splits & Hand-Labeled Chips ===")
    res = download_sen1floods11(data_dir=data_dir)
    logger.info("Download completed: %s", res)

    logger.info("=== Step 2: Verifying Split Files ===")
    for split_type in SPLIT_TYPES:
        for name, path in get_split_paths(data_dir, split_type).items():
            assert path.exists(), f"Missing {split_type}/{name} split: {path}"

    logger.info("=== Step 3: Computing Per-Band Normalization Stats (Train Only) ===")
    for split_type in SPLIT_TYPES:
        compute_split_stats(data_dir, split_type)

    logger.info("=== Step 4: Generating Visualization for 10 Chips ===")
    ds_fusion = Sen1Floods11Dataset(
        root_dir=data_dir, split_file=get_split_paths(data_dir, "official")["train"], modality="fusion"
    )
    num_samples = min(10, len(ds_fusion))

    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 3.8 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, 0)

    col_titles = [
        "Sentinel-2 Optical (RGB)",
        "Sentinel-1 SAR VV (dB)",
        "Sentinel-1 SAR VH (dB)",
        "Ground Truth Label",
    ]
    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=13, fontweight="bold", pad=10)

    cmap_label = mcolors.ListedColormap(["#d4af37", "#2b2b2b", "#00e5ff"])
    norm_label = mcolors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap_label.N)

    for i in range(num_samples):
        sample = ds_fusion.samples[i]
        raw = ds_fusion.get_raw_image(i)
        lbl = ds_fusion._read_label(sample["label_path"])

        # Optical RGB
        rgb = raw[:, :, [3, 2, 1]].copy()
        rgb = np.nan_to_num(rgb, nan=0.0)
        p2, p98 = np.percentile(rgb, (2, 98))
        rgb_vis = np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0.0, 1.0)

        # SAR VV / VH
        vv_vis = np.clip((raw[:, :, 0] - (-25.0)) / 25.0, 0.0, 1.0)
        vh_vis = np.clip((raw[:, :, 1] - (-25.0)) / 25.0, 0.0, 1.0)

        axes[i, 0].imshow(rgb_vis)
        axes[i, 0].set_ylabel(sample["chip_id"], fontsize=10, fontweight="bold")
        axes[i, 1].imshow(vv_vis, cmap="gray")
        axes[i, 2].imshow(vh_vis, cmap="gray")
        axes[i, 3].imshow(lbl, cmap=cmap_label, norm=norm_label)

        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])

    legend_patches = [
        plt.Rectangle((0, 0), 1, 1, color="#00e5ff", label="Water (1)"),
        plt.Rectangle((0, 0), 1, 1, color="#2b2b2b", label="Not Water (0)"),
        plt.Rectangle((0, 0), 1, 1, color="#d4af37", label="Invalid / No-data (-1)"),
    ]
    fig.legend(
        handles=legend_patches, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.005), fontsize=12
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    out_img = Path("notebooks") / "sample_chips_visualization.png"
    out_img.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_img, dpi=130, bbox_inches="tight")
    logger.info("Saved visualization figure to %s", out_img)

    logger.info("=== Milestone 1 Pipeline Execution Successful! ===")


if __name__ == "__main__":
    main()
