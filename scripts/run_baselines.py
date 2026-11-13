"""Execution and benchmark script for Milestone 2: Non-learned Baselines.

For each split regime the thresholds are tuned on *that regime's* train split only:
* ``official``: tune on official train -> evaluate on official test.
* ``loeo``:     tune on LOEO train (no India / Bolivia) -> evaluate on held-out events.

Tuning the LOEO baselines on the official train split would leak India chips into
the "held-out" evaluation, so the two regimes are fully separated.

Metrics: pooled water IoU (headline), F1, precision, recall, chip-mean IoU, and
95% CIs from a chip-level bootstrap of the *same* aggregate.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np
from tqdm import tqdm

from floodseg.data.dataset import Sen1Floods11Dataset
from floodseg.data.splits import SPLIT_TYPES, get_split_paths
from floodseg.metrics.evaluator import confusion_counts, summarize_with_ci
from floodseg.models.baselines import (
    MNDWIBaseline,
    NDWIBaseline,
    OtsuSARBaseline,
    SARThresholdBaseline,
    ThresholdBaseline,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Baselines")

MODALITY = {
    "NDWI": "Optical (S2)",
    "MNDWI": "Optical (S2)",
    "SAR_VV_Threshold": "SAR (S1)",
    "Otsu_SAR_Adaptive": "SAR (S1)",
}


def iter_chips(data_dir: Path, split_csv: Path, desc: str) -> Iterator[Tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """Yield (chip_id, s1 [H,W,2], s2 [H,W,13], label [H,W]) one chip at a time."""
    ds = Sen1Floods11Dataset(root_dir=data_dir, split_file=split_csv, modality="fusion")
    for i in tqdm(range(len(ds)), desc=desc, leave=False):
        sample = ds.samples[i]
        raw = ds.get_raw_image(i)
        label = ds._read_label(sample["label_path"])
        yield sample["chip_id"], raw[:, :, :2], raw[:, :, 2:], label


def _input_for(model, s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    return s1 if MODALITY[model.name].startswith("SAR") else s2


def tune(models: List[ThresholdBaseline], data_dir: Path, train_csv: Path, tag: str) -> Dict[str, float]:
    grids = {m.name: m.default_grid for m in models}
    acc = {m.name: np.zeros((len(grids[m.name]), 3), dtype=np.int64) for m in models}
    for _, s1, s2, lbl in iter_chips(data_dir, train_csv, f"Tune [{tag}]"):
        for m in models:
            acc[m.name] += m.threshold_counts(_input_for(m, s1, s2), lbl, grids[m.name])
    return {m.name: m.set_threshold_from_counts(acc[m.name], grids[m.name]) for m in models}


def evaluate(models: list, data_dir: Path, test_csv: Path, tag: str) -> Dict[str, Dict]:
    counts: Dict[str, List[Tuple[int, int, int, int]]] = {m.name: [] for m in models}
    per_chip_iou: Dict[str, Dict[str, float]] = {m.name: {} for m in models}
    for chip_id, s1, s2, lbl in iter_chips(data_dir, test_csv, f"Eval [{tag}]"):
        for m in models:
            c = confusion_counts(m.predict(_input_for(m, s1, s2)), lbl)
            counts[m.name].append(c)
            tp, fp, fn, _ = c
            per_chip_iou[m.name][chip_id] = tp / (tp + fp + fn) if (tp + fp + fn) else 1.0

    results = {}
    for m in models:
        res = summarize_with_ci(np.array(counts[m.name]))
        res["per_chip_iou"] = per_chip_iou[m.name]
        results[m.name] = res
    return results


def main():
    data_dir = Path("data")
    results_dir = Path("outputs") / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results: Dict[str, Dict] = {"thresholds": {}, "benchmarks": {}}

    for split_type in SPLIT_TYPES:
        paths = get_split_paths(data_dir, split_type)
        eval_name = "official_test" if split_type == "official" else "loeo_held_out"
        logger.info("=== Regime %s: tune on %s, evaluate on %s ===", split_type, paths["train"].name, paths["test"].name)

        tuned = [NDWIBaseline(), MNDWIBaseline(), SARThresholdBaseline()]
        all_results["thresholds"][split_type] = tune(tuned, data_dir, paths["train"], split_type)
        all_results["benchmarks"][eval_name] = evaluate(
            tuned + [OtsuSARBaseline()], data_dir, paths["test"], eval_name
        )

    json_path = results_dir / "baselines_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Saved baseline results to %s", json_path)

    md = render_markdown(all_results)
    table_path = results_dir / "baselines_table.md"
    table_path.write_text(md, encoding="utf-8")
    logger.info("Wrote markdown results table to %s", table_path)
    print("\n" + md)


def render_markdown(all_results: Dict) -> str:
    th = all_results["thresholds"]
    lines = [
        "# Milestone 2: Baseline Benchmark Results (Non-Learned)",
        "",
        "Thresholds are tuned per split regime on that regime's **train split only** "
        "(maximising pooled water IoU). Otsu is scene-adaptive and untuned.",
        "",
        "| Regime | NDWI τ | MNDWI τ | SAR VV τ (dB) |",
        "| :--- | :---: | :---: | :---: |",
    ]
    for regime, t in th.items():
        lines.append(f"| `{regime}` | {t['NDWI']:.2f} | {t['MNDWI']:.2f} | {t['SAR_VV_Threshold']:.1f} |")
    lines += [
        "",
        "**Water IoU** is pooled over all valid pixels of the split; **Chip-mean IoU** averages "
        "per-chip IoU (chips with no water and no predicted water count as 1.0). "
        "95% CIs come from 2,000 chip-level bootstrap resamples of the same aggregate.",
        "",
        "| Baseline | Modality | Eval split | Water IoU (%) | 95% CI | Chip-mean IoU (%) | 95% CI | F1 (%) | Precision (%) | Recall (%) | Chips |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for split_name, models in all_results["benchmarks"].items():
        for name, m in models.items():
            lines.append(
                f"| **{name}** | {MODALITY[name]} | `{split_name}` | **{m['iou']*100:.2f}** | "
                f"[{m['ci_95_lower']*100:.1f}, {m['ci_95_upper']*100:.1f}] | "
                f"{m['chip_mean_iou']*100:.2f} | "
                f"[{m['chip_mean_ci_95_lower']*100:.1f}, {m['chip_mean_ci_95_upper']*100:.1f}] | "
                f"{m['f1']*100:.2f} | {m['precision']*100:.2f} | {m['recall']*100:.2f} | {m['num_chips']} |"
            )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
