"""Milestone 3 experiment grid: run every training config sequentially, then aggregate.

Grid (21 runs):
* modality in {s1, s2, fusion} x split regime in {official, loeo} x seed in {42, 1337, 2024}
  with the BCE+Dice loss (18 runs);
* loss ablation: Dice-only, official regime, seed 42, all three modalities (3 runs).

Runs whose ``summary.json`` already exists are skipped, so the grid is resumable.
Seed 42 runs go first, so a partial grid already covers every configuration.

    python scripts/run_m3_grid.py            # run all + aggregate
    python scripts/run_m3_grid.py --dry-run  # list runs
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

MODALITIES = ("s1", "s2", "fusion")
SPLITS = ("official", "loeo")
SEEDS = (42, 1337, 2024)


def grid():
    for seed in SEEDS:
        for split in SPLITS:
            for mod in MODALITIES:
                yield {"dataset.split_type": split, "dataset.modality": mod, "seed": seed, "train.loss": "bce_dice"}
        if seed == SEEDS[0]:
            for mod in MODALITIES:
                yield {"dataset.split_type": "official", "dataset.modality": mod, "seed": seed, "train.loss": "dice_only"}


def run_name(o) -> str:
    return f"{o['dataset.split_type']}_{o['dataset.modality']}_{o['train.loss']}_seed{o['seed']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--wandb-mode", default="disabled", help="offline/online to also log to W&B")
    args = ap.parse_args()

    runs = list(grid())
    for i, o in enumerate(runs, 1):
        name = run_name(o)
        done = (Path("outputs/runs") / name / "summary.json").exists()
        print(f"[{i:02d}/{len(runs)}] {name}{'  (done, skipping)' if done else ''}", flush=True)
        if done or args.dry_run:
            continue
        t0 = time.time()
        cmd = [sys.executable, "scripts/train_model.py", f"wandb.mode={args.wandb_mode}"]
        cmd += [f"{k}={v}" for k, v in o.items()]
        rc = subprocess.call(cmd)
        print(f"    -> exit {rc} in {(time.time() - t0) / 60:.1f} min", flush=True)
        if rc != 0:
            print(f"    !! run {name} failed; continuing with the rest of the grid", flush=True)

    if not args.dry_run:
        subprocess.call([sys.executable, "scripts/aggregate_m3.py"])


if __name__ == "__main__":
    main()
