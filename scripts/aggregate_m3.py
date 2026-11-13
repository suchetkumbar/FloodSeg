"""Aggregate Milestone 3 runs into a results table.

* Per configuration (split regime x modality x loss): mean +- std over seeds of the
  pooled test water IoU, F1, precision, recall, ECE, plus per-event IoU.
* Paired modality comparisons: for each regime, per-chip confusion counts are summed
  over seeds, and a paired chip-level bootstrap gives a 95% CI for the difference in
  pooled IoU (e.g. fusion - S2). Pairing on the same chips removes between-chip
  variance, so this is far tighter than comparing two independent CIs.

Outputs: outputs/results/m3_results.json, outputs/results/m3_table.md
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

RUNS = Path("outputs/runs")
RESULTS = Path("outputs/results")
MOD_LABEL = {"s1": "SAR (S1, 2 ch)", "s2": "Optical (S2, 13 ch)", "fusion": "Early fusion (15 ch)"}
COMPARISONS = [("fusion", "s2"), ("fusion", "s1"), ("s2", "s1")]


def load_runs() -> List[Dict]:
    runs = []
    for d in sorted(RUNS.glob("*")):
        s, c = d / "summary.json", d / "per_chip.json"
        if s.exists() and c.exists():
            run = json.loads(s.read_text())
            run["per_chip"] = json.loads(c.read_text())
            runs.append(run)
    return runs


def mean_std(xs: List[float]) -> Dict[str, float]:
    a = np.asarray(xs, dtype=np.float64)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0, "n": len(a)}


def fmt(ms: Dict[str, float], pct: bool = True) -> str:
    k = 100.0 if pct else 1.0
    if ms["n"] > 1:
        return f"{ms['mean'] * k:.1f} ± {ms['std'] * k:.1f}"
    return f"{ms['mean'] * k:.1f}"


def seed_summed_counts(runs: List[Dict], split: str) -> Dict[str, np.ndarray]:
    """chip_id -> (tp, fp, fn) summed across seeds."""
    acc: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros(3, dtype=np.int64))
    for r in runs:
        for cid, c in r["per_chip"][split].items():
            acc[cid] += np.array([c["tp"], c["fp"], c["fn"]])
    return dict(acc)


def paired_diff(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray], n_boot: int = 5000, seed: int = 0) -> Dict:
    ids = sorted(set(a) & set(b))
    A = np.stack([a[i] for i in ids])
    B = np.stack([b[i] for i in ids])

    def iou(m):
        tot = m.sum(axis=-2)
        return tot[..., 0] / np.maximum(tot.sum(axis=-1), 1)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ids), size=(n_boot, len(ids)))
    diffs = iou(A[idx]) - iou(B[idx])
    point = float(iou(A) - iou(B))
    return {
        "diff": point,
        "ci_95_lower": float(np.quantile(diffs, 0.025)),
        "ci_95_upper": float(np.quantile(diffs, 0.975)),
        "p_two_sided": float(min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))),
        "num_chips": len(ids),
    }


def main():
    runs = load_runs()
    if not runs:
        print("No completed runs found in outputs/runs", file=sys.stderr)
        return
    groups: Dict[tuple, List[Dict]] = defaultdict(list)
    for r in runs:
        # Non-standard variants (e.g. cloud-augmented M4 runs) are reported by
        # scripts/run_m4_analysis.py, not in the M3 tables.
        if r.get("variant", "standard") == "standard":
            groups[(r["split_type"], r["modality"], r["loss"])].append(r)

    configs = {}
    for (split, mod, loss), rs in sorted(groups.items()):
        entry = {
            "split_type": split, "modality": mod, "loss": loss,
            "seeds": sorted(r["seed"] for r in rs),
            "best_epoch": mean_std([r["best_epoch"] for r in rs]),
            "train_minutes": mean_std([r["train_minutes"] for r in rs]),
            "num_params": rs[0]["num_params"],
            "eval": {},
            "per_seed_test": {
                r["seed"]: {"iou": r["eval"]["test"]["iou"], "ece": r["eval"]["test"]["ece"]} for r in rs
            },
        }
        for ev_split in rs[0]["eval"]:
            e = {m: mean_std([r["eval"][ev_split][m] for r in rs]) for m in ("iou", "f1", "precision", "recall", "chip_mean_iou", "ece")}
            events = rs[0]["eval"][ev_split]["per_event"].keys()
            e["per_event_iou"] = {ev: mean_std([r["eval"][ev_split]["per_event"][ev]["iou"] for r in rs]) for ev in events}
            entry["eval"][ev_split] = e
        configs[f"{split}_{mod}_{loss}"] = entry

    comparisons = {}
    for split in ("official", "loeo"):
        for a, b in COMPARISONS:
            ra = groups.get((split, a, "bce_dice"), [])
            rb = groups.get((split, b, "bce_dice"), [])
            seeds = sorted({r["seed"] for r in ra} & {r["seed"] for r in rb})
            if not seeds:
                continue
            ca = seed_summed_counts([r for r in ra if r["seed"] in seeds], "test")
            cb = seed_summed_counts([r for r in rb if r["seed"] in seeds], "test")
            comparisons[f"{split}: {a} - {b}"] = {**paired_diff(ca, cb), "seeds": seeds}

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = {"configs": configs, "paired_comparisons": comparisons}
    (RESULTS / "m3_results.json").write_text(json.dumps(out, indent=2))
    md = render(configs, comparisons)
    (RESULTS / "m3_table.md").write_text(md, encoding="utf-8")
    sys.stdout.reconfigure(errors="replace")
    print(md)


def render(configs: Dict, comparisons: Dict) -> str:
    L = [
        "# Milestone 3: U-Net (ResNet-34) Results",
        "",
        "Pooled water IoU (%) on the test split of each regime, mean ± sample std over seeds "
        "(42, 1337, 2024). Model selection uses the val split only; each seed's best checkpoint "
        "is evaluated once. ECE uses 15 equal-width probability bins over valid pixels.",
        "",
        "## Main results (BCE + Dice)",
        "",
        "| Modality | Official test IoU | Bolivia IoU (official model) | Held-out events IoU (LOEO) | LOEO: India | LOEO: Bolivia | Official F1 | Official ECE | Seeds |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for mod in ("s1", "s2", "fusion"):
        o = configs.get(f"official_{mod}_bce_dice")
        lo = configs.get(f"loeo_{mod}_bce_dice")
        if not o and not lo:
            continue
        dash = "–"
        L.append(
            "| " + " | ".join([
                f"**{MOD_LABEL[mod]}**",
                fmt(o["eval"]["test"]["iou"]) if o else dash,
                fmt(o["eval"]["bolivia"]["iou"]) if o and "bolivia" in o["eval"] else dash,
                fmt(lo["eval"]["test"]["iou"]) if lo else dash,
                fmt(lo["eval"]["test"]["per_event_iou"]["India"]) if lo else dash,
                fmt(lo["eval"]["test"]["per_event_iou"]["Bolivia"]) if lo else dash,
                fmt(o["eval"]["test"]["f1"]) if o else dash,
                fmt(o["eval"]["test"]["ece"]) if o else dash,
                f"{len(o['seeds']) if o else 0} / {len(lo['seeds']) if lo else 0}",
            ]) + " |"
        )
    L += [
        "",
        "## Paired modality comparisons (test split, seed-summed per-chip counts)",
        "",
        "| Comparison | Δ pooled IoU (pp) | 95% CI (paired chip bootstrap) | p (two-sided) | Chips |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    for name, c in comparisons.items():
        L.append(
            f"| {name} | {c['diff'] * 100:+.1f} | [{c['ci_95_lower'] * 100:+.1f}, {c['ci_95_upper'] * 100:+.1f}] | "
            f"{'<0.001' if c['p_two_sided'] < 0.001 else format(c['p_two_sided'], '.3f')} | {c['num_chips']} |"
        )
    L += [
        "",
        "## Loss ablation (official regime, seed 42)",
        "",
        "| Modality | BCE + Dice IoU (%) | Dice-only IoU (%) | BCE + Dice ECE (%) | Dice-only ECE (%) |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    for mod in ("s1", "s2", "fusion"):
        d = configs.get(f"official_{mod}_dice_only")
        b = configs.get(f"official_{mod}_bce_dice")
        if not d or not b:
            continue
        b42 = b["per_seed_test"].get(42)
        if b42 is None:
            continue
        L.append(
            f"| {MOD_LABEL[mod]} | {b42['iou'] * 100:.1f} | {d['eval']['test']['iou']['mean'] * 100:.1f} | "
            f"{b42['ece'] * 100:.1f} | {d['eval']['test']['ece']['mean'] * 100:.1f} |"
        )
    L += [
        "",
        "## Training cost",
        "",
        "| Config | Params | Best epoch | Train minutes |",
        "| :--- | :---: | :---: | :---: |",
    ]
    for name, c in configs.items():
        L.append(
            f"| `{name}` | {c['num_params'] / 1e6:.2f}M | {c['best_epoch']['mean']:.0f} | {c['train_minutes']['mean']:.1f} |"
        )
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
