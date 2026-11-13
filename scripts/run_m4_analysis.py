"""Milestone 4: where does each modality fail, and how much does SAR help under cloud?

Analyses (official-regime models, 3 seeds each; inference only):
A. Synthetic cloud occlusion: S2 bands of the official test chips are covered with
   synthetic thick cloud at 0/25/50/75/100% while the true labels are kept. Compared:
     * SAR (S1)             - unaffected by construction (radar penetrates cloud)
     * Optical / Fusion     - standard models
     * ... + mask fill      - cloud pixels (opacity >= 0.5) replaced by the train mean,
                              i.e. a perfect cloud mask is available at inference
     * ... cloud-aug        - trained with synthetic-cloud augmentation (if trained)
   Paired chip-bootstrap CIs on seed-summed counts for the key differences.
B. Real chips stratified by a cloud proxy (bright/cirrus fraction) and by the
   labellers' no-data fraction, official test + LOEO held-out events.
C. Per-event IoU on the official test split.
D. Reliability diagrams (seed-summed calibration histograms).
E. Qualitative figures (modality disagreements; an occlusion example).
F. Inference cost per 512 x 512 chip.

Outputs: outputs/results/m4_results.json, m4_table.md, outputs/figures/m4_*.png
"""

from __future__ import annotations

import json
import sys
import time
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from floodseg.data.clouds import apply_synthetic_clouds, cloud_proxy_mask, synthetic_cloud_alpha
from floodseg.data.dataset import Sen1Floods11Dataset
from floodseg.data.splits import get_split_paths
from floodseg.inference import load_model_from_checkpoint
from floodseg.metrics.evaluator import confusion_counts, iou_from_counts

RUNS = Path("outputs/runs")
RESULTS = Path("outputs/results")
FIGS = Path("outputs/figures")
DATA = Path("data")
CACHE = DATA / "cache" / "sen1floods11"
SEEDS = (42, 1337, 2024)
COVERAGES = (0.0, 0.25, 0.5, 0.75, 1.0)
CH = {"s1": slice(0, 2), "s2": slice(2, 15), "fusion": slice(0, 15)}

# Reference categorical palette, slots 1-3 (validated all-pairs, light mode).
COLOR = {"s1": "#2a78d6", "s2": "#eb6834", "fusion": "#1baf7a"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
LABEL = {"s1": "SAR (S1)", "s2": "Optical (S2)", "fusion": "Fusion (S1+S2)"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})


# ----------------------------------------------------------------------------- helpers
def pooled_iou(counts: np.ndarray) -> float:
    t = counts.sum(axis=0)
    return float(t[0] / max(t[0] + t[1] + t[2], 1))


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 5000, seed: int = 0) -> Dict[str, float]:
    """a, b: [N, 4] per-chip counts on the same chips. CI of pooled IoU(a) - IoU(b)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))

    def iou(m):
        t = m.sum(axis=1)
        return t[:, 0] / np.maximum(t[:, 0] + t[:, 1] + t[:, 2], 1)

    d = iou(a[idx]) - iou(b[idx])
    return {
        "diff": pooled_iou(a) - pooled_iou(b),
        "ci_95_lower": float(np.quantile(d, 0.025)),
        "ci_95_upper": float(np.quantile(d, 0.975)),
        "p_two_sided": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))),
    }


def ckpt(mod: str, seed: int, variant: str = "standard", split: str = "official") -> Optional[Path]:
    suffix = "" if variant == "standard" else f"_{variant}"
    p = RUNS / f"{split}_{mod}_bce_dice{suffix}_seed{seed}" / "best_model.pt"
    return p if p.exists() else None


def fmt_ms(vals: List[float]) -> str:
    a = np.asarray(vals) * 100
    return f"{a.mean():.1f} ± {a.std(ddof=1):.1f}" if len(a) > 1 else f"{a.mean():.1f}"


def fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def gpu_counts(pred: torch.Tensor, label: torch.Tensor) -> tuple:
    """(tp, fp, fn, tn) with a single device->host sync."""
    valid = label != -1
    t = label == 1
    tp = (pred & t).sum()
    fp = (pred & ~t & valid).sum()
    fn = (~pred & t).sum()
    tn = valid.sum() - tp - fp - fn
    return tuple(int(v) for v in torch.stack([tp, fp, fn, tn]).cpu())


# ----------------------------------------------------------------- A. cloud occlusion
def occlusion_experiment(device) -> Dict:
    variants = {  # name -> (modality, training variant, fill-with-mean?)
        "s1": ("s1", "standard", False),
        "s2": ("s2", "standard", False),
        "fusion": ("fusion", "standard", False),
        "s2_fill": ("s2", "standard", True),
        "fusion_fill": ("fusion", "standard", True),
        "s2_cloudaug": ("s2", "cloudaug", False),
        "fusion_cloudaug": ("fusion", "cloudaug", False),
    }
    models = {}
    for name, (mod, var, _) in variants.items():
        for seed in SEEDS:
            p = ckpt(mod, seed, var)
            if p is not None and (var, mod, seed) not in models:
                models[(var, mod, seed)] = load_model_from_checkpoint(p, device)[:2]
    gpu_stats = {
        k: (torch.as_tensor(n.mean, device=device)[:, None, None], torch.as_tensor(n.std + n.eps, device=device)[:, None, None])
        for k, (_, n) in models.items()
    }
    active = {n: v for n, v in variants.items() if any((v[1], v[0], s) in models for s in SEEDS)}
    print(f"[A] occlusion: {len(models)} models, variants {list(active)}", flush=True)

    ds = Sen1Floods11Dataset(DATA, get_split_paths(DATA, "official")["test"], "fusion", cache_dir=CACHE)
    counts = {n: {s: {c: [] for c in COVERAGES} for s in SEEDS} for n in active}
    use_amp = device.type == "cuda"
    t0 = time.time()
    for i in range(len(ds)):
        cid = ds.samples[i]["chip_id"]
        raw = ds.get_raw_image(i)
        label = torch.from_numpy(ds.get_label(i)).to(device)
        for cov in COVERAGES:
            rng = np.random.default_rng(zlib.crc32(f"{cid}|{cov:.3f}".encode()))
            alpha = synthetic_cloud_alpha(raw.shape[:2], cov, rng)
            img = torch.from_numpy(np.ascontiguousarray(
                apply_synthetic_clouds(raw, alpha, 2, rng).transpose(2, 0, 1))).to(device)  # [15, H, W]
            cloudy = torch.from_numpy(alpha >= 0.5).to(device)
            for name, (mod, var, fill) in active.items():
                if mod == "s1" and cov > 0:  # SAR is untouched by optical cloud
                    for s in SEEDS:
                        if counts[name][s][0.0]:
                            counts[name][s][cov].append(counts[name][s][0.0][-1])
                    continue
                for s in SEEDS:
                    if (var, mod, s) not in models:
                        continue
                    model, norm = models[(var, mod, s)]
                    mean, std = gpu_stats[(var, mod, s)]
                    x = img[CH[mod]]
                    if fill and cov > 0:
                        s2_off = 2 if mod == "fusion" else 0
                        x = x.clone()
                        x[s2_off:] = torch.where(cloudy, mean[s2_off:], x[s2_off:])
                    xn = torch.nan_to_num((x - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)[None]
                    with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
                        logit = model(xn)[0, 0]
                    counts[name][s][cov].append(gpu_counts(logit > 0, label))
        if (i + 1) % 15 == 0:
            print(f"    {i + 1}/{len(ds)} chips ({time.time() - t0:.0f}s)", flush=True)

    out: Dict = {"coverages": list(COVERAGES), "variants": {}, "comparisons": {}}
    arr = {n: {s: {c: np.array(v, dtype=np.int64) for c, v in d.items()} for s, d in sd.items() if d[0.0]}
           for n, sd in counts.items()}
    for n, sd in arr.items():
        out["variants"][n] = {
            "seeds": sorted(sd),
            "iou": {str(c): [pooled_iou(sd[s][c]) for s in sorted(sd)] for c in COVERAGES},
        }

    def summed(n, c):
        return sum(arr[n][s][c] for s in arr[n])

    pairs = [("fusion", "s2"), ("fusion", "s1"), ("s2", "s1"), ("fusion_fill", "s2_fill"),
             ("fusion_cloudaug", "s2_cloudaug"), ("fusion_cloudaug", "s1"), ("fusion_cloudaug", "fusion")]
    for a, b in pairs:
        if a in arr and b in arr:
            common = sorted(set(arr[a]) & set(arr[b]))
            if not common:
                continue
            out["comparisons"][f"{a} - {b}"] = {
                str(c): paired_bootstrap(sum(arr[a][s][c] for s in common), sum(arr[b][s][c] for s in common))
                for c in COVERAGES
            }
    return out


def plot_occlusion(res: Dict) -> None:
    style = {  # variant -> (modality colour, linestyle, marker, label)
        "s1": ("s1", "-", "o", "SAR (S1)"),
        "s2": ("s2", "-", "o", "Optical"),
        "fusion": ("fusion", "-", "o", "Fusion"),
        "s2_fill": ("s2", ":", "s", "Optical + mask fill"),
        "fusion_fill": ("fusion", ":", "s", "Fusion + mask fill"),
        "s2_cloudaug": ("s2", "--", "^", "Optical, cloud-aug"),
        "fusion_cloudaug": ("fusion", "--", "^", "Fusion, cloud-aug"),
    }
    cov = np.array(res["coverages"]) * 100
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ends = []
    for name, (mod, ls, mk, lab) in style.items():
        v = res["variants"].get(name)
        if not v:
            continue
        m = np.array([np.mean(v["iou"][str(c)]) for c in res["coverages"]]) * 100
        s = np.array([np.std(v["iou"][str(c)], ddof=1) if len(v["iou"][str(c)]) > 1 else 0 for c in res["coverages"]]) * 100
        ax.fill_between(cov, m - s, m + s, color=COLOR[mod], alpha=0.10, linewidth=0)
        ax.plot(cov, m, ls=ls, marker=mk, ms=8, lw=2, color=COLOR[mod], label=lab,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        ends.append([m[-1], lab])
    # Direct end labels, nudged apart to avoid collisions.
    ends.sort()
    placed: List[float] = []
    for y, lab in ends:
        y_adj = max([y] + [p + 3.2 for p in placed if abs(p - y) < 3.2])
        placed.append(y_adj)
        ax.annotate(lab, (100, y), xytext=(104, y_adj), textcoords="data", va="center", fontsize=8.5,
                    color=INK2, annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=GRID, lw=0.8) if abs(y_adj - y) > 0.5 else None)
    ax.set_xlim(-3, 103)
    ax.set_ylim(0, 100)
    ax.set_xticks(cov)
    ax.set_xticklabels([f"{int(c)}%" for c in cov])
    ax.set_xlabel("Synthetic thick-cloud coverage of the optical bands")
    ax.set_ylabel("Pooled water IoU (%), official test")
    ax.set_title("SAR is the only input unaffected by cloud; fusion only uses it if trained with clouds",
                 fontsize=10.5, loc="left", color=INK)
    ax.legend(loc="lower left", frameon=False, fontsize=8.5, ncol=2)
    fig.text(0.01, 0.01, "Mean over 3 seeds; band = ±1 s.d. Labels are the true water extent under the synthetic cloud.",
             fontsize=8, color=INK2)
    fig.tight_layout(rect=[0, 0.03, 0.86, 1])
    fig.savefig(FIGS / "m4_cloud_robustness.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------ B. real-chip stratification
def stratify_real_chips() -> Dict:
    bins = [(0.0, 0.05, "<5%"), (0.05, 0.30, "5–30%"), (0.30, 1.01, "≥30%")]
    out = {}
    for split in ("official", "loeo"):
        ds = Sen1Floods11Dataset(DATA, get_split_paths(DATA, split)["test"], "s2", cache_dir=CACHE)
        cloud_frac, nodata_frac = {}, {}
        for i in range(len(ds)):
            cid = ds.samples[i]["chip_id"]
            cloud_frac[cid] = float(cloud_proxy_mask(ds.get_raw_image(i)).mean())
            nodata_frac[cid] = float((ds.get_label(i) == -1).mean())
        per_mod = {}
        for mod in ("s1", "s2", "fusion"):
            summed: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros(4, dtype=np.int64))
            for seed in SEEDS:
                f = RUNS / f"{split}_{mod}_bce_dice_seed{seed}" / "per_chip.json"
                if not f.exists():
                    continue
                for cid, c in json.loads(f.read_text())["test"].items():
                    summed[cid] += np.array([c["tp"], c["fp"], c["fn"], c["tn"]])
            per_mod[mod] = dict(summed)
        strat = {}
        for key, frac in (("cloud_proxy", cloud_frac), ("labelled_nodata", nodata_frac)):
            rows = []
            for lo, hi, name in bins:
                ids = [c for c, v in frac.items() if lo <= v < hi]
                row = {"bin": name, "num_chips": len(ids)}
                for mod, d in per_mod.items():
                    have = [c for c in ids if c in d]
                    row[mod] = pooled_iou(np.array([d[c] for c in have])) if have else None
                rows.append(row)
            strat[key] = rows
        out[split] = strat
    return out


# --------------------------------------------------------- C/D. per-event, calibration
def per_event_and_calibration() -> Dict:
    per_event: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    calib = {}
    for mod in ("s1", "s2", "fusion"):
        cnt = mp = fw = None
        for seed in SEEDS:
            f = RUNS / f"official_{mod}_bce_dice_seed{seed}" / "summary.json"
            if not f.exists():
                continue
            t = json.loads(f.read_text())["eval"]["test"]
            for ev, r in t["per_event"].items():
                per_event[ev][mod].append(r["iou"])
            c = np.array(t["calibration"]["count"])
            cnt = c if cnt is None else cnt + c
            mp = c * np.array(t["calibration"]["mean_prob"]) + (0 if mp is None else mp)
            fw = c * np.array(t["calibration"]["frac_water"]) + (0 if fw is None else fw)
        if cnt is not None:
            calib[mod] = {"count": cnt.tolist(),
                          "mean_prob": np.divide(mp, cnt, out=np.zeros_like(mp), where=cnt > 0).tolist(),
                          "frac_water": np.divide(fw, cnt, out=np.zeros_like(fw), where=cnt > 0).tolist()}
    return {"per_event": {ev: dict(d) for ev, d in sorted(per_event.items())}, "calibration": calib}


def plot_reliability(calib: Dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True)
    for ax, mod in zip(axes, ("s1", "s2", "fusion")):
        c = calib.get(mod)
        ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls=":")
        if c:
            cnt = np.array(c["count"])
            nz = cnt > 0
            ax.plot(np.array(c["mean_prob"])[nz], np.array(c["frac_water"])[nz], marker="o", ms=8, lw=2,
                    color=COLOR[mod], markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.set_title(LABEL[mod], fontsize=10, loc="left")
        ax.set_xlabel("Predicted water probability")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Observed water fraction")
    fig.suptitle("Reliability on official test (3 seeds pooled; dotted = perfect calibration)",
                 fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIGS / "m4_reliability.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- E. qualitative
def _rgb(s2_raw: np.ndarray) -> np.ndarray:
    rgb = s2_raw[:, :, [3, 2, 1]].astype(np.float32)
    rgb = np.nan_to_num(rgb)
    lo, hi = np.percentile(rgb, (2, 98))
    return np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)


def _label_rgb(mask: np.ndarray, pred: Optional[np.ndarray] = None) -> np.ndarray:
    """Label view: water dark blue, land light, no-data grey. With pred: TP/FP/FN colouring."""
    out = np.full(mask.shape + (3,), 0.93, dtype=np.float32)
    out[mask == -1] = (0.75, 0.75, 0.75)
    if pred is None:
        out[mask == 1] = (0.10, 0.30, 0.65)
        return out
    out[(pred == 1) & (mask == 1)] = (0.10, 0.30, 0.65)   # hit
    out[(pred == 1) & (mask == 0)] = (0.92, 0.41, 0.20)   # false alarm
    out[(pred == 0) & (mask == 1)] = (0.38, 0.31, 0.84)   # miss
    return out


def qualitative_figures(device) -> List[str]:
    ds = Sen1Floods11Dataset(DATA, get_split_paths(DATA, "official")["test"], "fusion", cache_dir=CACHE)
    models = {}
    for mod in ("s1", "s2", "fusion"):
        p = ckpt(mod, 42)
        if p:
            models[mod] = load_model_from_checkpoint(p, device)[:2]
    ca = ckpt("fusion", 42, "cloudaug")
    if ca:
        models["fusion_cloudaug"] = load_model_from_checkpoint(ca, device)[:2]

    def predict(key, img):
        model, norm = models[key]
        mod = "fusion" if key.startswith("fusion") else key
        xn = np.nan_to_num(norm(img[:, :, CH[mod]]), nan=0.0, posinf=0.0, neginf=0.0)
        xt = torch.from_numpy(np.ascontiguousarray(xn.transpose(2, 0, 1)))[None].to(device)
        with torch.no_grad():
            return (model(xt)[0, 0] > 0).long().cpu().numpy()

    per = {m: json.loads((RUNS / f"official_{m}_bce_dice_seed42" / "per_chip.json").read_text())["test"]
           for m in ("s1", "s2", "fusion")}

    def area(cid):  # only pick chips with meaningful water
        c = per["s2"][cid]
        return c["tp"] + c["fn"]

    ids = [c for c in per["s2"] if area(c) > 2000]
    picks = []
    for key in (lambda c: per["s2"][c]["iou"] - per["s1"][c]["iou"],      # optical wins
                lambda c: per["s1"][c]["iou"] - per["s2"][c]["iou"],      # SAR wins
                lambda c: per["fusion"][c]["iou"] - per["s2"][c]["iou"],  # fusion wins
                lambda c: -max(per[m][c]["iou"] for m in per)):           # everyone fails
        for c in sorted(ids, key=key, reverse=True):
            if c not in picks:
                picks.append(c)
                break
    titles = ["Optical ≫ SAR", "SAR ≫ optical", "Fusion ≫ optical", "All fail"]
    index = {s["chip_id"]: i for i, s in enumerate(ds.samples)}

    cols = ["Optical RGB", "SAR VV (dB)", "Label", "SAR (S1)", "Optical (S2)", "Fusion"]
    fig, axes = plt.subplots(len(picks), len(cols), figsize=(2.1 * len(cols), 2.25 * len(picks)))
    for r, cid in enumerate(picks):
        i = index[cid]
        raw, lbl = ds.get_raw_image(i), ds.get_label(i)
        panels = [_rgb(raw[:, :, 2:]), np.clip((raw[:, :, 0] + 25) / 25, 0, 1), _label_rgb(lbl)]
        ious = []
        for m in ("s1", "s2", "fusion"):
            pred = predict(m, raw)
            panels.append(_label_rgb(lbl, pred))
            ious.append(iou_from_counts(*confusion_counts(pred, lbl)[:3]))
        for c, (ax, im) in enumerate(zip(axes[r], panels)):
            ax.imshow(im, cmap="gray" if c == 1 else None, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            if r == 0:
                ax.set_title(cols[c], fontsize=9.5)
            if c >= 3:
                ax.set_xlabel(f"IoU {ious[c - 3] * 100:.0f}%", fontsize=8.5, color=INK2)
        axes[r, 0].set_ylabel(f"{titles[r]}\n{cid}", fontsize=8.5, color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ((0.10, 0.30, 0.65), (0.92, 0.41, 0.20), (0.38, 0.31, 0.84), (0.75, 0.75, 0.75))]
    fig.legend(handles, ["Water (hit)", "False alarm", "Missed water", "No-data"], loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(FIGS / "m4_qualitative.png", dpi=130)
    plt.close(fig)

    # Occlusion example: a water-rich chip under 75% synthetic cloud.
    cid = max(ids, key=area)
    i = index[cid]
    raw, lbl = ds.get_raw_image(i), ds.get_label(i)
    rng = np.random.default_rng(zlib.crc32(f"{cid}|{0.75:.3f}".encode()))
    img = apply_synthetic_clouds(raw, synthetic_cloud_alpha(raw.shape[:2], 0.75, rng), 2, rng)
    keys = [k for k in ("s1", "s2", "fusion", "fusion_cloudaug") if k in models]
    names = {"s1": "SAR (S1)", "s2": "Optical", "fusion": "Fusion", "fusion_cloudaug": "Fusion, cloud-aug"}
    fig, axes = plt.subplots(1, 2 + len(keys), figsize=(2.3 * (2 + len(keys)), 2.8))
    axes[0].imshow(_rgb(img[:, :, 2:]))
    axes[0].set_title("Optical RGB, 75% cloud", fontsize=9.5)
    axes[1].imshow(_label_rgb(lbl))
    axes[1].set_title("True label", fontsize=9.5)
    for ax, k in zip(axes[2:], keys):
        pred = predict(k, img)
        ax.imshow(_label_rgb(lbl, pred))
        ax.set_title(names[k], fontsize=9.5)
        ax.set_xlabel(f"IoU {iou_from_counts(*confusion_counts(pred, lbl)[:3]) * 100:.0f}%", fontsize=8.5, color=INK2)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.suptitle(f"Synthetic cloud occlusion example ({cid})", fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIGS / "m4_occlusion_example.png", dpi=130)
    plt.close(fig)
    return picks


# ----------------------------------------------------------------- F. inference cost
def inference_cost(device) -> Dict:
    out = {}
    for mod in ("s1", "s2", "fusion"):
        p = ckpt(mod, 42)
        if not p:
            continue
        model, _, _ = load_model_from_checkpoint(p, device)
        x = torch.randn(1, model.in_channels, 512, 512, device=device)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            for _ in range(10):
                model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(50):
                model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
        out[mod] = {"ms_per_512px_chip": (time.perf_counter() - t) / 50 * 1000,
                    "params_millions": sum(q.numel() for q in model.parameters()) / 1e6,
                    "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"}
    return out


# -------------------------------------------------------------------------- report
def render(res: Dict) -> str:
    L = ["# Milestone 4: Failure Analysis & Cloud Robustness", ""]
    occ = res["occlusion"]
    names = {"s1": "SAR (S1)", "s2": "Optical (S2)", "fusion": "Fusion", "s2_fill": "Optical + cloud-mask fill",
             "fusion_fill": "Fusion + cloud-mask fill", "s2_cloudaug": "Optical, cloud-aug trained",
             "fusion_cloudaug": "Fusion, cloud-aug trained"}
    L += ["## A. Synthetic cloud occlusion (official test, 90 chips)", "",
          "S2 bands are covered by synthetic thick cloud (median train-set cloud spectrum, soft edges, "
          "±20% texture); labels keep the true water extent. *Mask fill* = cloud pixels (opacity ≥ 0.5) "
          "replaced by the train mean, i.e. a perfect cloud mask at inference. *Cloud-aug* = trained "
          "with the same synthetic clouds (p = 0.5, coverage ~ U(0, 1)) using different random fields.",
          "", "Pooled water IoU (%), mean ± s.d. over seeds:", "",
          "| Model | " + " | ".join(f"{int(c * 100)}%" for c in occ["coverages"]) + " |",
          "| :--- | " + " | ".join(":---:" for _ in occ["coverages"]) + " |"]
    for n, v in occ["variants"].items():
        L.append(f"| {names[n]} | " + " | ".join(fmt_ms(v["iou"][str(c)]) for c in occ["coverages"]) + " |")
    L += ["", "Paired differences (seed-summed per-chip counts, 5,000 chip bootstrap resamples), in pp:", "",
          "| Comparison | " + " | ".join(f"{int(c * 100)}%" for c in occ["coverages"]) + " |",
          "| :--- | " + " | ".join(":---:" for _ in occ["coverages"]) + " |"]
    for k, d in occ["comparisons"].items():
        a, b = k.split(" - ")
        cells = [f"{d[str(c)]['diff'] * 100:+.1f} [{d[str(c)]['ci_95_lower'] * 100:+.1f}, {d[str(c)]['ci_95_upper'] * 100:+.1f}]"
                 for c in occ["coverages"]]
        L.append(f"| {names[a]} − {names[b]} | " + " | ".join(cells) + " |")
    L += ["", "![Cloud robustness](../figures/m4_cloud_robustness.png)", "",
          "![Occlusion example](../figures/m4_occlusion_example.png)", ""]

    L += ["## B. Real chips stratified by cloud", "",
          "*Cloud proxy* = fraction of pixels with B2, B3, B4 > 0.18 or cirrus B10 > 0.012 (heuristic; "
          "it agrees only moderately with the labellers' no-data mask). *Labelled no-data* = fraction "
          "of pixels labelled −1 (mostly cloud and cloud shadow). Clouded pixels carry no label, so this "
          "measures performance on the *visible* part of cloudy scenes. Pooled IoU (%) of seed-summed counts.", ""]
    for split, strat in res["stratified"].items():
        for key, rows in strat.items():
            L += [f"**{'Official test' if split == 'official' else 'LOEO held-out'} — {key.replace('_', ' ')}**", "",
                  "| Bin | Chips | SAR (S1) | Optical (S2) | Fusion |", "| :--- | :---: | :---: | :---: | :---: |"]
            for r in rows:
                cell = lambda m: "–" if r.get(m) is None else f"{r[m] * 100:.1f}"  # noqa: E731
                L.append(f"| {r['bin']} | {r['num_chips']} | {cell('s1')} | {cell('s2')} | {cell('fusion')} |")
            L.append("")

    L += ["## C. Per-event IoU (official test, mean ± s.d. over seeds)", "",
          "| Event | SAR (S1) | Optical (S2) | Fusion |", "| :--- | :---: | :---: | :---: |"]
    for ev, d in res["per_event"].items():
        L.append(f"| {ev} | " + " | ".join(fmt_ms(d[m]) if m in d else "–" for m in ("s1", "s2", "fusion")) + " |")
    L += ["", "## D. Calibration", "", "![Reliability](../figures/m4_reliability.png)", "",
          "## E. Qualitative failure cases (seed 42)", "",
          f"Chips: {', '.join(res['qualitative_chips'])}. Blue = hit, orange = false alarm, violet = missed water.", "",
          "![Qualitative](../figures/m4_qualitative.png)", "",
          "## F. Inference cost (batch 1, AMP)", "", "| Model | Params | ms / 512×512 chip | Device |",
          "| :--- | :---: | :---: | :--- |"]
    for m, c in res["inference_cost"].items():
        L.append(f"| {LABEL[m]} | {c['params_millions']:.2f}M | {c['ms_per_512px_chip']:.1f} | {c['device']} |")
    return "\n".join(L) + "\n"


def main(skip_occlusion: bool = False):
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res_path = RESULTS / "m4_results.json"
    prev = json.loads(res_path.read_text()) if res_path.exists() else {}

    res = {}
    res["occlusion"] = prev["occlusion"] if skip_occlusion and "occlusion" in prev else occlusion_experiment(device)
    res_path.write_text(json.dumps({**prev, "occlusion": res["occlusion"]}, indent=2))
    plot_occlusion(res["occlusion"])
    print("[B] stratifying real chips", flush=True)
    res["stratified"] = stratify_real_chips()
    print("[C/D] per-event & calibration", flush=True)
    res.update(per_event_and_calibration())
    plot_reliability(res["calibration"])
    print("[E] qualitative figures", flush=True)
    res["qualitative_chips"] = qualitative_figures(device)
    print("[F] inference cost", flush=True)
    res["inference_cost"] = inference_cost(device)

    res_path.write_text(json.dumps(res, indent=2))
    md = render(res)
    (RESULTS / "m4_table.md").write_text(md, encoding="utf-8")
    sys.stdout.reconfigure(errors="replace")
    print(md)


if __name__ == "__main__":
    main(skip_occlusion="--skip-occlusion" in sys.argv)
