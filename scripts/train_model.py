"""Milestone 3: train a U-Net (ResNet-34) on Sen1Floods11 and evaluate it.

Hydra app. Examples:
    python scripts/train_model.py                                   # fusion, official, seed 42
    python scripts/train_model.py dataset.modality=s1 seed=1337
    python scripts/train_model.py dataset.split_type=loeo train.loss=dice_only
    python scripts/train_model.py +experiment=s2_only

Leakage rules enforced here:
* Normalization stats come from the train split of the *active* regime.
* Model selection (early stopping / best checkpoint) uses the val split only.
* The test split (official test, or LOEO held-out India + Bolivia) is evaluated
  exactly once, with the best checkpoint, after training finishes.

Outputs (in ``output_dir``): best_model.pt, history.json, summary.json, .hydra/
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from floodseg.data.dataset import Sen1Floods11Dataset
from floodseg.data.normalizer import ChannelNormalizer
from floodseg.data.splits import get_split_paths, get_stats_path
from floodseg.data.transforms import get_train_transforms
from floodseg.evaluation import evaluate_model_on_split
from floodseg.losses.combined import CombinedBCEDiceLoss
from floodseg.models.train_engine import Trainer, warmup_cosine_scheduler
from floodseg.models.unet import FloodUNet, rgb_indices_for_modality
from floodseg.utils.logging import setup_wandb
from floodseg.utils.seed import seed_everything

logger = logging.getLogger("Train")


def _strip_heavy(res: dict) -> dict:
    """Summary-friendly copy without the per-chip table."""
    return {k: v for k, v in res.items() if k != "per_chip"}


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> dict:
    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(cfg.dataset.root_dir)
    modality = cfg.dataset.modality
    split_type = cfg.dataset.split_type
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tc = cfg.train

    splits = get_split_paths(data_dir, split_type)
    stats_path = get_stats_path(data_dir, split_type, modality)
    if not stats_path.exists():
        raise FileNotFoundError(f"{stats_path} missing - run `python -m floodseg.cli compute-stats`")
    normalizer = ChannelNormalizer.from_json(stats_path)
    logger.info("Run %s | stats %s | device %s", cfg.experiment_name, stats_path, device)

    cache = cfg.dataset.cache_dir
    crop = tuple(cfg.dataset.train_crop_size) if cfg.dataset.train_crop_size else None
    train_ds = Sen1Floods11Dataset(
        data_dir, splits["train"], modality, transform=get_train_transforms(),
        normalizer=normalizer, cache_dir=cache, crop_size=crop,
    )
    val_ds = Sen1Floods11Dataset(data_dir, splits["val"], modality, normalizer=normalizer, cache_dir=cache)

    loader_kw = dict(
        num_workers=tc.num_workers,
        pin_memory=tc.pin_memory and device.type == "cuda",
        persistent_workers=tc.num_workers > 0,
    )
    g = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(train_ds, batch_size=tc.batch_size, shuffle=True, drop_last=True, generator=g, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=tc.eval_batch_size, shuffle=False, **loader_kw)

    rgb_idx = rgb_indices_for_modality(modality)
    model = FloodUNet(
        in_channels=cfg.model.in_channels,
        num_classes=cfg.model.classes,
        encoder_name=cfg.model.encoder_name,
        encoder_weights=cfg.model.encoder_weights,
        rgb_indices=rgb_idx,
    )
    n_params = sum(p.numel() for p in model.parameters())

    criterion = CombinedBCEDiceLoss(
        bce_weight=tc.bce_weight,
        dice_weight=tc.dice_weight,
        pos_weight=tc.pos_weight,
        ignore_index=cfg.dataset.ignore_index,
        mode=tc.loss,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    scheduler = warmup_cosine_scheduler(optimizer, tc.epochs, tc.warmup_epochs, tc.min_lr)

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    run = setup_wandb(
        project=cfg.wandb.project,
        name=cfg.experiment_name,
        config=cfg_dict,
        mode=cfg.wandb.mode,
        tags=list(cfg.wandb.tags),
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=out_dir,
        amp=tc.amp,
        early_stopping_patience=tc.early_stopping_patience,
        checkpoint_name="best_model.pt",
        grad_clip=tc.grad_clip,
        use_wandb=run is not None,
        checkpoint_metadata={
            "modality": modality,
            "split_type": split_type,
            "in_channels": cfg.model.in_channels,
            "encoder_name": cfg.model.encoder_name,
            "rgb_indices": list(rgb_idx) if rgb_idx else None,
            "normalizer": normalizer.to_dict(),
            "config": cfg_dict,
        },
    )
    history = trainer.fit(epochs=tc.epochs)

    # Evaluate the best checkpoint (selected on val) exactly once on held-out splits.
    state = torch.load(trainer.checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    eval_splits = {"test": splits["test"]}
    if split_type == "official":
        eval_splits["bolivia"] = data_dir / "processed" / "splits" / "official" / "flood_bolivia_data.csv"

    evals = {}
    for name, path in eval_splits.items():
        res = evaluate_model_on_split(
            model, path, data_dir, modality, normalizer, device,
            batch_size=tc.eval_batch_size, amp=tc.amp, cache_dir=cache,
        )
        evals[name] = res
        logger.info(
            "[%s] %s: IoU %.4f [%.4f, %.4f] | F1 %.4f | ECE %.4f | chips %d",
            cfg.experiment_name, name, res["iou"], res["ci_95_lower"], res["ci_95_upper"],
            res["f1"], res["ece"], res["num_chips"],
        )

    summary = {
        "exp_name": cfg.experiment_name,
        "modality": modality,
        "split_type": split_type,
        "seed": cfg.seed,
        "loss": tc.loss,
        "num_params": n_params,
        "best_epoch": history["best_epoch"],
        "epochs_run": len(history["epochs"]),
        "best_val_iou": history["best_val_iou"],
        "train_minutes": history["train_minutes"],
        "eval": {k: _strip_heavy(v) for k, v in evals.items()},
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "per_chip.json", "w", encoding="utf-8") as f:
        json.dump({k: v["per_chip"] for k, v in evals.items()}, f)

    if run is not None:
        try:
            run.summary.update({f"{k}/iou": v["iou"] for k, v in evals.items()})
            run.finish()
        except Exception as e:
            logger.warning("W&B finalisation failed (%s); local results are unaffected.", e)
    return summary


if __name__ == "__main__":
    main()
