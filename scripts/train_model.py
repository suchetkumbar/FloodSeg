"""Training runner script for FloodSeg experiments.

Supports:
- Modalities: s1, s2, fusion
- Loss formulations: bce_dice, dice_only (loss ablation)
- Multi-seed runs (e.g. 42, 1337, 2024)
- AMP mixed precision training with CUDA
- Early stopping on validation Water IoU
- Checkpoint saving and test set evaluation
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from floodseg.data.dataset import Sen1Floods11Dataset
from floodseg.data.normalizer import ChannelNormalizer
from floodseg.data.transforms import get_train_transforms
from floodseg.losses.combined import CombinedBCEDiceLoss
from floodseg.metrics.evaluator import SegmentationMetrics, bootstrap_ci, compute_sample_iou
from floodseg.models.train_engine import Trainer
from floodseg.models.unet import FloodUNet
from floodseg.utils.logging import setup_wandb
from floodseg.utils.seed import seed_everything

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Train")


def evaluate_checkpoint_on_split(
    model: nn.Module,
    split_path: Path,
    data_dir: Path,
    modality: str,
    normalizer: Optional[ChannelNormalizer],
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate a trained model checkpoint on an evaluation split."""
    ds = Sen1Floods11Dataset(
        root_dir=data_dir,
        split_file=split_path,
        modality=modality,
        normalizer=normalizer,
    )
    if len(ds) == 0:
        return {}

    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    model.eval()
    metrics = SegmentationMetrics(ignore_index=-1)
    chip_ious = []

    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            masks = batch["mask"].to(device)

            logits = model(imgs)
            preds = (torch.sigmoid(logits) > 0.5).squeeze(1).long()

            metrics.update(preds, masks)

            # Compute chip-level IoUs
            p_np = preds.cpu().numpy()
            m_np = masks.cpu().numpy()
            for p, m in zip(p_np, m_np):
                chip_ious.append(compute_sample_iou(p, m, ignore_index=-1))

    res = metrics.compute()
    mean_ci, low_ci, high_ci = bootstrap_ci(chip_ious)
    res["chip_mean_iou"] = mean_ci
    res["ci_95_lower"] = low_ci
    res["ci_95_upper"] = high_ci
    res["num_chips"] = len(ds)
    return res


def run_experiment(
    modality: str = "fusion",
    seed: int = 42,
    epochs: int = 30,
    batch_size: int = 8,
    lr: float = 3.0e-4,
    loss_mode: str = "bce_dice",
    pos_weight: Optional[float] = None,
    split_type: str = "official",
    data_dir: Path = Path("data"),
    output_dir: Path = Path("outputs"),
    encoder_weights: str = "imagenet",
) -> Dict[str, Any]:
    """Run a single training experiment with specified configuration."""
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    channel_map = {"s1": 2, "s2": 13, "fusion": 15}
    in_channels = channel_map[modality]

    exp_name = f"exp_{modality}_{loss_mode}_seed{seed}"
    exp_output_dir = output_dir / exp_name
    exp_output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Setup Splits
    processed_dir = data_dir / "processed"
    if split_type == "official":
        train_split = processed_dir / "splits" / "official" / "flood_train_data.csv"
        val_split = processed_dir / "splits" / "official" / "flood_val_data.csv"
        test_split = processed_dir / "splits" / "official" / "flood_test_data.csv"
    else:
        train_split = processed_dir / "splits" / "loeo" / "loeo_train_data.csv"
        val_split = processed_dir / "splits" / "loeo" / "loeo_val_data.csv"
        test_split = processed_dir / "splits" / "loeo" / "loeo_held_out_test_data.csv"

    # 2. Load Normalizer
    stats_file = data_dir / "stats" / f"train_stats_official_{modality}.json"
    normalizer = ChannelNormalizer.from_json(stats_file) if stats_file.exists() else None

    # 3. Create Datasets & DataLoaders
    train_transform = get_train_transforms(p_flip=0.5, p_rot=0.5)
    train_ds = Sen1Floods11Dataset(
        root_dir=data_dir,
        split_file=train_split,
        modality=modality,
        transform=train_transform,
        normalizer=normalizer,
    )
    val_ds = Sen1Floods11Dataset(
        root_dir=data_dir,
        split_file=val_split,
        modality=modality,
        normalizer=normalizer,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available()
    )

    # 4. Model Architecture & First Conv Adaptation
    model = FloodUNet(
        in_channels=in_channels,
        num_classes=1,
        encoder_name="resnet34",
        encoder_weights=encoder_weights if encoder_weights != "none" else None,
    )

    # 5. Loss & Optimizer
    criterion = CombinedBCEDiceLoss(
        bce_weight=0.5,
        dice_weight=0.5,
        pos_weight=pos_weight,
        ignore_index=-1,
        mode=loss_mode,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1.0e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # 6. W&B logging
    setup_wandb(
        project="floodseg",
        name=exp_name,
        config={
            "modality": modality,
            "seed": seed,
            "loss": loss_mode,
            "lr": lr,
            "batch_size": batch_size,
            "epochs": epochs,
            "in_channels": in_channels,
        },
        mode="offline",
    )

    # 7. Train with Early Stopping
    ckpt_name = f"best_model_{exp_name}.pt"
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=exp_output_dir,
        amp=True,
        early_stopping_patience=8,
        checkpoint_name=ckpt_name,
    )

    history = trainer.fit(epochs=epochs, run_name=exp_name)

    # 8. Load Best Checkpoint & Evaluate on Test Set
    best_ckpt = exp_output_dir / ckpt_name
    if best_ckpt.exists():
        state = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        logger.info("Loaded best model state from %s (Val IoU: %.4f)", best_ckpt, state.get("val_iou", 0.0))

    test_results = evaluate_checkpoint_on_split(
        model=model,
        split_path=test_split,
        data_dir=data_dir,
        modality=modality,
        normalizer=normalizer,
        device=device,
    )
    logger.info("Test Set Evaluation (%s): IoU = %.4f | F1 = %.4f", exp_name, test_results.get("iou", 0.0), test_results.get("f1", 0.0))

    summary = {
        "exp_name": exp_name,
        "modality": modality,
        "seed": seed,
        "loss_mode": loss_mode,
        "best_epoch": history.get("best_epoch"),
        "best_val_iou": history.get("best_val_iou"),
        "test_results": test_results,
    }

    summary_file = exp_output_dir / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FloodSeg Model")
    parser.add_argument("--modality", type=str, default="fusion", choices=["s1", "s2", "fusion"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--loss", type=str, default="bce_dice", choices=["bce_dice", "dice_only"])
    parser.add_argument("--pos-weight", type=float, default=None)
    parser.add_argument("--split", type=str, default="official", choices=["official", "loeo"])
    parser.add_argument("--encoder-weights", type=str, default="imagenet")
    args = parser.parse_args()

    run_experiment(
        modality=args.modality,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        loss_mode=args.loss,
        pos_weight=args.pos_weight,
        split_type=args.split,
        encoder_weights=args.encoder_weights,
    )
