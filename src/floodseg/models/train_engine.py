"""PyTorch training and validation engine for FloodSeg.

Features:
- Automatic Mixed Precision (forward in FP16, loss computed in FP32)
- Linear warmup + cosine LR schedule (stepped per epoch)
- Early stopping and best-checkpointing on validation (pooled) water IoU
- Self-describing checkpoints (modality, channels, normalizer stats) for inference
- Optional Weights & Biases logging; always writes a local history.json
"""

from __future__ import annotations

import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from floodseg.metrics.evaluator import SegmentationMetrics

logger = logging.getLogger(__name__)


def warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    warmup_epochs: int = 0,
    min_lr: float = 1e-6,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Per-epoch schedule: linear warmup to base LR, then cosine decay to ``min_lr``."""
    base_lr = optimizer.param_groups[0]["lr"]
    floor = min_lr / base_lr

    def factor(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / (warmup_epochs + 1)
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


class Trainer:
    """End-to-end trainer for FloodSeg models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
        output_dir: Path = Path("outputs"),
        amp: bool = True,
        early_stopping_patience: int = 10,
        checkpoint_name: str = "best_model.pt",
        checkpoint_metadata: Optional[Dict[str, Any]] = None,
        grad_clip: float = 1.0,
        use_wandb: bool = False,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.amp = amp and self.device.type == "cuda"
        self.early_stopping_patience = early_stopping_patience
        self.checkpoint_path = self.output_dir / checkpoint_name
        self.checkpoint_metadata = checkpoint_metadata or {}
        self.grad_clip = grad_clip
        self.use_wandb = use_wandb

        self.model.to(self.device)
        self.criterion.to(self.device)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)

        self.best_val_iou = -1.0
        self.best_epoch = 0
        self.patience_counter = 0

    def _forward_loss(self, images: torch.Tensor, masks: torch.Tensor):
        with torch.amp.autocast("cuda", enabled=self.amp):
            logits = self.model(images)
        # Loss in FP32: Dice sums over millions of pixels.
        loss = self.criterion(logits.float(), masks)
        return logits, loss

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss, n_batches = 0.0, 0
        metrics = SegmentationMetrics(ignore_index=-1)

        pbar = tqdm(
            self.train_loader, desc=f"Epoch {epoch:02d} [Train]", leave=False, disable=not sys.stderr.isatty()
        )
        for batch in pbar:
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            logits, loss = self._forward_loss(images, masks)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            n_batches += 1
            metrics.update((logits.detach() > 0).long(), masks)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        if self.scheduler is not None:
            self.scheduler.step()

        res = metrics.compute()
        res["loss"] = total_loss / max(1, n_batches)
        return res

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        total_loss, n_batches = 0.0, 0
        metrics = SegmentationMetrics(ignore_index=-1)
        for batch in self.val_loader:
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)
            logits, loss = self._forward_loss(images, masks)
            total_loss += loss.item()
            n_batches += 1
            metrics.update((logits > 0).long(), masks)  # logit > 0  <=>  p > 0.5
        res = metrics.compute()
        res["loss"] = total_loss / max(1, n_batches)
        return res

    def save_checkpoint(self, epoch: int, val_res: Dict[str, float]) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "val_iou": val_res["iou"],
                "val_f1": val_res["f1"],
                **self.checkpoint_metadata,
            },
            self.checkpoint_path,
        )

    def fit(self, epochs: int) -> Dict[str, Any]:
        """Full training loop with early stopping."""
        history: Dict[str, Any] = {"epochs": []}
        logger.info("Training for up to %d epochs on %s (AMP: %s)", epochs, self.device, self.amp)
        t_start = time.time()

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            lr = self.optimizer.param_groups[0]["lr"]
            train_res = self.train_epoch(epoch)
            val_res = self.validate()

            row = {
                "epoch": epoch,
                "lr": lr,
                "train_loss": train_res["loss"],
                "train_iou": train_res["iou"],
                "val_loss": val_res["loss"],
                "val_iou": val_res["iou"],
                "val_f1": val_res["f1"],
                "seconds": time.time() - t0,
            }
            history["epochs"].append(row)
            logger.info(
                "Epoch %02d/%d | train loss %.4f IoU %.4f | val loss %.4f IoU %.4f | lr %.2e | %.1fs",
                epoch, epochs, row["train_loss"], row["train_iou"], row["val_loss"], row["val_iou"], lr, row["seconds"],
            )
            if self.use_wandb:
                try:
                    import wandb

                    if wandb.run is not None:
                        wandb.log(dict(row))
                except Exception as e:  # W&B is optional; never let it kill a training run
                    logger.warning("W&B logging failed (%s); continuing without it.", e)
                    self.use_wandb = False

            if val_res["iou"] > self.best_val_iou:
                self.best_val_iou = val_res["iou"]
                self.best_epoch = epoch
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_res)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    logger.info("Early stopping: no val IoU gain for %d epochs.", self.early_stopping_patience)
                    break

        history["best_val_iou"] = self.best_val_iou
        history["best_epoch"] = self.best_epoch
        history["train_minutes"] = (time.time() - t_start) / 60.0
        with open(self.output_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        return history

