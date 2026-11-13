"""PyTorch training and validation engine for FloodSeg.

Features:
- Automatic Mixed Precision (AMP FP16)
- Early stopping on validation Water IoU
- Checkpointing of best models
- Weights & Biases logging (with offline mode support)
- Gradient clipping and LR scheduling
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from floodseg.metrics.evaluator import SegmentationMetrics

logger = logging.getLogger(__name__)


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
        self.amp = amp and torch.cuda.is_available()
        self.early_stopping_patience = early_stopping_patience
        self.checkpoint_name = checkpoint_name

        self.model.to(self.device)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)

        self.best_val_iou = -1.0
        self.best_epoch = 0
        self.patience_counter = 0

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        metrics = SegmentationMetrics(ignore_index=-1)

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch:02d} [Train]", leave=False)
        for batch in pbar:
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=self.amp):
                logits = self.model(images)
                loss = self.criterion(logits, masks)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).long()
            metrics.update(preds, masks)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        if self.scheduler is not None:
            self.scheduler.step()

        epoch_loss = total_loss / max(1, len(self.train_loader))
        res = metrics.compute()
        res["loss"] = epoch_loss
        return res

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate over validation set."""
        self.model.eval()
        total_loss = 0.0
        metrics = SegmentationMetrics(ignore_index=-1)

        for batch in self.val_loader:
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=self.amp):
                logits = self.model(images)
                loss = self.criterion(logits, masks)

            total_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).long()
            metrics.update(preds, masks)

        val_loss = total_loss / max(1, len(self.val_loader))
        res = metrics.compute()
        res["loss"] = val_loss
        return res

    def fit(self, epochs: int, run_name: Optional[str] = None) -> Dict[str, Any]:
        """Full training loop with early stopping."""
        history = {"train_loss": [], "val_loss": [], "train_iou": [], "val_iou": []}

        logger.info(
            "Starting training for %d epochs on %s (AMP: %s)",
            epochs, self.device, self.amp
        )

        for epoch in range(1, epochs + 1):
            train_res = self.train_epoch(epoch)
            val_res = self.validate()

            train_loss, train_iou = train_res["loss"], train_res["iou"]
            val_loss, val_iou = val_res["loss"], val_res["iou"]

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_iou"].append(train_iou)
            history["val_iou"].append(val_iou)

            lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                "Epoch %02d/%d | Train Loss: %.4f | Train IoU: %.4f | Val Loss: %.4f | Val IoU: %.4f | LR: %.2e",
                epoch, epochs, train_loss, train_iou, val_loss, val_iou, lr
            )

            # Log to W&B if active
            if wandb.run is not None:
                wandb.log({
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/iou": train_iou,
                    "val/loss": val_loss,
                    "val/iou": val_iou,
                    "val/f1": val_res["f1"],
                    "lr": lr,
                })

            # Checkpoint best model
            if val_iou > self.best_val_iou:
                self.best_val_iou = val_iou
                self.best_epoch = epoch
                self.patience_counter = 0

                ckpt_path = self.output_dir / self.checkpoint_name
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_iou": val_iou,
                    "val_f1": val_res["f1"],
                }, ckpt_path)
                logger.info("Saved new best checkpoint (Val IoU: %.4f) -> %s", val_iou, ckpt_path)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    logger.info("Early stopping triggered after %d epochs without improvement.", self.early_stopping_patience)
                    break

        history["best_val_iou"] = self.best_val_iou
        history["best_epoch"] = self.best_epoch
        return history
