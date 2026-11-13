"""Combined Binary Cross-Entropy and Soft-Dice Loss for flood water segmentation.

Strictly excludes invalid/no-data pixels (-1) from both loss components.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class SoftDiceLoss(nn.Module):
    """Soft Dice Loss operating over valid pixels only."""

    def __init__(self, smooth: float = 1.0, ignore_index: int = -1):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Predicted raw logits of shape [B, 1, H, W] or [B, H, W].
            target: Ground truth masks with values {-1, 0, 1}, shape [B, H, W] or [B, 1, H, W].
        """
        if logits.dim() == 4 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        if target.dim() == 4 and target.shape[1] == 1:
            target = target.squeeze(1)

        probs = torch.sigmoid(logits)
        valid_mask = (target != self.ignore_index)

        probs_valid = probs[valid_mask]
        target_valid = target[valid_mask].float()

        if target_valid.numel() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        intersection = torch.sum(probs_valid * target_valid)
        cardinality = torch.sum(probs_valid) + torch.sum(target_valid)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice


class CombinedBCEDiceLoss(nn.Module):
    """Joint Binary Cross-Entropy and Soft Dice loss."""

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        pos_weight: Optional[float] = None,
        ignore_index: int = -1,
        mode: str = "bce_dice",  # 'bce_dice' or 'dice_only'
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index
        self.mode = mode

        pos_weight_tensor = torch.tensor([pos_weight]) if pos_weight is not None else None
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor, reduction="none")
        self.dice_loss = SoftDiceLoss(ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.dim() == 4 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        if target.dim() == 4 and target.shape[1] == 1:
            target = target.squeeze(1)

        valid_mask = (target != self.ignore_index)
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        if self.mode == "dice_only":
            return self.dice_loss(logits, target)

        # Compute masked BCE
        target_float = target.float()
        raw_bce = self.bce_loss(logits, target_float)
        bce = raw_bce[valid_mask].mean()

        # Compute Dice
        dice = self.dice_loss(logits, target)

        return (self.bce_weight * bce) + (self.dice_weight * dice)
