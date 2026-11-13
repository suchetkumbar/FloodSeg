"""Loss functions for flood water segmentation."""

from floodseg.losses.combined import CombinedBCEDiceLoss, SoftDiceLoss

__all__ = ["CombinedBCEDiceLoss", "SoftDiceLoss"]
