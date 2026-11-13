"""Geospatial data augmentations and preprocessing transforms.

Enforces physics-aware constraints:
- Spatial symmetries: Flips (horizontal/vertical) and 90-degree rotations.
- Random crops.
- STRICT RULE: No photometric jitter, color jitter, hue/contrast shifts on SAR channels.
"""

from __future__ import annotations

from typing import Optional, Tuple

import albumentations as A
import numpy as np
import torch


def get_train_transforms(
    crop_size: Optional[Tuple[int, int]] = None,
    p_flip: float = 0.5,
    p_rot: float = 0.5,
) -> A.Compose:
    """Compose data augmentation transforms for training.

    Preserves radiometric fidelity: Only rigid spatial transforms are applied.
    """
    transforms = []
    if crop_size is not None:
        transforms.append(
            A.RandomCrop(
                height=crop_size[0],
                width=crop_size[1],
                p=1.0,
            )
        )

    transforms.extend(
        [
            A.HorizontalFlip(p=p_flip),
            A.VerticalFlip(p=p_flip),
            A.RandomRotate90(p=p_rot),
        ]
    )

    return A.Compose(
        transforms,
        is_check_shapes=False,
    )


def get_val_transforms(
    crop_size: Optional[Tuple[int, int]] = None,
) -> A.Compose:
    """Compose deterministic validation/test preprocessing transforms."""
    transforms = []
    if crop_size is not None:
        transforms.append(
            A.CenterCrop(
                height=crop_size[0],
                width=crop_size[1],
                p=1.0,
            )
        )
    return A.Compose(transforms, is_check_shapes=False)


def apply_transforms(
    image: np.ndarray,
    mask: np.ndarray,
    transform: Optional[A.Compose] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply albumentations transforms to an image [H, W, C] and mask [H, W].

    Converts output to PyTorch tensors with shape [C, H, W] for image
    and [H, W] for mask. Mask values {-1, 0, 1} are strictly preserved.
    """
    if transform is not None:
        augmented = transform(image=image, mask=mask)
        image = augmented["image"]
        mask = augmented["mask"]

    # Convert image from [H, W, C] to [C, H, W] float32 Tensor
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
    image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).contiguous().float()

    # Convert mask to int64 or float32 Tensor
    mask_tensor = torch.from_numpy(mask).contiguous().long()

    return image_tensor, mask_tensor
