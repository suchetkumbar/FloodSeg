"""U-Net model with ResNet-34 encoder and multi-channel input adaptation."""

from __future__ import annotations

import logging
from typing import Optional, Sequence, Tuple

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Positions of Sentinel-2 Red (B4), Green (B3), Blue (B2) in each modality's channel
# stack, ordered to match ImageNet's R, G, B input channels.
#   s2:     [B1, B2, B3, B4, ...]          -> R=3, G=2, B=1
#   fusion: [VV, VH, B1, B2, B3, B4, ...]  -> R=5, G=4, B=3
RGB_INDICES = {"s1": None, "s2": (3, 2, 1), "fusion": (5, 4, 3)}


def rgb_indices_for_modality(modality: str) -> Optional[Tuple[int, int, int]]:
    """Channel positions that receive the pretrained R, G, B kernels (None for SAR)."""
    return RGB_INDICES[modality]


def adapt_resnet_conv1(
    conv1: nn.Conv2d,
    in_channels: int,
    weight_init_method: str = "rgb_channel_mean",
    rgb_indices: Optional[Sequence[int]] = None,
) -> nn.Conv2d:
    """Adapt ImageNet-pretrained ResNet first conv layer to N input bands.

    ``rgb_indices`` gives the input-channel positions of the Red, Green and Blue
    bands. When given, pretrained R/G/B kernels are copied there; when None and
    N > 3 the first three channels are assumed to be R, G, B.

    Methodology:
    1. If in_channels == 3: preserve original weights directly.
    2. If in_channels < 3 (e.g. SAR VV/VH, N=2):
       Take the first N channels or average across RGB channels.
    3. If in_channels > 3 (e.g. Optical N=13, Early Fusion N=15):
       - Preserve the 3 pretrained channels for RGB.
       - Initialize the remaining (N - 3) channels using the channel-wise mean of RGB weights.
       - Apply variance scaling factor (3.0 / in_channels) across all channels to preserve
         the expected activation variance entering downstream residual layers.
    """
    if conv1.in_channels == in_channels:
        return conv1

    old_weight = conv1.weight.data  # [out_channels, 3, 7, 7]
    out_channels = conv1.out_channels
    kernel_size = conv1.kernel_size
    stride = conv1.stride
    padding = conv1.padding
    bias = conv1.bias is not None

    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
    )

    with torch.no_grad():
        if in_channels < 3:
            # Average across RGB to initialize SAR channels
            rgb_mean = old_weight.mean(dim=1, keepdim=True)  # [out_channels, 1, 7, 7]
            new_weight = rgb_mean.repeat(1, in_channels, 1, 1) * (3.0 / in_channels)
        else:
            # Multi-spectral or fused: copy RGB and fill extra channels with RGB mean
            new_weight = torch.zeros(
                (out_channels, in_channels, kernel_size[0], kernel_size[1]),
                dtype=old_weight.dtype,
                device=old_weight.device,
            )
            # Every channel starts from the channel-mean of the RGB kernels ...
            rgb_mean = old_weight.mean(dim=1, keepdim=True)
            new_weight[:] = rgb_mean.repeat(1, in_channels, 1, 1)

            # ... then the true R, G, B bands receive their pretrained kernels.
            targets = list(rgb_indices) if rgb_indices is not None else [0, 1, 2]
            if len(targets) != 3 or len(set(targets)) != 3 or max(targets) >= in_channels:
                raise ValueError(f"Invalid rgb_indices {rgb_indices} for {in_channels} channels")
            for src_c, dst_c in enumerate(targets):
                new_weight[:, dst_c, :, :] = old_weight[:, src_c, :, :]

            # Variance scaling factor to maintain activation scale
            new_weight = new_weight * (3.0 / in_channels)

        new_conv.weight.data.copy_(new_weight)
        if bias and conv1.bias is not None:
            new_conv.bias.data.copy_(conv1.bias.data)

    logger.info(
        "Adapted first conv from 3 to %d channels (variance scale: %.3f)",
        in_channels,
        3.0 / in_channels,
    )
    return new_conv


class FloodUNet(nn.Module):
    """U-Net with ResNet-34 encoder for flood water segmentation."""

    def __init__(
        self,
        in_channels: int = 15,
        num_classes: int = 1,
        encoder_name: str = "resnet34",
        encoder_weights: Optional[str] = "imagenet",
        weight_init_method: str = "rgb_channel_mean",
        rgb_indices: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Instantiate SMP U-Net with standard 3-channel pretrained encoder
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
            activation=None,  # Return raw logits for AMP and BCEWithLogitsLoss
        )

        # Adapt first conv layer to arbitrary in_channels
        if in_channels != 3:
            self.model.encoder.conv1 = adapt_resnet_conv1(
                self.model.encoder.conv1,
                in_channels=in_channels,
                weight_init_method=weight_init_method,
                rgb_indices=rgb_indices,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, C, H, W]
        Returns:
            Logits tensor [B, 1, H, W]
        """
        return self.model(x)
