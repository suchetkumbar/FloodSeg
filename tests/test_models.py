"""Unit tests for FloodUNet model, channel adaptation, and loss functions."""

import pytest
import torch

from floodseg.losses.combined import CombinedBCEDiceLoss, SoftDiceLoss
from floodseg.models.unet import FloodUNet, adapt_resnet_conv1


@pytest.mark.parametrize("in_channels", [2, 13, 15])
def test_flood_unet_forward_pass(in_channels):
    """Test FloodUNet forward pass with different input modalities."""
    batch_size = 2
    height, width = 64, 64  # Fast resolution for test

    model = FloodUNet(
        in_channels=in_channels,
        num_classes=1,
        encoder_name="resnet34",
        encoder_weights=None,  # No remote download during unit test
    )
    x = torch.randn(batch_size, in_channels, height, width)
    logits = model(x)

    assert logits.shape == (batch_size, 1, height, width)
    assert not torch.isnan(logits).any()


def test_first_conv_adaptation_weights():
    """Verify first conv weight adaptation and variance scaling."""
    conv3 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    torch.nn.init.constant_(conv3.weight, 1.0)

    # Adapt to 15 channels (Early Fusion)
    conv15 = adapt_resnet_conv1(conv3, in_channels=15)
    assert conv15.in_channels == 15
    assert conv15.weight.shape == (64, 15, 7, 7)

    # Scale factor 3 / 15 = 0.2
    assert torch.allclose(conv15.weight, torch.tensor(0.2))


def test_loss_functions_with_no_data():
    """Verify BCE+Dice and Dice-only losses ignore -1 target values."""
    logits = torch.randn(2, 1, 32, 32, requires_grad=True)
    target = torch.randint(0, 2, (2, 32, 32)).long()

    # Set some pixels to -1
    target[:, 0:10, :] = -1

    loss_fn = CombinedBCEDiceLoss(ignore_index=-1)
    loss = loss_fn(logits, target)

    assert loss.item() >= 0.0
    assert not torch.isnan(loss)

    # Test backward pass
    loss.backward()
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()


def test_dice_only_ablation():
    """Verify Dice-only loss ablation formulation."""
    logits = torch.randn(2, 1, 32, 32)
    target = torch.randint(0, 2, (2, 32, 32)).long()
    target[:, 0:5, :] = -1

    dice_fn = SoftDiceLoss(ignore_index=-1)
    loss = dice_fn(logits, target)

    assert 0.0 <= loss.item() <= 1.0


@pytest.mark.parametrize("modality,in_ch", [("s2", 13), ("fusion", 15)])
def test_pretrained_rgb_kernels_land_on_true_rgb_bands(modality, in_ch):
    """ImageNet R, G, B kernels must be copied to S2 B4, B3, B2 (not B1/VV/VH)."""
    from floodseg.models.unet import rgb_indices_for_modality

    conv3 = torch.nn.Conv2d(3, 8, kernel_size=7, bias=False)
    with torch.no_grad():
        for c in range(3):
            conv3.weight[:, c] = float(c + 1)  # R=1, G=2, B=3
    idx = rgb_indices_for_modality(modality)
    conv = adapt_resnet_conv1(conv3, in_channels=in_ch, rgb_indices=idx)
    scale = 3.0 / in_ch
    for src_c, dst_c in enumerate(idx):
        assert torch.allclose(conv.weight[:, dst_c], torch.tensor((src_c + 1) * scale))
    others = [c for c in range(in_ch) if c not in idx]
    assert torch.allclose(conv.weight[:, others], torch.tensor(2.0 * scale))  # RGB mean
