"""Inference utilities: checkpoint loading and tiled prediction over arbitrarily large scenes.

Checkpoints written by ``Trainer`` are self-describing (modality, channels, RGB band
mapping, train-split normalizer), so a model can be rebuilt from the file alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from floodseg.data.normalizer import ChannelNormalizer
from floodseg.models.unet import FloodUNet


def load_model_from_checkpoint(
    path: Union[str, Path], device: Optional[torch.device] = None
) -> Tuple[nn.Module, ChannelNormalizer, Dict]:
    """Rebuild a FloodUNet from a training checkpoint.

    Returns (model in eval mode on ``device``, normalizer, metadata).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = FloodUNet(
        in_channels=state["in_channels"],
        num_classes=1,
        encoder_name=state.get("encoder_name", "resnet34"),
        encoder_weights=None,
        rgb_indices=state.get("rgb_indices"),
    )
    model.load_state_dict(state["model_state_dict"])
    model.to(device).eval()
    norm = state["normalizer"]
    normalizer = ChannelNormalizer(norm["mean"], norm["std"], norm.get("eps", 1e-6))
    meta = {k: v for k, v in state.items() if k not in ("model_state_dict", "normalizer")}
    return model, normalizer, meta


def _hann_window(size: int) -> np.ndarray:
    """2-D blending weights, strictly positive so every pixel gets weight."""
    w = np.hanning(size + 2)[1:-1]
    return np.outer(w, w).astype(np.float32)


def _starts(length: int, tile: int, stride: int) -> List[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


@torch.no_grad()
def predict_tiled(
    models: Union[nn.Module, Sequence[nn.Module]],
    image: np.ndarray,
    normalizer: Union[ChannelNormalizer, Sequence[ChannelNormalizer]],
    tile: int = 512,
    overlap: int = 128,
    batch_size: int = 4,
    device: Optional[torch.device] = None,
    amp: bool = True,
) -> np.ndarray:
    """Water probability [H, W] for a raw (unnormalised) [H, W, C] scene.

    Overlapping tiles are blended with a Hann window, which removes the seams that
    tile-edge context loss would otherwise create. Passing several models (e.g. one
    per seed) averages their probabilities, a simple ensemble. Non-finite inputs are
    zero-filled after normalisation (the training-mean value), as in training; such
    pixels should be masked by the caller.
    """
    models = [models] if isinstance(models, nn.Module) else list(models)
    norms = [normalizer] * len(models) if isinstance(normalizer, ChannelNormalizer) else list(normalizer)
    device = device or next(models[0].parameters()).device
    use_amp = amp and device.type == "cuda"

    H, W, _ = image.shape
    pad_h, pad_w = max(0, tile - H), max(0, tile - W)
    if pad_h or pad_w:
        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    Hp, Wp = image.shape[:2]
    stride = tile - overlap
    weight = _hann_window(tile)
    coords = [(y, x) for y in _starts(Hp, tile, stride) for x in _starts(Wp, tile, stride)]

    prob_sum = np.zeros((Hp, Wp), dtype=np.float32)
    w_sum = np.zeros((Hp, Wp), dtype=np.float32)
    for i in range(0, len(coords), batch_size):
        chunk = coords[i : i + batch_size]
        tiles = np.stack([image[y : y + tile, x : x + tile] for y, x in chunk])  # [B, t, t, C]
        probs = np.zeros((len(chunk), tile, tile), dtype=np.float32)
        for model, norm in zip(models, norms):
            x_np = np.nan_to_num(norm(tiles), nan=0.0, posinf=0.0, neginf=0.0)
            x = torch.from_numpy(np.ascontiguousarray(x_np.transpose(0, 3, 1, 2))).to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x)
            probs += torch.sigmoid(logits.float())[:, 0].cpu().numpy()
        probs /= len(models)
        for (y, x), p in zip(chunk, probs):
            prob_sum[y : y + tile, x : x + tile] += p * weight
            w_sum[y : y + tile, x : x + tile] += weight

    return (prob_sum / np.maximum(w_sum, 1e-8))[:H, :W]
