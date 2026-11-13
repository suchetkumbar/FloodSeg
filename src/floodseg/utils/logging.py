"""Weights & Biases and local experiment logging utilities."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import wandb

logger = logging.getLogger(__name__)


def setup_wandb(
    project: str = "floodseg",
    name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    mode: str = "offline",  # 'offline' allows free runs without network or accounts
    tags: Optional[list] = None,
) -> Optional[wandb.sdk.wandb_run.Run]:
    """Initialize Weights & Biases run with graceful offline handling.

    Args:
        project: W&B project name.
        name: Experiment run name.
        config: Hyperparameters dictionary.
        mode: 'offline' or 'online'.
        tags: Optional tags list.
    """
    if mode == "disabled":
        os.environ["WANDB_MODE"] = "disabled"
        return None

    os.environ["WANDB_MODE"] = mode
    try:
        run = wandb.init(
            project=project,
            name=name,
            config=config,
            mode=mode,
            tags=tags,
            reinit=True,
        )
        logger.info("Initialized W&B (Mode: %s, Run: %s)", mode, name)
        return run
    except Exception as e:
        logger.warning("Could not initialize W&B: %s. Continuing in offline mode.", e)
        return None
