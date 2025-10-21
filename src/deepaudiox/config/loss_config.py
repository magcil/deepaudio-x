import torch
from dataclasses import dataclass
from .loss_config_registry import register_loss_config


@dataclass
class LossConfig:
    name: str = "CrossEntropyLoss"

@dataclass 
@register_loss_config("CrossEntropyLoss")
class CrossEntropyLossConfig(LossConfig):
    """Configuration for setting up a Cross Entropy Loss module."""

    # weight: torch.Tensor = None,
    # ignore_index: int = -100,
    size_average: bool = True
    reduce: bool = True
    reduction: str = 'mean'
    label_smoothing: float = 0.0

