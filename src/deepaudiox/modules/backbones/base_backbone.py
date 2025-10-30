# deepaudiox/modules/backbones/base_backbone.py

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseBackbone(nn.Module, ABC):
    """Abstract base class for all audio backbone models."""

    out_dim: int  # every subclass must set this in __init__
    sample_frequency: int  # every subclass must set this in __init__

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute forward pass returning an embedding tensor of shape (B, out_dim)."""
        pass

    @abstractmethod
    def load_pretrained_encoder(self, weights: str) -> None:
        """Optionally load pretrained model weights from a file path."""
        pass

    @abstractmethod
    def freeze_encoder_weights(self) -> None:
        """Freeze encoder parameters to disable gradient updates."""
        pass
