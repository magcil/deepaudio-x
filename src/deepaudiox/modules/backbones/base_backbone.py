# deepaudiox/modules/backbones/base_backbone.py

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseBackbone(nn.Module, ABC):
    """Abstract base class for all audio backbone models."""

    def __init__(self, out_dim: int, sample_frequency: int) -> None:
        """Initialize the BaseBackbone.
        Args:
            out_dim (int): Output dimension of the backbone embeddings.
            sample_frequency (int): Sample frequency for audio input.
        """
        super().__init__()
        self.out_dim = out_dim
        self.sample_frequency = sample_frequency

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
