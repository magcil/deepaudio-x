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
    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Computes of the embeddings of the input features.

        Args:
            x: (torch.Tensor) Input audio-specific features of shape (B, 1, F, T) or (B, 1, T, F)
            padding_mask: (torch.Tensor) Optional padding mask.

        Returns:
            torch.Tensor: Embeddings of shape (B, D), where D is the embedding dimension.
        """
        pass

    @abstractmethod
    def extract_features(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Convert raw waveforms into internal acoustic features.

        Args:
            waveforms (torch.Tensor): Tensor of shape (B, T).

        Returns:
            torch.Tensor: Model-specific feature representation before final forward().
        """
        pass

    @abstractmethod
    def freeze_encoder_weights(self) -> None:
        """Freeze encoder parameters to disable gradient updates."""
        pass

    def forward_pipeline(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard processing pipeline:
            1. Extract features from raw audio
            2. Pass features through forward()

        Args:
            x (torch.Tensor): Input waveforms of shape (B, T), where T is the length of waveforms.

        Returns:
            torch.Tensor:
                Final model output of shape (B, out_dim).
        """
        x = self.extract_features(x)
        return self.forward(x)
