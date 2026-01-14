# deepaudiox/modules/backbones/base_backbone.py

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseBackbone(nn.Module, ABC):
    """Abstract base class for all audio backbone models.

    This class defines the common interface for backbone architectures that
    convert raw waveforms into fixed-dimensional embeddings. Subclasses must
    implement the core feature extraction and forward-processing logic.

    Methods:
        __init__: Initializes the embedding dimension and the sample_rate of the audios.
        forward: Computes embeddings from pre-extracted audio features.
        extract_features: Converts raw waveforms into model-specific features.
        forward_pipeline: Extracts features and then applies forward().
    """

    def __init__(self, out_dim: int, sample_rate: int) -> None:
        """Initialize the BaseBackbone.

        Args:
            out_dim (int): Output dim of the backbone feature map. For CNNs the embeddings are of shape (B, C, H, W)
            and for Transformers of shape (B, T, D), where out_dim is either C or D respectively. The output embeddings
            could be of shape (B, out_dim) in case of pooling backbones.
            sample_rate (int): Sample rate for audio input.
        """
        super().__init__()
        self.out_dim = out_dim
        self.sample_rate = sample_rate

    @abstractmethod
    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Computes of the embeddings of the input features.

        Args:
            x: (torch.Tensor) Input audio-specific features of shape (B, 1, F, T) or (B, 1, T, F)
            padding_mask: (torch.Tensor) Optional padding mask.

        Returns:
            torch.Tensor: Embeddings of shape (B, T, D) or (B, D, H, W) where D is the embedding dimension.
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

    def forward_pipeline(self, x: torch.Tensor) -> torch.Tensor:
        """Standard processing pipeline:

            1. Extract features from raw audio
            2. Pass features through forward()

        Args:
            x (torch.Tensor): Input waveforms of shape (B, T), where T is the length of waveforms.

        Returns:
            torch.Tensor: Final model output of shape (B, out_dim, H, W) for CNNs or (B, T, out_dim) for Transformers.
        """
        x = self.extract_features(x)
        return self.forward(x)
