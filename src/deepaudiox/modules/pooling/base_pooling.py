# deepaudiox/modules/projection/base_projection.py

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BasePooling(nn.Module, ABC):
    """Abstract base class for all pooling modules.

    This class defines the interface for pooling that operate an input
    feature map obtained from a CNN or a Transformer BaseBackbone. Subclasses must
    implement the forward-processing logic. The input is expected to be a feature map of shape (B, D, H, W) for CNNs
    or (B, T, D) for Transformers.

    Methods:
        __init__: Store input dimensionality.
        forward: Apply the pooling module to an input tensor and return the result.
    """

    def __init__(self, in_dim: int | None = None) -> None:
        """Initialize the BaseProjection.

        Args:
            in_dim (int): Input dimension. This is D for both CNNs and Transformers.
        """
        super().__init__()
        self.in_dim = in_dim

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute forward pass returning a projected tensor."""
        pass
