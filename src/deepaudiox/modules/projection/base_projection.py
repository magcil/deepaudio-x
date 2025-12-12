# deepaudiox/modules/projection/base_projection.py

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseProjection(nn.Module, ABC):
    """Abstract base class for all projection modules.

    This class defines the interface for modules that project an input
    representation from one dimensionality to another, typically as a final
    transformation step in a larger model.

    Methods:
        __init__: Store input and output dimensionalities.
        forward: Apply the projection to an input tensor and return the result.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        """Initialize the BaseProjection.

        Args:
            in_dim (int): Input dimension.
            out_dim (int): Output dimension.
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute forward pass returning a projected tensor."""
        pass
