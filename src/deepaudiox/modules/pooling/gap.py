# deepaudiox/modules/pooling/gap.py

# Implementation of the GlobalAveragePooling module
import torch

from deepaudiox.modules.baseclasses import BasePooling


class GAP(BasePooling):
    """Global Average Pooling (GAP) Layer."""

    def __init__(self):
        """Initialize the GAP module."""
        super().__init__(in_dim=None)  # GAP does not require a predefined input dimension

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Global Average Pooling to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (B, D, H, W) for CNNs or (B, T, D) for Transformers.

        Returns:
            torch.Tensor: Output tensor of shape (B, D).
        """
        if len(x.shape) == 4:
            # Input shape: (B, D, H, W) for CNNs
            return x.mean(dim=[2, 3])  # Average over H and W
        elif len(x.shape) == 3:
            # Input shape: (B, T, D) for Transformers
            return x.mean(dim=1)  # Average over T
        elif len(x.shape) == 2:  # (B, D) Already pooled input return as is
            return x
        else:
            raise ValueError(f"Unsupported input shape for GAP: {x.shape}")
