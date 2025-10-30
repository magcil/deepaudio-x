# deepaudiox/modules/projection/base_projection.py

from abc import ABC, abstractmethod  
import torch
import torch.nn as nn

class BaseProjection(nn.Module, ABC):
    """Abstract base class for all projection modules."""

    in_dim: int  # every subclass must set this in __init__ - input dimension (backbone output dimension)
    out_dim: int  # every subclass must set this in __init__ - output dimension after projection
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute forward pass returning a projected tensor."""
        pass