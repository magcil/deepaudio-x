from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseAudioClassifier(nn.Module, ABC):
    """Base class for creating custom audio classifiers."""

    @abstractmethod
    def __init__(self, *args, **kwargs):
        """Initialize the audio classifier."""
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor):
        """Pass the input through the model and return logits.

        Args:
            x (torch.Tensor): The input tensor.
        """
        raise NotImplementedError
    
    def predict(self, x: torch.Tensor):
        """Compute predicted class and posterior probabilities.
        
        Args:
            x (torch.Tensor): The input tensor.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        logits = self.forward(x)
        posteriors = F.softmax(logits.cpu(), dim=1)
        max_posteriors = posteriors.max(dim=1)

        return {
            "posterior_probs": max_posteriors.values.tolist(),
            "y_preds": max_posteriors.indices.tolist()
        }

    def extract_feature(self, x: torch.Tensor):
        """Pass the input through the model and return feature.
        
        Args:
            x (torch.Tensor): The input tensor.
        """
        return self.forward(x)
