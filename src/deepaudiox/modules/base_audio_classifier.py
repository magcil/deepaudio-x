from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseAudioClassifier(nn.Module, ABC):
    """Base class for creating custom audio classifiers.

    This class defines the standard interface for audio classification models.
    Subclasses must implement the core initialization and forward pass. The
    built-in `predict` method provides a convenience wrapper to obtain predicted
    labels, posterior probabilities, and raw logits.

    Methods:
        __init__: Initialize the classifier and its components.
        forward: Process input waveforms and return logits.
        predict: Compute predicted classes, posterior probabilities, and logits.
    """

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

    def predict(self, x: torch.Tensor) -> dict[str, np.ndarray]:
        """Compute predicted class and posterior probabilities.

        Args:
            x (torch.Tensor): Input Waveforms of shape B x N*SR, B Batch size, N length, SR sample rate

        Returns:
            dict[str, np.ndarray]: y_preds, posteriors, logits.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        logits = self.forward(x)
        posteriors = F.softmax(logits, dim=1)
        max_posteriors = posteriors.max(dim=1)

        return {
            "y_preds": max_posteriors.indices.numpy(force=True),
            "posteriors": max_posteriors.values.numpy(force=True),
            "logits": logits.numpy(force=True),
        }
