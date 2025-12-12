from typing import Literal

import torch
import torch.nn as nn


class MLPHead(nn.Module):
    """Audio classification head for downstream tasks.

    Attributes:
        model (nn.Module): The audio classifier head.
    """

    def __init__(
        self,
        num_classes: int,
        in_dim: int,
        hidden_layers: list[int] | None = None,
        activation: Literal["relu", "gelu", "tanh", "leakyrelu"] = "relu",
        apply_batch_norm: bool = False,
    ):
        """Initializes the MLP head for classification.

        Args:
            num_classes (int): The number of input classes.
            in_dim (int): The embedding dimension of the backbone model.
            hidden_layers (list[int]): A sequence of integers for indicating the classifier layers.
            activation (Literal["relu", "gelu", "tanh", "leakyrelu"]): The activation function between layers.
            apply_batch_norm (bool): Wheter to use batch norm or not.

        Example:
            >>> from deepaudiox.modules.classifier.classifier import MLPHead
            >>> head = MLPHead(num_classes=10, in_dim=768, hidden_layers=[768, 256, 32]) # 2 hidden layers
        """
        super().__init__()

        if hidden_layers is None:
            hidden_layers = []

        layers = []
        input_dim = in_dim

        activation_fn = {"relu": nn.ReLU(), "gelu": nn.GELU(), "tanh": nn.Tanh(), "leakyrelu": nn.LeakyReLU()}.get(
            activation.lower(), nn.ReLU()
        )

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim, bias=not apply_batch_norm))
            if apply_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(activation_fn)
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, num_classes))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method

        Args:
            x (torch.Tensor): The input one-dimensional embedding.

        Returns:
            torch.Tensor: The output logits of shape (B, num_classes)
        """

        return self.model(x)
