from typing import Literal

import torch.nn as nn


class MLPHead(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_dim: int,
        hidden_layers: list[int] | None = None,
        activation: Literal["relu", "gelu", "tanh", "leakyrelu"] = "relu",
        apply_batch_norm: bool = False,
    ):
        """Audio classification head for downstream tasks.

        Attributes:
            in_dim (int): Input feature dimension.
            num_classes (int): Number of output classes.
            hidden_layers (list[int] or None): List of hidden layer sizes. If None, just a single linear layer.
            activation (str): Activation function name ("relu", "gelu", "tanh" or "leakyrelu").
            apply_batch_norm (bool): Whether to use BatchNorm1d after each Linear layer.

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

    def forward(self, x):
        return self.model(x)
