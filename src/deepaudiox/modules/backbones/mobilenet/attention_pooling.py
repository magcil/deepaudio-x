import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from deepaudiox.modules.backbones.mobilenet.utils import collapse_dim


class MultiHeadAttentionPooling(nn.Module):
    """
    Multi-Head Attention Pooling layer as proposed in the PSLA paper.

    Reference: https://arxiv.org/pdf/2102.01243.pdf

    This module performs weighted temporal pooling of feature maps. It projects
    the input into multiple subspaces, calculating an attention score (att) and
    a value (val) for each time step across multiple heads. The results are
    aggregated to form a single global feature vector.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        att_activation: str = "sigmoid",
        clf_activation: str = "ident",
        num_heads: int = 4,
        epsilon: float = 1e-7,
    ):
        """
        Initializes the MultiHeadAttentionPooling module.

        Args:
            in_dim (int): Number of input channels/features per time step.
            out_dim (int): Size of the output feature dimension for each head.
            att_activation (str): Activation function for the attention path.
                Commonly 'sigmoid' or 'softmax'.
            clf_activation (str): Activation function for the classification/value path.
            num_heads (int): Number of independent attention heads.
            epsilon (float): Small constant to prevent division by zero or log(0).
        """
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.epsilon = epsilon

        self.att_activation = att_activation
        self.clf_activation = clf_activation

        # out size: out dim x 2 (att and clf paths) x num_heads
        self.subspace_proj = nn.Linear(self.in_dim, self.out_dim * 2 * self.num_heads)
        self.head_weight = nn.Parameter(torch.tensor([1.0 / self.num_heads] * self.num_heads).view(1, -1, 1))

    def activate(self, x: torch.Tensor, activation: str):
        """
        Applies the specified activation function to a tensor.

        Args:
            x (Tensor): The input tensor to activate.
            activation (str): The name of the activation function
                ('linear', 'relu', 'sigmoid', 'softmax', 'ident').

        Returns:
            Tensor: The activated tensor.
        """
        if activation == "linear":
            return x
        elif activation == "relu":
            return F.relu(x)
        elif activation == "sigmoid":
            return torch.sigmoid(x)
        elif activation == "softmax":
            return F.softmax(x, dim=1)
        elif activation == "ident":
            return x

    def forward(self, x: torch.Tensor) -> Tensor:
        """
        Forward pass for Multi-Head Attention Pooling.

        Args:
            x (Tensor): Input feature map of shape
                (batch_size, channels, frequency_bands, sequence_length).

        Returns:
            Tensor: Globally pooled output of shape (batch_size, out_dim).
        """
        x = collapse_dim(x, dim=2)
        x = x.transpose(1, 2)
        b, n, c = x.shape

        x = self.subspace_proj(x).reshape(b, n, 2, self.num_heads, self.out_dim).permute(2, 0, 3, 1, 4)
        att, val = x[0], x[1]
        val = self.activate(val, self.clf_activation)
        att = self.activate(att, self.att_activation)
        att = torch.clamp(att, self.epsilon, 1.0 - self.epsilon)
        att = att / torch.sum(att, dim=2, keepdim=True)

        out = torch.sum(att * val, dim=2) * self.head_weight
        out = torch.sum(out, dim=1)

        return out
