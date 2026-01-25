# deepaudiox/modules/pooling/ep.py

# This file is derived from:
# https://github.com/billpsomas/efficient-probing/blob/master/poolings/ep.py

# Copyright 2022 billpsomas
# Copyright 2026 magcil
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications:
# - Adapted for audio classification pipelines in DeepAudioX.
# - Changed class and method names to fit DeepAudioX conventions.
# - Updated type hints and added docstrings for clarity.


import torch
from torch import nn

from deepaudiox.modules.baseclasses import BasePooling


class EfficientProbing(BasePooling):
    """
    Efficient Probing (EP) pooling as described in "ATTENTION, PLEASE! REVISITING ATTENTIVE
    PROBING THROUGH THE LENS OF EFFICIENCY"

    Atributes:
        dim (int): Input dimension.
        num_heads (int): Number of attention heads.
        qkv_bias (bool): Whether to include bias terms in the query and key linear layers.
        qk_scale (float | None): Scaling factor for the query-key dot product.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        num_queries: int = 32,
        d_out: int = 1,
    ):
        """
        Initialize the EfficientProbing module.

        Args:
            in_dim (int): Input dimension.
            num_heads (int): Number of attention heads.
            qkv_bias (bool): Whether to include bias terms in the query and key linear layers.
            qk_scale (float | None): Scaling factor for the query-key dot product.
        """
        super().__init__(in_dim=dim)

        assert dim % num_queries == 0, "Input dimension must be divisible by num_queries"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.d_out = d_out
        self.num_queries = num_queries

        self.v = nn.Linear(dim, dim // d_out, bias=qkv_bias)
        self.cls_token = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 4:  # CNN feature map (B, D, H, W)
            B, N, H, W = x.shape
            C = H * W
            x = x.permute(0, 2, 3, 1).reshape(B, N, C)  # (B, N, C) where C=H*W
        elif len(x.shape) == 3:  # Transformer feature map (B, N, C)
            B, N, C = x.shape
        else:
            raise ValueError("Input tensor must be of shape (B, D, H, W) or (B, N, C)")

        C_prime = C // self.d_out

        cls_token = self.cls_token.expand(B, -1, -1)  # newly created class token

        # q: (B, num_queries, num_heads, head_dim) -> (B, num_heads, num_queries, head_dim)
        q = cls_token.reshape(B, self.num_queries, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # k: (B, N, num_heads, head_dim) -> (B, num_heads, N, head_dim)
        k = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale

        # self.v(x): (B, N, C // d_out) -> (B, N, num_queries, dv) -> (B, num_queries, N, dv)
        v = self.v(x).reshape(B, N, self.num_queries, C // (self.d_out * self.num_queries)).permute(0, 2, 1, 3)

        attn = q @ k.transpose(-2, -1)  # (B, num_heads, num_queries, N)
        attn = attn.softmax(dim=-1)

        x_cls = torch.matmul(attn.squeeze(1).unsqueeze(2), v)  # (B, num_heads, 1, num_queries, dv))
        x_cls = x_cls.view(B, C_prime)

        return x_cls
