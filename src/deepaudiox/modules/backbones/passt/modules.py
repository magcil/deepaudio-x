import collections
import warnings
from itertools import repeat

import torch
import torch.nn as nn

from deepaudiox.modules.backbones.passt.vit_helpers import DropPath


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
        self, 
        in_features: int, 
        hidden_features = None, 
        out_features = None, 
        act_layer  =nn.GELU, 
        drop: float = 0.
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
        self, 
        img_size: int = 224, 
        patch_size: int = 16, 
        stride: int =16, 
        in_chans: int = 3, 
        embed_dim: int = 768, 
        norm_layer: nn.Module | None = None,
        flatten: bool = True
    ):
        super().__init__()
        to_2tuple =  self._ntuple(2)
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        self.img_size = img_size
        self.patch_size = patch_size
        self.stride = stride
        self.grid_size = (img_size[0] // stride[0], img_size[1] // stride[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    # From PyTorch internals
    def _ntuple(self, n):
        def parse(x):
            if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
                return tuple(x)
            return tuple(repeat(x, n))
        return parse

    def forward(self, x):
        B, C, H, W = x.shape
        if not (self.img_size[0] == H and self.img_size[1] == W):
            warnings.warn(
                f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}).",
                stacklevel=2
            )
        # to do maybe replace weights
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

class Attention(nn.Module):
    def __init__(
        self, 
        dim: int, 
        num_heads: int = 8, 
        qkv_bias: bool = False, 
        attn_drop: float = 0., 
        proj_drop: float = 0.
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.plus1_trick = False
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.plus1_trick:
            # +1 trick
            attn = torch.cat([attn, torch.zeros(attn.shape[:-1]+(1,), dtype=attn.dtype, device=attn.device)], dim=-1)
        attn = attn.softmax(dim=-1)
        if self.plus1_trick:
            # +1 trick
            attn = attn[...,:-1]
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(
        self, 
        dim: int, 
        num_heads: int, 
        mlp_ratio: float = 4., 
        qkv_bias: bool =False, 
        drop: float = 0., 
        attn_drop: float = 0.,
        drop_path: float = 0., 
        act_layer: nn.Module = nn.GELU, 
        norm_layer: nn.Module = nn.LayerNorm
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x