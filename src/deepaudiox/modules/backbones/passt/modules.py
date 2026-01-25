from collections.abc import Callable

import torch
import torch.nn as nn

from deepaudiox.modules.backbones.passt.vit_helpers import DropPath


class Mlp(nn.Module):
    """
    Multilayer Perceptron (MLP) used in Vision Transformers.

    Consists of two linear layers with an activation and dropout applied
    between and after the layers.

    Args:
        in_features (int): Input feature dimension.
        hidden_features (int, optional): Hidden layer dimension. Defaults to in_features.
        out_features (int, optional): Output feature dimension. Defaults to in_features.
        act_layer (nn.Module, optional): Activation function. Defaults to nn.GELU.
        drop (float, optional): Dropout probability. Defaults to 0.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features=None,
        out_features=None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        Forward pass through the MLP.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch, seq_len, out_features).
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PatchEmbed(nn.Module):
    """
    2D Patch Embedding layer for spectrogram or image input.

    Converts an input image/spectrogram into non-overlapping or strided patches,
    then projects each patch to a given embedding dimension.

    Args:
        img_size (int | tuple): Input image size (height, width).
        patch_size (int | tuple): Size of each patch.
        stride (int | tuple): Stride for patch extraction. Defaults to patch_size.
        in_chans (int): Number of input channels. Defaults to 3.
        embed_dim (int): Dimension of the output embeddings.
        norm_layer (nn.Module, optional): Optional normalization layer. Defaults to None.
        flatten (bool): If True, flattens patches into sequence (B, N, C). Defaults to True.
    """

    def __init__(
        self,
        img_size: int | tuple[int, int] = 224,
        patch_size: int | tuple[int, int] = 16,
        stride: int | tuple[int, int] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: nn.Module | None = None,
        flatten: bool = True,
    ):
        super().__init__()
        img_size = self._to_2tuple(img_size)
        patch_size = self._to_2tuple(patch_size)
        stride = self._to_2tuple(stride)
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
    def _to_2tuple(self, x: int | tuple[int, int]) -> tuple[int, int]:
        """
        Converts input into a tuple of length 2.

        Args:
            x (int | tuple[int, int]): Input size.

        Returns:
            tuple[int, int]: Normalized 2-tuple.
        """
        if isinstance(x, tuple):
            return x
        return (x, x)

    def forward(self, x):
        """
        Forward pass to convert input image/spectrogram to patch embeddings.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Patch embeddings of shape (B, N, embed_dim) if flatten=True,
                          else (B, embed_dim, H_patch, W_patch).
        """
        B, C, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class Attention(nn.Module):
    """
    Multi-Head Self-Attention module.

    Args:
        dim (int): Input embedding dimension.
        num_heads (int): Number of attention heads. Defaults to 8.
        qkv_bias (bool): If True, include bias in Q/K/V projections. Defaults to False.
        attn_drop (float): Dropout probability on attention weights. Defaults to 0.
        proj_drop (float): Dropout probability on output projection. Defaults to 0.
    """

    def __init__(
        self, dim: int, num_heads: int = 8, qkv_bias: bool = False, attn_drop: float = 0.0, proj_drop: float = 0.0
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.plus1_trick = False

    def forward(self, x):
        """
        Forward pass for multi-head attention.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N, C).

        Returns:
            torch.Tensor: Output tensor of shape (B, N, C).
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.plus1_trick:
            # +1 trick
            attn = torch.cat([attn, torch.zeros(attn.shape[:-1] + (1,), dtype=attn.dtype, device=attn.device)], dim=-1)
        attn = attn.softmax(dim=-1)
        if self.plus1_trick:
            # +1 trick
            attn = attn[..., :-1]
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """
    Transformer encoder block.

    Consists of LayerNorm -> Multi-Head Attention -> Add & Norm -> MLP -> Add & Norm,
    with optional stochastic depth (DropPath).

    Args:
        dim (int): Input embedding dimension.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Hidden dimension ratio for MLP. Defaults to 4.0.
        qkv_bias (bool): If True, include bias in Q/K/V projections. Defaults to False.
        drop (float): Dropout probability. Defaults to 0.
        attn_drop (float): Dropout probability for attention. Defaults to 0.
        drop_path (float): Stochastic depth rate. Defaults to 0.
        act_layer (nn.Module): Activation function for MLP. Defaults to nn.GELU.
        norm_layer (nn.Module): Normalization layer. Defaults to nn.LayerNorm.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[[int], nn.Module] = nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        """
        Forward pass through the transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N, C).

        Returns:
            torch.Tensor: Output tensor of same shape as input.
        """
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
