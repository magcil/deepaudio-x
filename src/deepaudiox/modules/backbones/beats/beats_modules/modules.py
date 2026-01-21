# --------------------------------------------------------
# BEATs: Audio Pre-Training with Acoustic Tokenizers (https://arxiv.org/abs/2212.09058)
# Github source: https://github.com/microsoft/unilm/tree/master/beats
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq code bases
# https://github.com/pytorch/fairseq
# --------------------------------------------------------


import torch
import torch.nn.functional as F
from torch import nn


class GradMultiply(torch.autograd.Function):
    """Gradient scaling operator.

    Scales the gradient during the backward pass by a given factor while
    leaving the forward pass unchanged. Useful for custom gradient manipulation.
    """
    @staticmethod
    def forward(ctx, x, scale):
        """Forward pass.

        Args:
            ctx: Context object to store information for backward computation.
            x (torch.Tensor): Input tensor.
            scale (float): Scaling factor for the gradient.

        Returns:
            torch.Tensor: Same as input tensor `x`.
        """
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        """Backward pass: scales the gradient by `scale`.

        Args:
            ctx: Context object containing saved information.
            grad (torch.Tensor): Gradient of the output.

        Returns:
            tuple: Scaled gradient for input `x` and `None` for `scale`.
        """
        return grad * ctx.scale, None


class SamePad(nn.Module):
    """Applies "same" padding to a sequence tensor.

    Can be used for causal or non-causal padding in convolutional layers.
    """
    def __init__(self, kernel_size: int, causal: bool = False):
        """
        Args:
            kernel_size (int): Kernel size of the convolution.
            causal (bool): If True, apply causal padding (remove end positions).
        """
        super().__init__()
        if causal:
            self.remove = kernel_size - 1
        else:
            self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply padding adjustment.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, T).

        Returns:
            torch.Tensor: Tensor after removing extra positions for "same" padding.
        """
        if self.remove > 0:
            x = x[:, :, :-self.remove]
        return x


class Swish(nn.Module):
    """Swish activation function: x * sigmoid(x)."""
    def __init__(self):
        super().__init__()
        self.act = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Swish activation.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Activated tensor.
        """
        return x * self.act(x)


class GLU_Linear(nn.Module):
    """Linear layer with Gated Linear Unit (GLU) activation.

    Supports different GLU types including sigmoid, swish, relu, and gelu.
    """
    def __init__(self, input_dim: int, output_dim: int, glu_type: str = "sigmoid", bias_in_glu: bool = True):
        """
        Args:
            input_dim (int): Input feature dimension.
            output_dim (int): Output feature dimension.
            glu_type (str): Type of GLU activation. Options: "sigmoid", "swish", "relu", "gelu", "bilinear".
            bias_in_glu (bool): Whether to include bias in linear layer.
        """
        super().__init__()

        self.glu_type = glu_type
        self.output_dim = output_dim

        if glu_type == "sigmoid":
            self.glu_act = torch.nn.Sigmoid()
        elif glu_type == "swish":
            self.glu_act = Swish()
        elif glu_type == "relu":
            self.glu_act = torch.nn.ReLU()
        elif glu_type == "gelu":
            self.glu_act = torch.nn.GELU()

        self.linear = nn.Linear(input_dim, output_dim * 2, bias_in_glu)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through GLU linear layer.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (B, T, output_dim) after GLU activation.
        """
        x = self.linear(x)

        if self.glu_type == "bilinear":
            x = x[:, :, :self.output_dim] * x[:, :, self.output_dim: self.output_dim * 2]
        else:
            x = x[:, :, :self.output_dim] * self.glu_act(x[:, :, self.output_dim: self.output_dim * 2])

        return x


def gelu(x: torch.Tensor) -> torch.Tensor:
    """Gaussian Error Linear Unit (GELU) activation.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Activated tensor.
    """
    return torch.nn.functional.gelu(x.float()).type_as(x)


def get_activation_fn(activation: str):
    """Get activation function by name.

    Args:
        activation (str): Name of the activation function. Options: "relu", "gelu", "tanh", "linear", "glu".

    Raises:
        RuntimeError: If the activation function is not supported.

    Returns:
        callable: Activation function.
    """
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return gelu
    elif activation == "tanh":
        return torch.tanh
    elif activation in ["linear", "glu"]:
        return lambda x: x
    else:
        raise RuntimeError(f"--activation-fn {activation} not supported")


def quant_noise(module: nn.Module, p: float, block_size: int) -> nn.Module:
    """Apply Quantization Noise (QN) to Linear, Embedding, or Conv2d modules.

    This simulates stochastic block-wise dropping of weights, useful for
    training models with iterative product quantization (iPQ).

    Args:
        module (nn.Module): Module to wrap (Linear, Embedding, or Conv2d).
        p (float): Probability of dropping a block (0 disables QN).
        block_size (int): Size of the blocks for quantization.

    Raises:
        ValueError: If module type is unsupported or weight dimensions are incompatible with `block_size`.

    Returns:
        nn.Module: The same module with a forward pre-hook applied for quantization noise.
    """
    if p <= 0:
        return module

    if not isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d)):
        raise ValueError(
            f"Expected module to be one of (nn.Linear, nn.Embedding, nn.Conv2d), "
            f"but got {type(module)}"
        )

    is_conv = module.weight.ndim == 4

    # Check input dimensions are compatible with block_size
    if not is_conv:
        if module.weight.size(1) % block_size != 0:
            raise ValueError(
                f"Input features ({module.weight.size(1)}) must be a multiple of block size ({block_size})"
            )
    else:
        if module.kernel_size == (1, 1):
            if module.in_channels % block_size != 0:
                raise ValueError(
                    f"Input channels ({module.in_channels}) must be a multiple of block size ({block_size})"
                )
        else:
            k = module.kernel_size[0] * module.kernel_size[1]
            if k % block_size != 0:
                raise ValueError(f"Kernel size ({k}) must be a multiple of block size ({block_size})")

    def _forward_pre_hook(mod: nn.Module, input):
        """Forward pre-hook to apply quantization noise to weights.

        Args:
            mod (nn.Module): Module being executed.
            input: Module input (unused here).
        """
        if not mod.training:
            return

        weight = mod.weight
        if not is_conv:
            in_features = weight.size(1)
            out_features = weight.size(0)
            mask = torch.zeros(in_features // block_size * out_features, device=weight.device)
            mask.bernoulli_(p)
            mask = mask.repeat_interleave(block_size, -1).view(-1, in_features)
        else:
            in_channels = mod.in_channels
            out_channels = mod.out_channels
            if mod.kernel_size == (1, 1):
                mask = torch.zeros(int(in_channels // block_size * out_channels), device=weight.device)
                mask.bernoulli_(p)
                mask = mask.repeat_interleave(block_size, -1).view(-1, in_channels)
            else:
                mask = torch.zeros(weight.size(0), weight.size(1), device=weight.device)
                mask.bernoulli_(p)
                mask = mask.unsqueeze(2).unsqueeze(3).repeat(1, 1, mod.kernel_size[0], mod.kernel_size[1])

        mask = mask.to(torch.bool)
        s = 1 / (1 - p)
        mod.weight.data = s * weight.masked_fill(mask, 0)

    module.register_forward_pre_hook(_forward_pre_hook)
    return module
