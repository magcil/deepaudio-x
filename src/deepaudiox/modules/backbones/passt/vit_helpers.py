"""
Adapted from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
Credit to @leo19941227  for remove timm dependencies here : https://github.com/s3prl/passt_hear21/blob/48a0dc1b824641ca59884ced53f5b86053fed141/hear21passt/models/helpers/vit_helpers.py

"""

import math

import torch
from torch import nn
from torch.nn.init import _calculate_fan_in_and_fan_out


def init_vit_weights(module: nn.Module, name: str = "", head_bias: float = 0.0, jax_impl: bool = False):
    """
    Initialize weights for Vision Transformer (ViT) modules.

    This function handles initialization for Linear layers, Conv2d layers,
    and normalization layers according to the original ViT / DeiT schemes.
    It optionally matches the JAX implementation of ViT when `jax_impl=True`.

    Args:
        module (nn.Module): Module to initialize (Linear, Conv2d, or Norm layer).
        name (str, optional): Module name used to differentiate heads, pre_logits, etc.
        head_bias (float, optional): Bias for classifier heads. Defaults to 0.
        jax_impl (bool, optional): Whether to use initialization compatible with JAX ViT. Defaults to False.
    """

    if isinstance(module, nn.Linear):
        if name.startswith("head"):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith("pre_logits"):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if "mlp" in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


def adapt_input_conv(in_chans: int, conv_weight: torch.Tensor) -> torch.Tensor:
    """
    Adapt a pretrained Conv2d weight tensor to a different number of input channels.

    Supports:
        - RGB → Grayscale (sum channels)
        - Space2Depth stems
        - Arbitrary number of channels via repetition

    Args:
        in_chans (int): Number of input channels for the new model.
        conv_weight (torch.Tensor): Pretrained Conv2d weight tensor
                                    of shape (out_channels, in_channels, kernel_h, kernel_w)

    Returns:
        torch.Tensor: Adapted Conv2d weight tensor.

    Raises:
        ValueError: If in_channels not divisible by 3 for Space2Depth.
        NotImplementedError: If unsupported input channel configuration is provided.
    """
    original_dtype = conv_weight.dtype

    # Ensure float32 for safe summation on CPU
    conv_weight = conv_weight.float()

    out_channels, in_channels, kernel_height, kernel_width = conv_weight.shape

    if in_chans == 1:
        # Convert RGB (or space2depth) weights to single-channel
        if in_channels > 3:
            # space2depth stem (channels divisible by 3)
            if in_channels % 3 != 0:
                raise ValueError(f"in_channels={in_channels} must be divisible by 3")
            conv_weight = conv_weight.reshape(
                out_channels,
                in_channels // 3,
                3,
                kernel_height,
                kernel_width,
            )
            conv_weight = conv_weight.sum(dim=2)
        else:
            # Standard RGB → grayscale
            conv_weight = conv_weight.sum(dim=1, keepdim=True)

    elif in_chans != 3:
        # Adapt RGB weights to arbitrary number of input channels
        if in_channels != 3:
            raise NotImplementedError("Weight format not supported for non-RGB input.")

        repeat = int(math.ceil(in_chans / 3))
        conv_weight = conv_weight.repeat(1, repeat, 1, 1)[:, :in_chans, :, :]
        conv_weight *= 3.0 / float(in_chans)

    # Restore original dtype (e.g. float16)
    conv_weight = conv_weight.to(original_dtype)

    return conv_weight


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """
    Apply stochastic depth (drop path) regularization.

    Randomly drops entire paths (residual blocks) during training to improve generalization.

    Args:
        x (torch.Tensor): Input tensor.
        drop_prob (float, optional): Probability of dropping a path. Defaults to 0.
        training (bool, optional): Whether the model is in training mode. Defaults to False.

    Returns:
        torch.Tensor: Tensor with dropped paths applied.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (stochastic depth) layer for residual networks.

    This module wraps the `drop_path` function to be used as a standard nn.Module.

    Args:
        drop_prob (float, optional): Probability of dropping paths. Defaults to None.

    Forward Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Tensor with stochastic depth applied.
    """

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    """
    Fill a tensor with values from a truncated normal distribution in-place.
    Values are clipped to [a, b].

    Args:
        tensor (torch.Tensor): Tensor to fill.
        mean (float): Mean of the normal distribution.
        std (float): Standard deviation.
        a (float): Minimum cutoff value.
        b (float): Maximum cutoff value.

    Returns:
        torch.Tensor: The same tensor filled in-place.
    """

    def norm_cdf(x):
        """
        Standard normal cumulative distribution function.

        Args:
            x (float or Tensor): Input value(s).

        Returns:
            float or Tensor: CDF evaluated at x.
        """
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        lower = norm_cdf((a - mean) / std)
        upper = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * lower - 1, 2 * upper - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """
    Fill tensor with values drawn from a truncated normal distribution.

    Values outside [a, b] are redrawn until they lie within bounds.

    Args:
        tensor (torch.Tensor): Tensor to initialize.
        mean (float, optional): Distribution mean. Defaults to 0.0.
        std (float, optional): Distribution standard deviation. Defaults to 1.0.
        a (float, optional): Minimum cutoff. Defaults to -2.0.
        b (float, optional): Maximum cutoff. Defaults to 2.0.

    Returns:
        torch.Tensor: The same tensor initialized in-place.
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode="fan_in", distribution="normal"):
    """
    Initialize a tensor with variance scaling.

    Supports normal, truncated normal, and uniform distributions.

    Args:
        tensor (torch.Tensor): Tensor to initialize.
        scale (float, optional): Scaling factor. Defaults to 1.0.
        mode (str, optional): One of ['fan_in', 'fan_out', 'fan_avg']. Defaults to 'fan_in'.
        distribution (str, optional): Distribution type. One of ['normal', 'truncated_normal', 'uniform'].
                                        Defaults to 'normal'.

    Raises:
        ValueError: If `distribution` is invalid.
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == "fan_in":
        denom = fan_in
    elif mode == "fan_out":
        denom = fan_out
    elif mode == "fan_avg":
        denom = (fan_in + fan_out) / 2

    variance = scale / denom

    if distribution == "truncated_normal":
        # constant is stddev of standard normal truncated to (-2, 2)
        trunc_normal_(tensor, std=math.sqrt(variance) / 0.87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    """
    Initialize a tensor with LeCun normal initialization (truncated normal).

    Args:
        tensor (torch.Tensor): Tensor to initialize.
    """
    variance_scaling_(tensor, mode="fan_in", distribution="truncated_normal")
