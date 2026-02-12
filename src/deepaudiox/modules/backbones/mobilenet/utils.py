import math
from collections.abc import Callable

import torch
import torch.nn as nn
from torch import Tensor


def NAME_TO_WIDTH(name: str):
    """
    Maps a model name string to its corresponding width multiplier.

    Args:
        name (str): The model identifier string (e.g., 'mn04', 'dymn10').

    Returns:
        float: The width multiplier used to scale the number of channels.
               Defaults to 1.0 if the name is not found.
    """
    mn_map = {
        "mn01": 0.1,
        "mn02": 0.2,
        "mn04": 0.4,
        "mn05": 0.5,
        "mn06": 0.6,
        "mn08": 0.8,
        "mn10": 1.0,
        "mn12": 1.2,
        "mn14": 1.4,
        "mn16": 1.6,
        "mn20": 2.0,
        "mn30": 3.0,
        "mn40": 4.0,
    }

    dymn_map = {"dymn04": 0.4, "dymn10": 1.0, "dymn20": 2.0}

    try:
        w = dymn_map[name[:6]] if name.startswith("dymn") else mn_map[name[:4]]
    except Exception:
        w = 1.0

    return w


def make_divisible(v: float, divisor: int, min_value: int | None = None) -> int:
    """
    Ensures that all layers have a channel number that is divisible by the divisor.

    This function is taken from the original TensorFlow MobileNet implementation.
    It rounds the channel count to the nearest multiple of the divisor while
    ensuring the result does not drop below 90% of the original value.

    Args:
        v (float): The original/calculated channel count.
        divisor (int): The value the output must be divisible by (e.g., 8).
        min_value (Optional[int]): Minimum possible value for the output.
            Defaults to the divisor value if None.

    Returns:
        int: The nearest divisible channel count.
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def cnn_out_size(in_size: int, padding: int, dilation: int, kernel: int, stride: int):
    """
    Calculates the output spatial dimension (width or height) of a CNN layer.

    The calculation follows the standard formula:
    $$O = \lfloor \frac{I + 2P - D(K-1) - 1}{S} + 1 \rfloor$$

    Args:
        in_size (int): Input spatial dimension.
        padding (int): Padding size applied to the input.
        dilation (int): Spacing between kernel elements.
        kernel (int): Size of the kernel/filter.
        stride (int): Stride of the convolution.

    Returns:
        int: The resulting output spatial dimension.
    """
    s = in_size + 2 * padding - dilation * (kernel - 1) - 1
    return math.floor(s / stride + 1)


def collapse_dim(
    x: Tensor,
    dim: int,
    mode: str = "pool",
    pool_fn: Callable[[Tensor, int], Tensor] = torch.mean,
    combine_dim: int | None = None,
) -> Tensor:
    """
    Collapses a specific dimension of a multi-dimensional tensor by pooling or reshaping.

    Args:
        x (Tensor): The input tensor.
        dim (int): The index of the dimension to collapse.
        mode (str): The method of collapsing. Options:
            - 'pool': Applies the `pool_fn` across the dimension (reduces rank).
            - 'combine': Multiplies the size of `combine_dim` by the size of `dim` (maintains rank).
        pool_fn (Callable): The reduction function to use if mode is 'pool' (e.g., torch.mean, torch.max).
        combine_dim (int, optional): The target dimension to merge `dim` into if mode is 'combine'.

    Returns:
        Tensor: The tensor with the specified dimension collapsed.
    """
    if mode == "pool":
        return pool_fn(x, dim)
    elif mode == "combine":
        if combine_dim is None:
            raise ValueError("combine_dim must be provided when mode='combine'")
        s = list(x.size())
        s[combine_dim] *= dim
        s[dim] //= dim
        return x.view(s)
    raise ValueError(f"Unsupported collapse mode: {mode}")


class CollapseDim(nn.Module):
    """
    A PyTorch Module wrapper for the collapse_dim function.

    Attributes:
        dim (int): Dimension to collapse.
        mode (str): 'pool' or 'combine'.
        pool_fn (Callable): The function to apply if pooling.
        combine_dim (int): The dimension to merge into if combining.
    """

    def __init__(
        self,
        dim: int,
        mode: str = "pool",
        pool_fn: Callable[[Tensor, int], Tensor] = torch.mean,
        combine_dim: int | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.mode = mode
        self.pool_fn = pool_fn
        self.combine_dim = combine_dim

    def forward(self, x: torch.Tensor) -> Tensor:
        """Applies collapse_dim to the input tensor."""
        return collapse_dim(x, dim=self.dim, mode=self.mode, pool_fn=self.pool_fn, combine_dim=self.combine_dim)
