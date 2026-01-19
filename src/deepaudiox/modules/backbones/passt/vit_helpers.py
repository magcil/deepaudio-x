"""
Adapted from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
Credit to @leo19941227  for remove timm dependencies here : https://github.com/s3prl/passt_hear21/blob/48a0dc1b824641ca59884ced53f5b86053fed141/hear21passt/models/helpers/vit_helpers.py

"""
import logging
import math
import warnings

import torch
from torch import nn

try:
    from timm.models._hub import download_cached_file
except ModuleNotFoundError:
    from timm.models.hub import download_cached_file
from torch.nn.init import _calculate_fan_in_and_fan_out

# Global variables for rarely used pretrained checkpoint download progress and hash check.
# Use set_pretrained_download_progress / set_pretrained_check_hash functions to toggle.
_DOWNLOAD_PROGRESS = True
_CHECK_HASH = False


_logger = logging.getLogger(__name__)


def init_vit_weights(
    module: nn.Module, 
    name: str = '', 
    head_bias: float = 0., 
    jax_impl: bool = False
):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
    as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
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
    Adapt pretrained Conv2d weights to a different number of input channels.

    Args:
        in_chans: Number of input channels for the new model.
        conv_weight: Conv2d weight tensor of shape
                     (out_channels, in_channels, kernel_height, kernel_width)

    Returns:
        Adapted convolution weight tensor.
    """
    original_dtype = conv_weight.dtype

    # Ensure float32 for safe summation on CPU
    conv_weight = conv_weight.float()

    out_channels, in_channels, kernel_height, kernel_width = conv_weight.shape

    if in_chans == 1:
        # Convert RGB (or space2depth) weights to single-channel
        if in_channels > 3:
            # space2depth stem (channels divisible by 3)
            assert in_channels % 3 == 0
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
            raise NotImplementedError(
                "Weight format not supported for non-RGB input."
            )

        repeat = int(math.ceil(in_chans / 3))
        conv_weight = conv_weight.repeat(1, repeat, 1, 1)[:, :in_chans, :, :]
        conv_weight *= 3.0 / float(in_chans)

    # Restore original dtype (e.g. float16)
    conv_weight = conv_weight.to(original_dtype)

    return conv_weight


def load_pretrained_weights(
    model: nn.Module,
    first_conv: str | None = None,
    pretrained_url: str | None = None,
    filter_fn=None,
    num_classes: int = 1000,
    in_chans: int =3,
    strict: bool = False,
    progress: bool = False,
    classifiers: tuple | None = None,
    label_offset: int = 0
):
    """Load pretrained checkpoint

    Args:
        model (nn.Module) : PyTorch model module
        default_cfg (Optional[Dict]): default configuration for pretrained weights / target dataset
        num_classes (int): num_classes for model
        in_chans (int): in_chans for model
        filter_fn (Optional[Callable]): state_dict filter fn for load (takes state_dict, model as args)
        strict (bool): strict load of checkpoint
        progress (bool): enable progress bar for weight download

    """
    if not pretrained_url:
        _logger.warning(
            "No pretrained weights exist for this model. Using random initialization."
        )
        return

    _logger.info(f"Loading pretrained weights from url ({pretrained_url})")
    
    pretrained_loc = download_cached_file(
        pretrained_url,
        check_hash=_CHECK_HASH,
        progress=_DOWNLOAD_PROGRESS,
    )

    state_dict = torch.load(pretrained_loc, map_location="cpu")

    if filter_fn is not None:
        # for backwards compat with filter fn that take one arg, try one first, the two
        try:
            state_dict = filter_fn(state_dict)
        except TypeError:
            state_dict = filter_fn(state_dict, model)

    input_convs = first_conv
    if input_convs is not None and in_chans != 3:
        if isinstance(input_convs, str):
            input_convs = (input_convs,)
        for input_conv_name in input_convs:
            weight_name = input_conv_name + ".weight"
            try:
                state_dict[weight_name] = adapt_input_conv(
                    in_chans, state_dict[weight_name]
                )
                _logger.info(
                    f"Converted input conv {input_conv_name} pretrained weights from 3 to {in_chans} channel(s)"
                )
            except NotImplementedError:
                del state_dict[weight_name]
                strict = False
                _logger.warning(
                    f"Unable to convert pretrained {input_conv_name} weights, using random init for this layer."
                )

    if classifiers is not None:
        if isinstance(classifiers, str):
            classifiers = (classifiers,)

        # Always drop classifier heads for backbone usage
        for classifier_name in classifiers:
            for suffix in ("weight", "bias"):
                key = f"{classifier_name}.{suffix}"
                if key in state_dict:
                    del state_dict[key]

        strict = False

    model.load_state_dict(state_dict, strict=strict)


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

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
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode="fan_in", distribution="normal"):
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
    variance_scaling_(tensor, mode="fan_in", distribution="truncated_normal")