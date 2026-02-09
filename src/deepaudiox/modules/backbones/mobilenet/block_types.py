from collections.abc import Callable

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.ops.misc import ConvNormActivation

from deepaudiox.modules.backbones.mobilenet.utils import cnn_out_size, make_divisible


class ConcurrentSEBlock(torch.nn.Module):
    """
    Applies multiple Squeeze-and-Excitation (SE) operations concurrently across 
    different dimensions and aggregates the results.

    This block allows the model to attend to channel, frequency, or time dimensions 
    independently before merging the attention masks using a specified aggregation 
    operation (max, avg, add, or min).
    """
    def __init__(
        self,
        c_dim: int,
        f_dim: int,
        t_dim: int,
        se_cnf: dict
    ) -> None:
        """
        Initializes the ConcurrentSEBlock.

        Args:
            c_dim (int): Number of channels.
            f_dim (int): Frequency dimension size.
            t_dim (int): Time dimension size.
            se_cnf (Dict): Configuration dictionary containing:
                - 'se_dims': List of dimensions to apply SE on (1=C, 2=F, 3=T).
                - 'se_r': Reduction ratio for the bottleneck.
                - 'se_agg': Aggregation method ('max', 'avg', 'add', 'min').
        """
        super().__init__()
        dims = [c_dim, f_dim, t_dim]
        self.conc_se_layers = nn.ModuleList()
        for d in se_cnf['se_dims']:
            input_dim = dims[d-1]
            squeeze_dim = make_divisible(input_dim // se_cnf['se_r'], 8)
            self.conc_se_layers.append(SqueezeExcitation(input_dim, squeeze_dim, d))
        if se_cnf['se_agg'] == "max":
            self.agg_op = lambda x: torch.max(x, dim=0)[0]
        elif se_cnf['se_agg'] == "avg":
            self.agg_op = lambda x: torch.mean(x, dim=0)
        elif se_cnf['se_agg'] == "add":
            self.agg_op = lambda x: torch.sum(x, dim=0)
        elif se_cnf['se_agg'] == "min":
            self.agg_op = lambda x: torch.min(x, dim=0)[0]
        else:
            raise NotImplementedError(f"SE aggregation operation '{self.agg_op}' not implemented")

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass of the concurrent SE block.

        Args:
            input (Tensor): Input tensor of shape (B, C, F, T).

        Returns:
            Tensor: Attention-weighted tensor aggregated from multiple SE paths.
        """
        se_outs = []
        for se_layer in self.conc_se_layers:
            se_outs.append(se_layer(input))
        out = self.agg_op(torch.stack(se_outs, dim=0))
        return out


class SqueezeExcitation(torch.nn.Module):
    """
    This block implements the Squeeze-and-Excitation block from https://arxiv.org/abs/1709.01507.
    """

    def __init__(
        self,
        input_dim: int,
        squeeze_dim: int,
        se_dim: int,
        activation: Callable[..., torch.nn.Module] = torch.nn.ReLU,
        scale_activation: Callable[..., torch.nn.Module] = torch.nn.Sigmoid,
    ) -> None:
        """
        Initializes the SE block.

        Args:
            input_dim (int): Number of features in the target dimension.
            squeeze_dim (int): Size of the bottleneck (input_dim // reduction_ratio).
            se_dim (int): The dimension to preserve (1, 2, or 3). 
            activation (Callable): Non-linear activation for the bottleneck.
            scale_activation (Callable): Activation for the final attention mask.
        """
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, squeeze_dim)
        self.fc2 = torch.nn.Linear(squeeze_dim, input_dim)
        assert se_dim in [1, 2, 3]
        self.se_dim = [1, 2, 3]
        self.se_dim.remove(se_dim)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, input: Tensor) -> Tensor:
        """
        Computes the attention mask by squeezing spatial/channel information.

        Args:
            input (Tensor): Input feature map.

        Returns:
            Tensor: The computed attention weights (0 to 1).
        """
        scale = torch.mean(input, self.se_dim, keepdim=True)
        shape = scale.size()
        scale = self.fc1(scale.squeeze(2).squeeze(2))
        scale = self.activation(scale)
        scale = self.fc2(scale)
        scale = scale
        return self.scale_activation(scale).view(shape)

    def forward(self, input: Tensor) -> Tensor:
        """
        Applies the computed attention mask to the input tensor.

        Args:
            input (Tensor): Input feature map.

        Returns:
            Tensor: Element-wise scaled feature map.
        """
        scale = self._scale(input)
        return scale * input


class InvertedResidualConfig:
    """
    Configuration helper for MobileNetV3 Inverted Residual blocks.
    
    Stores architectural parameters for a single block including expansion, 
    kernel size, stride, and Squeeze-and-Excitation settings.
    """
    def __init__(
        self,
        input_channels: int,
        kernel: int,
        expanded_channels: int,
        out_channels: int,
        use_se: bool,
        activation: str,
        stride: int,
        dilation: int,
        width_mult: float,
    ):
        """
        Initializes block configuration and adjusts channels by the width multiplier.
        """
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(expanded_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.use_se = use_se
        self.use_hs = activation == "HS"
        self.stride = stride
        self.dilation = dilation
        self.f_dim = None
        self.t_dim = None

    @staticmethod
    def adjust_channels(channels: int, width_mult: float):
        """
        Scales the number of channels by width_mult and ensures divisibility by 8.

        Args:
            channels (int): Base number of channels.
            width_mult (float): Scaling factor.

        Returns:
            int: Adjusted channel count.
        """
        return make_divisible(channels * width_mult, 8)

    def out_size(self, in_size: int):
        """
        Calculates the output spatial size for this block given an input size.

        Args:
            in_size (int): Input height or width.

        Returns:
            int: Output height or width after convolution.
        """
        padding = (self.kernel - 1) // 2 * self.dilation
        return cnn_out_size(in_size, padding, self.dilation, self.kernel, self.stride)


class InvertedResidual(nn.Module):
    """
    MobileNetV3 Inverted Residual Block.
    
    Consists of:
    1. 1x1 Expansion convolution (if necessary).
    2. Depthwise convolution.
    3. Squeeze-and-Excitation (optional).
    4. 1x1 Projection convolution.
    5. Residual connection (if stride=1 and input_dims == output_dims).
    """
    def __init__(
        self,
        cnf: InvertedResidualConfig,
        se_cnf: dict,
        norm_layer: Callable[..., nn.Module],
        depthwise_norm_layer: Callable[..., nn.Module]
    ):
        """
        Initializes the Inverted Residual block.

        Args:
            cnf (InvertedResidualConfig): Structural settings for the block.
            se_cnf (Dict): Configuration for the Squeeze-Excitation layers.
            norm_layer (Callable): Normalization for expansion and projection.
            depthwise_norm_layer (Callable): Normalization for the depthwise layer.
        """
        super().__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: list[nn.Module] = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        # expand
        if cnf.expanded_channels != cnf.input_channels:
            layers.append(
                ConvNormActivation(
                    cnf.input_channels,
                    cnf.expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # depthwise
        stride = 1 if cnf.dilation > 1 else cnf.stride
        layers.append(
            ConvNormActivation(
                cnf.expanded_channels,
                cnf.expanded_channels,
                kernel_size=cnf.kernel,
                stride=stride,
                dilation=cnf.dilation,
                groups=cnf.expanded_channels,
                norm_layer=depthwise_norm_layer,
                activation_layer=activation_layer,
            )
        )
        if cnf.use_se and se_cnf['se_dims'] is not None:
            layers.append(ConcurrentSEBlock(cnf.expanded_channels, cnf.f_dim, cnf.t_dim, se_cnf))

        # project
        layers.append(
            ConvNormActivation(
                cnf.expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=None
            )
        )

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels
        self._is_cn = cnf.stride > 1

    def forward(self, inp: Tensor) -> Tensor:
        """
        Forward pass with optional residual skip connection.

        Args:
            inp (Tensor): Input feature map of shape (B, C, F, T).

        Returns:
            Tensor: Processed feature map.
        """
        result = self.block(inp)
        if self.use_res_connect:
            result += inp
        return result