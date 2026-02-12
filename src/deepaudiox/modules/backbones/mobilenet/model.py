from collections.abc import Callable, Sequence
from functools import partial

import torch
from torch import Tensor, nn
from torchvision.ops.misc import Conv2dNormActivation

from deepaudiox.modules.backbones.mobilenet.block_types import InvertedResidual, InvertedResidualConfig
from deepaudiox.modules.backbones.mobilenet.preprocess import AugmentMelSTFT
from deepaudiox.modules.backbones.mobilenet.utils import cnn_out_size
from deepaudiox.modules.baseclasses import BaseBackbone


class MobileNetConfig:
    """
    Configuration class for MobileNetV3-based architectures.

    This class manages hyperparameters, architectural settings, and the generation
    of inverted residual block configurations. It supports dynamic updates via
    a configuration dictionary.
    """

    def __init__(self, cfg=None):
        """
        Initializes the MobileNetConfig with default or custom parameters.

        Args:
            cfg (dict, optional): A dictionary containing configuration keys to
                override defaults. If provided, the model settings are
                re-calculated after the update.
        """
        self.block: Callable[..., nn.Module] | None = None
        self.norm_layer: Callable[..., nn.Module] | None = None
        self.num_classes: int = 527
        self.in_conv_kernel: int = 3
        self.in_conv_stride: int = 2
        self.in_channels: int = 1
        self.width_mult: float = 1.0
        self.reduced_tail: bool = False
        self.dilated: bool = False
        self.strides: tuple[int, int, int, int] = (2, 2, 2, 2)
        self.multihead_attention_heads: int = 4
        self.f_dim: int = 128
        self.t_dim: int = 1000
        self.se_conf: dict = dict(se_dims=[1], se_agg="max", se_r=4)

        self.inverted_residual_setting, self.last_channel = self._configure_mobilenet(
            width_mult=self.width_mult, reduced_tail=self.reduced_tail, dilated=self.dilated, strides=self.strides
        )

        if cfg is not None:
            self.update(cfg)
            self.inverted_residual_setting, self.last_channel = self._configure_mobilenet(
                width_mult=self.width_mult, reduced_tail=self.reduced_tail, dilated=self.dilated, strides=self.strides
            )

    def _configure_mobilenet(
        self,
        width_mult: float = 1,
        reduced_tail: bool = False,
        dilated: bool = False,
        strides: tuple[int, int, int, int] = (2, 2, 2, 2),
    ):
        """
        Defines the structural layout of the MobileNetV3 backbone.

        Calculates the input/output channels, kernel sizes, and strides for
        each Inverted Residual block based on the width multiplier and
        architectural flags.



        Args:
            width_mult (float): Scaling factor for the number of channels in each layer.
            reduced_tail (bool): If True, reduces the number of channels in the
                final layers to decrease model size.
            dilated (bool): If True, applies a dilation of 2 to the final blocks
                to maintain spatial resolution.
            strides (Tuple[int, ...]): A sequence of strides for the 4 main
                downsampling stages of the network.

        Returns:
            Tuple[list, int]: A tuple containing:
                - A list of InvertedResidualConfig objects defining each block.
                - The number of channels in the final convolution layer.
        """

        reduce_divider = 2 if reduced_tail else 1
        dilation = 2 if dilated else 1

        bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)
        adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_mult=width_mult)

        inverted_residual_setting = [
            bneck_conf(16, 3, 16, 16, False, "RE", 1, 1),
            bneck_conf(16, 3, 64, 24, False, "RE", strides[0], 1),
            bneck_conf(24, 3, 72, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 72, 40, True, "RE", strides[1], 1),
            bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
            bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
            bneck_conf(40, 3, 240, 80, False, "HS", strides[2], 1),
            bneck_conf(80, 3, 200, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 480, 112, True, "HS", 1, 1),
            bneck_conf(112, 3, 672, 112, True, "HS", 1, 1),
            bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", strides[3], dilation),
            bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation),
            bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation),
        ]
        last_channel = adjust_channels(1280 // reduce_divider)

        return inverted_residual_setting, last_channel

    def update(self, cfg: dict):
        """
        Updates the configuration attributes with values from a dictionary.

        Args:
            cfg (dict): A dictionary where keys match class attribute names.
        """
        self.__dict__.update(cfg)


class MobileNet(BaseBackbone):
    """
    MobileNet V3 implementation tailored for audio classification and feature extraction.

    This class constructs a deep convolutional neural network using Inverted Residual blocks.
    It is optimized for processing 2D spectrogram inputs, typically generated from raw
    audio waveforms via an internal feature extractor.
    """

    def __init__(
        self, out_dim: int | None = None, cfg: MobileNetConfig | None = None, sample_rate: int = 32_000
    ) -> None:
        """
        Initializes the MobileNet V3 backbone.

        Args:
            out_dim (int | None, optional): Specific output dimension for the final layer.
                If None, defaults to the last convolution output channels.
            cfg (MobileNetConfig | None, optional): Configuration object containing
                model hyperparameters. Defaults to a standard MobileNetConfig.
            sample_rate (int, optional): The audio sample rate for the input waveforms.
                Defaults to 32,000.

        Raises:
            ValueError: If `inverted_residual_setting` in the config is empty.
            TypeError: If `inverted_residual_setting` is not a list of
                InvertedResidualConfig objects.
        """
        if cfg is None:
            cfg = MobileNetConfig()

        # Retieve values from config object
        in_channels = cfg.in_channels
        in_conv_stride = cfg.in_conv_stride
        in_conv_kernel = cfg.in_conv_kernel
        block = getattr(cfg, "block", None) or InvertedResidual
        # norm_layer = getattr(cfg, "norm_layer", partial(nn.BatchNorm2d, eps=0.001, momentum=0.01))
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
        se_cnf = getattr(cfg, "se_conf", None)
        inverted_residual_setting = cfg.inverted_residual_setting
        f_dim = getattr(cfg, "f_dim", 128)
        t_dim = getattr(cfg, "t_dim", 1000)

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
            isinstance(inverted_residual_setting, Sequence)
            and all(isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting)
        ):
            raise TypeError("The inverted_residual_setting should be List[InvertedResidualConfig]")
        if se_cnf is None:
            raise ValueError("cfg.se_conf should not be None")
        if not isinstance(se_cnf, dict):
            raise TypeError("cfg.se_conf should be a dict")

        resolved_out_dim = out_dim if out_dim is not None else 6 * inverted_residual_setting[-1].out_channels
        super().__init__(sample_rate=sample_rate, out_dim=resolved_out_dim)

        depthwise_norm_layer = norm_layer

        # Build first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels

        layers = []
        layers.append(
            Conv2dNormActivation(
                in_channels,
                firstconv_output_channels,
                kernel_size=in_conv_kernel,
                stride=in_conv_stride,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish,
                bias=False,
            )
        )

        # Build inverted residual blocks
        f_dim = cnn_out_size(f_dim, 1, 1, 3, 2)
        t_dim = cnn_out_size(t_dim, 1, 1, 3, 2)

        kernel_sizes = [in_conv_kernel]
        strides = [in_conv_stride]

        for cnf in inverted_residual_setting:
            f_dim = cnf.out_size(f_dim)
            t_dim = cnf.out_size(t_dim)
            cnf.f_dim, cnf.t_dim = f_dim, t_dim
            layers.append(block(cnf, se_cnf, norm_layer, depthwise_norm_layer))
            kernel_sizes.append(cnf.kernel)
            strides.append(cnf.stride)

        # Build last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = resolved_out_dim
        layers.append(
            Conv2dNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish,
                bias=False,
            )
        )

        self.out_dim = lastconv_output_channels

        # Assemble model modules
        self.features = nn.Sequential(*layers)
        self.feature_extractor = AugmentMelSTFT()

        # Init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def extract_features(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Convert raw audio waveforms into spectrogram features.

        Args:
            waveforms (torch.Tensor): Audio tensor of shape (batch, time).

        Returns:
            torch.Tensor: Spectrogram features of shape (batch, 1, freq, time).
        """
        return self.feature_extractor(waveforms)

    def forward(self, x: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        """
        Forward pass of the MobileNet backbone.

        Processes the input spectrogram through the convolutional features
        and inverted residual blocks.

        Args:
            x (Tensor): Input tensor. Expected shape (batch, freq, time)
                or (batch, 1, freq, time).

        Returns:
            Tensor: High-level feature maps of shape (batch, out_channels, f_out, t_out).
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)

        for layer in self.features:
            x = layer(x)

        return x
