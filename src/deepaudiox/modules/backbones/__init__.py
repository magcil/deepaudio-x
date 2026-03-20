# deepaudiox/modules/backbones/__init__.py

import warnings
from collections.abc import Callable

from deepaudiox.modules.backbones.beats.beats_modules.BEATs import BEATs
from deepaudiox.modules.backbones.mobilenet.model import MobileNet, MobileNetConfig
from deepaudiox.modules.backbones.passt.passt import PaSST
from deepaudiox.modules.baseclasses import BaseBackbone as Backbone

# A dictionary mapping backbone names to their constructor functions
BACKBONES: dict[str, Callable[[], Backbone]] = {}


def register_backbone(name: str):
    """Decorator to register a backbone class or factory function."""

    def decorator(fn: Callable[[], Backbone]) -> Callable[[], Backbone]:
        if name in BACKBONES:
            raise ValueError(f"Backbone '{name}' already registered.")
        BACKBONES[name] = fn
        return fn

    return decorator


@register_backbone("beats")
def beats_base() -> BEATs:
    """BEATs backbone without DivEncLayer."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.utils.weight_norm")
        return BEATs()


@register_backbone("passt")
def passt_base() -> PaSST:
    """PaSST backbone"""
    return PaSST()


@register_backbone("mobilenet_05_as")
def mobilenet_05_base() -> MobileNet:
    """MobileNet backbone"""
    return MobileNet(cfg=MobileNetConfig({"width_mult": 0.5}))


@register_backbone("mobilenet_10_as")
def mobilenet_10_base() -> MobileNet:
    """MobileNet backbone"""
    return MobileNet(cfg=MobileNetConfig({"width_mult": 1}))


@register_backbone("mobilenet_40_as")
def mobilenet_40_base() -> MobileNet:
    """MobileNet backbone"""
    return MobileNet(cfg=MobileNetConfig({"width_mult": 4}))
