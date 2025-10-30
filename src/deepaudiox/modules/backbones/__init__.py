# deepaudiox/modules/backbones/__init__.py

from collections.abc import Callable

from deepaudiox.modules.backbones.base_backbone import BaseBackbone as Backbone
from deepaudiox.modules.backbones.beats.beatswrapper import BEATsBackbone

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
def beats_base() -> BEATsBackbone:
    """BEATs backbone without DivEncLayer."""
    return BEATsBackbone()
