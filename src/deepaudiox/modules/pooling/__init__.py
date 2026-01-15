# deepaudiox/modules/pooling/__init__.py

"""
The 'pooling' package contains pooling modules for pooling the embeddings of the backbones.
"""

from collections.abc import Callable

from deepaudiox.modules.pooling.base_pooling import BasePooling
from deepaudiox.modules.pooling.gap import GAP
from deepaudiox.modules.pooling.simpool import SimPool

# A dictionary mapping pooling names to their constructor functions
POOLING: dict[str, Callable[..., BasePooling]] = {}


def register_pooling(name: str):
    """Decorator to register a pooling layer class or factory function."""

    def decorator(fn: Callable[..., BasePooling]) -> Callable[..., BasePooling]:
        if name in POOLING:
            raise ValueError(f"Pooling layer '{name}' already registered.")
        POOLING[name] = fn
        return fn

    return decorator


@register_pooling("gap")
def gap(**kwargs) -> GAP:
    """Global Average Pooling. Ignores dim and other kwargs."""
    return GAP()


@register_pooling("simpool")
def simpool(dim: int, **kwargs) -> SimPool:
    """Simple Pooling.

    Args:
        dim (int): Input feature dimension.
    """
    return SimPool(dim=dim)
