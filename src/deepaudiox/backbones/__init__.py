# deepaudiox/models/backbones/__init__.py

from typing import Callable
import torch.nn as nn

from deepaudiox.backbones.beats.beatswrapper import beats_base, beats_div

# A dictionary mapping backbone names to their constructor functions
BACKBONES: dict[str, Callable[[], nn.Module]] = {}


def register_backbone(name: str):
    """Decorator to register a backbone class or factory function."""

    def decorator(fn: Callable[[], nn.Module]) -> Callable[[], nn.Module]:
        if name in BACKBONES:
            raise ValueError(f"Backbone '{name}' already registered.")
        BACKBONES[name] = fn
        return fn

    return decorator
