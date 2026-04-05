from typing import Literal

BackboneName = Literal["beats", "passt", "mobilenet_05_as", "mobilenet_10_as", "mobilenet_40_as"]
"""Supported pretrained backbone names."""

PoolingName = Literal["gap", "simpool", "ep"]
"""Supported pooling layer names."""

DeviceName = Literal["cuda", "mps", "cpu"]
"""Supported device names."""
