"""
The `data_classes` package provides Python dataclasses for modeling and managing
fundamental data objects, such as configuration settings.
"""
from .loss_config import CrossEntropyLossConfig as CrossEntropyLossConfig
from .optimization_config import AdamOptimizationConfig as AdamOptimizationConfig
from .optimization_config import AdamwOptimizationConfig as AdamwOptimizationConfig
from .scheduling_config import CosineAnnealingConfig as CosineAnnealingConfig
