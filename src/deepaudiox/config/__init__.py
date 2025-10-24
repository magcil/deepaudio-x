"""
The `data_classes` package provides Python dataclasses for modeling and managing
fundamental data objects, such as configuration settings.
"""
from .loss_config import CrossEntropyLossConfig
from .optimization_config import AdamOptimizationConfig
from .optimization_config import AdamwOptimizationConfig
from .scheduling_config import CosineAnnealingConfig
