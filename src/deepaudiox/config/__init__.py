"""
The `data_classes` package provides Python dataclasses for modeling and managing
fundamental data objects, such as configuration settings.
"""
from .optimization_config import AdamOptimizationConfig, AdamwOptimizationConfig
from .scheduling_config import CosineAnnealingConfig
from .loss_config import LossConfig