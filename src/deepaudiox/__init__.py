"""
This page provides the core API reference for DeepAudioX.
"""

__version__ = "0.4.1"

from deepaudiox.datasets.audio_classification_dataset import (  # noqa: F401
    AudioClassificationDataset,
    audio_classification_dataset_from_dictionary,
    audio_classification_dataset_from_dir,
)
from deepaudiox.loops.evaluator import Evaluator  # noqa: F401
from deepaudiox.loops.trainer import Trainer  # noqa: F401
from deepaudiox.modules.constructors import (  # noqa: F401
    AudioClassifierConstructor as AudioClassifier,
)
from deepaudiox.modules.constructors import (
    BackboneConstructor as Backbone,
)
from deepaudiox.schemas.types import BackboneName, PoolingName  # noqa: F401
from deepaudiox.utils.training_utils import (  # noqa: F401
    get_class_mapping_from_dir,
    get_class_mapping_from_list,
)

AVAILABLE_BACKBONES: tuple[str, ...] = ("beats", "passt", "mobilenet_05_as", "mobilenet_10_as", "mobilenet_40_as")
"""Supported pretrained backbone names available at runtime."""

AVAILABLE_POOLING: tuple[str, ...] = ("gap", "simpool", "ep")
"""Supported pooling layer names available at runtime."""

__all__ = [
    "AudioClassifier",
    "Backbone",
    "AudioClassificationDataset",
    "audio_classification_dataset_from_dictionary",
    "audio_classification_dataset_from_dir",
    "Evaluator",
    "Trainer",
    "BackboneName",
    "PoolingName",
    "AVAILABLE_BACKBONES",
    "AVAILABLE_POOLING",
    "get_class_mapping_from_dir",
    "get_class_mapping_from_list",
]
