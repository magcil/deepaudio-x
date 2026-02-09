API Reference
=============

.. currentmodule:: deepaudiox

Dataset Construction
--------------------

Methods for building datasets and class mappings from directories or label lists.

.. autofunction:: get_class_mapping_from_dir

.. autofunction:: get_class_mapping_from_list

.. autoclass:: AudioClassificationDataset
   :members:
   :exclude-members: __init__
   :undoc-members:

.. autofunction:: audio_classification_dataset_from_dir

.. autofunction:: audio_classification_dataset_from_dictionary

Models & Backbones
------------------

Constructors and registries for initializing classifiers, backbones, and pooling.

.. autoclass:: AudioClassifierConstructor
   :members:
   :exclude-members: __init__
   :undoc-members:

.. autoclass:: BackboneConstructor
   :members:
   :exclude-members: __init__
   :undoc-members:

.. data:: BACKBONES
   :annotation: = dict

.. data:: POOLING
   :annotation: = dict

.. autoclass:: AudioClassifier
   :noindex:

.. autoclass:: Backbone
   :noindex:

Training & Evaluation
---------------------

Interfaces for training models and evaluating performance on held-out data.

.. autoclass:: Trainer
   :members:
   :exclude-members: __init__
   :undoc-members:

.. autoclass:: Evaluator
   :members:
   :exclude-members: __init__
   :undoc-members:

Base Classes & Inference
------------------------

Base interfaces and inference helpers used across models.

.. automodule:: deepaudiox.modules.baseclasses
   :members: BaseAudioClassifier, BaseBackbone, BasePooling
   :undoc-members:

Full Paths
----------

The API re-exports the following symbols. If you prefer importing from the original modules, use these paths:

- ``AudioClassifier`` -> ``deepaudiox.modules.constructors.AudioClassifierConstructor``
- ``AudioClassifierConstructor`` -> ``deepaudiox.modules.constructors.AudioClassifierConstructor``
- ``Backbone`` -> ``deepaudiox.modules.constructors.BackboneConstructor``
- ``BackboneConstructor`` -> ``deepaudiox.modules.constructors.BackboneConstructor``
- ``AudioClassificationDataset`` -> ``deepaudiox.datasets.audio_classification_dataset.AudioClassificationDataset``
- ``audio_classification_dataset_from_dir`` -> ``deepaudiox.datasets.audio_classification_dataset.audio_classification_dataset_from_dir``
- ``audio_classification_dataset_from_dictionary`` -> ``deepaudiox.datasets.audio_classification_dataset.audio_classification_dataset_from_dictionary``
- ``get_class_mapping_from_dir`` -> ``deepaudiox.utils.training_utils.get_class_mapping_from_dir``
- ``get_class_mapping_from_list`` -> ``deepaudiox.utils.training_utils.get_class_mapping_from_list``
- ``Trainer`` -> ``deepaudiox.loops.trainer.Trainer``
- ``Evaluator`` -> ``deepaudiox.loops.evaluator.Evaluator``
- ``BACKBONES`` -> ``deepaudiox.modules.backbones.BACKBONES``
- ``POOLING`` -> ``deepaudiox.modules.pooling.POOLING``
