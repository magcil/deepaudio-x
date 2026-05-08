import itertools
import logging
import math
import warnings
from pathlib import Path

import numpy as np
import torch
from torch import Generator, default_generator, randperm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Subset

from deepaudiox.datasets.audio_classification_dataset import AudioClassificationDataset
from deepaudiox.schemas.types import DeviceName


def get_logger() -> logging.Logger:
    """Initialize and return a console logger."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("ConsoleLogger")
    return logger


def get_class_mapping_from_list(labels: list[str], sort_alphabetically: bool = True) -> dict[str, int]:
    """Get a class mapping dictionary given a list of class names.

    Args:
        labels (list[str]): List of class names
        sort_alphabetically (boolean): Determines if alphabetical sorting should be applied to class names.

    Returns:
        dict[str, int]: The class mapping dictionary

    Example:
        >>> from deepaudiox import get_class_mapping_from_list
        >>> labels = ["speech", "music", "noise"]
        >>> class_mapping = get_class_mapping_from_list(labels, sort_alphabetically=True)
        >>> # Example output:
        >>> # {'music': 0, 'noise': 1, 'speech': 2}
    """
    if sort_alphabetically:
        class_mapping = {name: idx for idx, name in enumerate(sorted(labels))}
    else:
        class_mapping = {name: idx for idx, name in enumerate(labels)}

    return class_mapping


def get_class_mapping_from_dir(root_dir: str) -> dict[str, int]:
    """Load the class mapping given a folder of class sub-folders.

    Expected directory structure::

        root_dir/
        ├── class_a/
        │   ├── audio1.wav
        │   └── audio2.wav
        └── class_b/
            ├── audio3.wav
            └── audio4.wav

    Args:
        root_dir (str): The path to root folder

    Returns:
        dict[str, int]: The class mapping dictionary, ordered alphabetically by folder name.

    Example:
        >>> from deepaudiox import get_class_mapping_from_dir
        >>> class_mapping = get_class_mapping_from_dir(root_dir="path/to/data")
        >>> # Example output:
        >>> # {'class_a': 0, 'class_b': 1}
    """

    root_path = Path(root_dir)
    if not root_path.is_dir():
        raise ValueError(f"The path '{root_dir}' is not a directory")

    subdirs = [d for d in root_path.iterdir() if d.is_dir()]
    class_mapping = {d.name: idx for idx, d in enumerate(sorted(subdirs))}

    return class_mapping


def get_device(device: DeviceName = "cpu", device_index: int | None = None) -> torch.device:
    """Returns a PyTorch device based on the user's choice.

    Args:
        device (DeviceName): The device to use. One of ``"cuda"``, ``"mps"``, or ``"cpu"``.
            Defaults to ``"cpu"``.
        device_index (int | None): The GPU device index. Only applicable when ``device="cuda"``.
            If ``None``, uses the default CUDA device.

    Returns:
        torch.device

    Raises:
        ValueError: If ``device="cuda"`` but CUDA is not available.
        ValueError: If ``device="cuda"`` and ``device_index`` is out of range.
        ValueError: If ``device="mps"`` but MPS is not available.
        ValueError: If ``device_index`` is provided for a non-CUDA device.
    """
    if device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available on this machine. Use device='cpu' or device='mps' instead.")
        if device_index is not None and (device_index < 0 or device_index >= torch.cuda.device_count()):
            raise ValueError(f"Invalid device_index {device_index}. Available GPU count: {torch.cuda.device_count()}")
        if device_index is not None:
            torch_device = torch.device(f"cuda:{device_index}")
            print(f"Using GPU: {torch.cuda.get_device_name(device_index)}")
        else:
            torch_device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif device == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("MPS is not available on this machine. Use device='cpu' instead.")
        if device_index is not None:
            raise ValueError("device_index is not supported for MPS. Apple Silicon has a single GPU.")
        torch_device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    else:
        if device_index is not None:
            print("Warning: device_index is ignored when device='cpu'.")
        torch_device = torch.device("cpu")
        print("Using CPU.")

    return torch_device


def pad_collate_fn(batch) -> dict:
    """Collate function that pads variable-length audio tensors in a batch.

    Uses ``torch.stack`` when all waveforms in the batch have the same length
    (e.g. when ``segment_duration`` is set), and falls back to ``pad_sequence``
    for variable-length batches.

    Args:
        batch (list of dict): Each dict contains 'feature', 'y_true', and 'class_name'.

    Returns:
        dict[str, torch.Tensor | list[str]]: Batched and padded tensors.
    """
    features = [torch.from_numpy(item["feature"]) for item in batch]
    labels = torch.tensor([item["y_true"] for item in batch], dtype=torch.long)
    class_names = [item["class_name"] for item in batch]

    if len({f.shape[0] for f in features}) == 1:
        batch_features = torch.stack(features)
    else:
        batch_features = pad_sequence(features, batch_first=True)

    return {"feature": batch_features, "y_true": labels, "class_name": class_names}


def random_split_audio_dataset(
    dataset: AudioClassificationDataset, train_ratio: float, generator: Generator = default_generator
) -> list[Subset[AudioClassificationDataset]]:
    """
    Split AudioClassificationDataset into train / val subsets specified by train ratio.
    Method accounts for segmentized waveforms.

    Args:
        dataset (AudioClassificationDataset): An AudioClassificationDataset
        train_ratio (float): Percentage of training set.
        generator (Generator): Random Generator.

    Returns:
        list[Subset[AudioClassificationDataset]]: List containing train and validation Subsets.
    """
    # Validate ratio
    if not (0 <= train_ratio <= 1):
        raise ValueError("train_ratio must be between 0 and 1.")

    # Extract recording paths
    file_paths = np.array([item.path for item in dataset.items])
    num_files = len(np.unique(file_paths)) if dataset.segment_duration else len(file_paths)

    # Compute split sizes
    n_train = int(math.floor(num_files * train_ratio))
    n_valid = num_files - n_train
    subset_lengths = [n_train, n_valid]

    # Validate split sizes
    if n_train == 0:
        warnings.warn("Training split has length 0.", stacklevel=2)
    if n_valid == 0:
        warnings.warn("Validation split has length 0.", stacklevel=2)
    if sum(subset_lengths) != num_files:
        raise ValueError("Split sizes do not sum to the total number of items.")

    if dataset.segment_duration:
        # Segmentized dataset: shuffle files, then map to segments
        unique_files = np.unique(file_paths)
        num_unique_files = len(unique_files)
        shuffled_file_indices = randperm(num_unique_files, generator=generator).tolist()
        shuffled_files = unique_files[shuffled_file_indices]

        split_files_list = [
            shuffled_files[offset - length : offset]
            for offset, length in zip(itertools.accumulate(subset_lengths), subset_lengths, strict=False)
        ]

        subsets = [
            Subset(dataset, np.where(np.isin(file_paths, split_files))[0].tolist()) for split_files in split_files_list
        ]

        return subsets

    else:
        # Non-segmentized dataset: shuffle indices directly
        indices = randperm(num_files, generator=generator).tolist()
        subsets = [
            Subset(dataset, indices[offset - length : offset])
            for offset, length in zip(itertools.accumulate(subset_lengths), subset_lengths, strict=False)
        ]
        return subsets
