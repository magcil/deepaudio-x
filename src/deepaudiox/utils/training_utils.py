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


def get_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("ConsoleLogger")
    return logger


def get_class_mapping(root_dir: str) -> dict[str, int]:
    """Load the class mapping given a folder of class sub-folders.

    Args:
        root_dir (str): The path to root folder

    Returns:
        Dict: The class mapping dictionary

    """
    root_path = Path(root_dir)
    if not root_path.is_dir():
        raise ValueError(f"The path '{root_dir}' is not a directory")

    subdirs = [d for d in root_path.iterdir() if d.is_dir()]
    class_mapping = {d.name: idx for idx, d in enumerate(sorted(subdirs))}

    return class_mapping


def get_device(device_index: int | None = None) -> torch.device:
    """Returns the best available device for PyTorch computations.

    Args:
        device_index (int | None): The GPU device index to use. If None, uses the default GPU if available.

    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        if device_index is not None and (device_index < 0 or device_index >= torch.cuda.device_count()):
            raise ValueError(f"Invalid device_index {device_index}. Available GPU count: {torch.cuda.device_count()}")
        if device_index is not None:
            device = torch.device(f"cuda:{device_index}")
            print(f"Using GPU: {torch.cuda.get_device_name(device_index)}")
        else:
            device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU (no GPU available)")

    return device


def pad_collate_fn(batch) -> dict:
    """
    Collate function that pads variable-length audio tensors in a batch.

    Args:
        batch (list of dict): Each dict contains 'feature', 'class_id', and 'class_name'.

    Returns:
        dict: Batched and padded tensors.

    """
    # Extract data
    features = [torch.from_numpy(item["feature"]) for item in batch]
    labels = torch.tensor([item["class_id"] for item in batch], dtype=torch.long)
    class_names = [item["class_name"] for item in batch]

    # Stack into a single batch tensor with padding zeros on right to max_len
    batch_features = pad_sequence(features, batch_first=True)

    return {"feature": batch_features, "class_id": labels, "class_name": class_names}


def random_split_audio_dataset(
    dataset: AudioClassificationDataset, train_ratio: float, generator: Generator = default_generator
) -> list[Subset[AudioClassificationDataset]]:
    """
    Split AudioClassificationDataset into train / val subsets specified by train ratio. Method accounts for segmentized
    waveforms.

    Args:
        dataset (AudioClassificationDataset): An AudioClassificationDataset
        train_ratio (float): Percentage of training set.
        generator (Generator): Random Generator.
    """
    # Validate ratio
    if not (0 <= train_ratio <= 1):
        raise ValueError("train_ratio must be between 0 and 1.")

    # Extract recording paths
    wav_files = np.array([d["path"] for d in dataset.instances])
    num_files = len(wav_files)

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

    # Generate list of shuffled indices
    indices = randperm(num_files, generator=generator).tolist()

    # Split dataset to train and validation
    if dataset.segment_map:
        segment_mapping_array = np.array([d["file_path"] for d in dataset.segment_map])
        shuffled_wav_files = wav_files[np.array(indices)]

        return [
            Subset(
                dataset,
                np.where(np.isin(segment_mapping_array, shuffled_wav_files[offset - length : offset]))[0].tolist(),
            )
            for offset, length in zip(itertools.accumulate(subset_lengths), subset_lengths, strict=False)
        ]

    else:
        return [
            Subset(dataset, indices[offset - length : offset])
            for offset, length in zip(itertools.accumulate(subset_lengths), subset_lengths, strict=False)
        ]
