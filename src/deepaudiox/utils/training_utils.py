import logging
from pathlib import Path

import torch
import torch.nn.functional as F


def get_logger() -> object:
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


def get_device() -> torch.device:
    """Returns the best available device for PyTorch computations.

    Returns:
        torch.device
    """
    if torch.cuda.is_available():
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
    features = [item["feature"] for item in batch]
    labels = torch.tensor([item["class_id"] for item in batch], dtype=torch.long)
    class_names = [item["class_name"] for item in batch]

    # Ensure features are tensors
    features = [torch.as_tensor(f) for f in features]

    # Find max length in this batch
    max_len = max(f.shape[-1] for f in features)

    # Pad each tensor to max_len
    padded_features = [
        F.pad(f, (0, max_len - f.shape[-1])) if f.shape[-1] < max_len else f[..., :max_len] for f in features
    ]

    # Stack into a single batch tensor: (batch_size, num_channels, num_samples)
    batch_features = torch.stack(padded_features)

    return {"feature": batch_features, "class_id": labels, "class_name": class_names}
