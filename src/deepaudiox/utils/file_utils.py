from pathlib import Path

import torch


def load_checkpoint(model_path: Path | str):
    """Loads a pretrained model checkpoint from a PyTorch file (.pt or .pth).

    Args:
        model_path (Path or str): Path to the model checkpoint file.

    Returns:
        dict: The loaded model state dictionary.

    """
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {model_path}. "
            f"Use `from deepaudiox.utils.downloader import Downloader` "
            f"and call`Downloader.download_pretrained_backbone()` to download the checkpoint locally."
        )

    ckpt = torch.load(model_path, map_location="cpu")

    return ckpt
