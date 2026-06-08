from pathlib import Path

import huggingface_hub
from huggingface_hub import hf_hub_download, try_to_load_from_cache

huggingface_hub.logging.set_verbosity_error()

from deepaudiox.schemas.types import BackboneName

HF_REPO_ID = "magcil/deepaudiox-backbones"

BACKBONE_FILENAMES = {
    "beats": "BEATs_iter3_plus_AS2M.pt",
    "passt": "passt-s-kd-ap.486.pt",
    "mobilenet_05_as": "mn05_as_mAP_443.pt",
    "mobilenet_10_as": "mn10_as_mAP_471.pt",
    "mobilenet_40_as": "mn40_as_mAP_484.pt",
}


class Downloader:
    """Downloads a checkpoint (.pt or .pth file) with pretrained weights for the backbone.

    Attributes:
        BACKBONE_FILENAMES (dict): A dictionary mapping backbone names to their filenames on HF Hub.
    """

    def __init__(self):
        """Initializes the Downloader instance."""
        self.BACKBONE_FILENAMES = BACKBONE_FILENAMES

    def download_checkpoint(self, backbone: BackboneName) -> Path:
        """Downloads the pretrained backbone weights if not already cached.

        Args:
            backbone (BackboneName): Backbone name to download weights for.
                One of: "beats", "passt", "mobilenet_05_as", "mobilenet_10_as", "mobilenet_40_as".

        Returns:
            Path to the downloaded model file.
        """
        filename = self.BACKBONE_FILENAMES[backbone]
        if try_to_load_from_cache(repo_id=HF_REPO_ID, filename=filename) is None:
            print(f"Downloading pretrained weights for '{backbone}' from HuggingFace Hub ({HF_REPO_ID})...")
        return Path(hf_hub_download(repo_id=HF_REPO_ID, filename=filename))
