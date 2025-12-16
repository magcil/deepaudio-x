from pathlib import Path
from typing import Literal

import requests
from platformdirs import user_cache_dir
from tqdm import tqdm

# Repository URL where the pretrained backbone models are hosted
BACKBONE_REPO_URL = "https://github.com/magcil/pretrained-ssl-audio-backbones/raw/refs/heads/main/models/"

BACKBONE_URLS = {"beats": BACKBONE_REPO_URL + "BEATs_iter3_plus_AS2M.pt"}


class Downloader:
    """Downloads a checkpoint (.pt or .pth file) with pretrained weights for the backbone.

    Attributes:
        BACKBONES_URLS (dict): A dictionary mapping backbone names to their respective download URLs.
        cache_dir (Path): Directory path where downloaded models are cached.

    Cache dirs:
            - Linux: ~/.cache/deepaudiox
            - macOS: ~/Library/Caches/deepaudiox
            - Windows: C:\\Users\\<Username>\\AppData\\Local\\deepaudiox\\Cache
    """

    def __init__(self):
        """Initializes the Downloader instance."""
        self.BACKBONES_URLS = BACKBONE_URLS

        self.cache_dir = Path(user_cache_dir("deepaudiox"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download_checkpoint(self, backbone: Literal["beats"]) -> Path:
        """Downloads the pretrained backbone weights if not already cached.

        Args:
            backbone (Literal["beats"]): Name of the backbone model to download weights for.

        Returns:
            Path to the downloaded model file.
        """
        url = self.BACKBONES_URLS[backbone]
        model_path = self.cache_dir / url.split("/")[-1]
        if model_path.exists():
            return model_path
        else:
            r = requests.get(url, stream=True)
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))
            total_mb = total_size / (1024 * 1024)
            chunk_size = 1024

            progress = tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=f"Downloading pretrained backbone(Saving in {self.cache_dir}) {total_mb:.2f} MB",
            )

            try:
                with open(model_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            progress.update(len(chunk))

                progress.close()
                print(f"Model stored as {url.split('/')[-1]} in {model_path}")

                return model_path

            except OSError as e:
                raise RuntimeError(f"Failed to write file to {model_path}: {e}") from e
