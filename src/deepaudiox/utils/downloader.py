from pathlib import Path
from typing import Literal

import requests
from tqdm import tqdm


BACKBONE_URLS = {
    "beats": "https://github.com/magcil/pretrained-ssl-audio-backbones/raw/refs/heads/main/models/BEATs_iter3_plus_AS2M_encoder.pt"
}


class Downloader():
    def __init__(self, backbone: Literal["beats"]):
        """Downloads a checkpoint (.pt or .pth file) with pretrained weights for the backbone.

        Attributes:
            backbone (str): Name of the backbone model to download weights for.

        """
        self.backbone = backbone
        self.url = BACKBONE_URLS[self.backbone]

        self.cache_dir = Path.home() / ".cache" / "deepaudiox"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.model_path = self.cache_dir /self.url.split("/")[-1]

    def download_pretrained_backbone(self):
        if self.model_path.exists():
            return self.model_path
        
        else:
            r = requests.get(self.url, stream=True)
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))
            total_mb = total_size / (1024 * 1024)
            chunk_size = 1024

            progress = tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=f"Downloading {total_mb:.2f} MB",
            )

            try:
                with open(self.model_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            progress.update(len(chunk))

                progress.close()
                print(f"Model stored as {self.url.split('/')[-1]} in {self.model_path}")

                return self.model_path
            
            except OSError as e:
                raise RuntimeError(f"Failed to write file to {self.model_path}: {e}")
