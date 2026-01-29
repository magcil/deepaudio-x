from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class AudioClassificationItem:
    """Class for modeling a single-class classification item."""

    path: str | Path
    y_true: int
    class_name: str
    feature: np.ndarray | None = None
    segment_idx: int = 0

    def __post_init__(self):
        if isinstance(self.path, str):
            self.path = Path(self.path)

    def to_dict(self) -> dict:
        return {
            "path": str(self.path),
            "y_true": self.y_true,
            "class_name": self.class_name,
            "segment_idx": self.segment_idx,
            "feature": self.feature,
        }


@dataclass
class AudioPrediction:
    final_label: str
    final_posterior: float

    segment_labels: list[str] | None = None
    segment_posteriors: list[float] | None = None
