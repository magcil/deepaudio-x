from dataclasses import dataclass


@dataclass
class AudioPrediction:
    final_label: str
    final_posterior: float

    segment_labels: list[str] | None = None
    segment_posteriors: list[float] | None = None

    def to_dict(self) -> dict:
        return {
            "final_label": self.final_label,
            "final_posterior": self.final_posterior,
            "segment_labels": self.segment_labels,
            "segment_posteriors": self.segment_posteriors,
        }
