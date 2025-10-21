from torch.nn import CrossEntropyLoss
from dataclasses import asdict
from src.deepaudiox.config.loss_config import CrossEntropyLossConfig
from .loss_registry import register_loss_function

@register_loss_function("CrossEntropyLoss")
class CrossEntropy(CrossEntropyLoss):
    def __init__(self, config: CrossEntropyLossConfig):
        kwargs = asdict(config)
        kwargs.pop("name", None)
        super().__init__(**kwargs)