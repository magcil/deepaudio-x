from torch.optim.lr_scheduler import CosineAnnealingLR
from ..config.scheduling_config import CosineAnnealingConfig
from .scheduler_registry import register_scheduler
from dataclasses import asdict


@register_scheduler("CosineAnnealingLR")
class CosineAnnealingScheduler(CosineAnnealingLR):
    def __init__(self, optimizer: object, config: CosineAnnealingConfig):
        kwargs = asdict(config)
        kwargs.pop("name", None)
        super().__init__(optimizer=optimizer, **kwargs)
