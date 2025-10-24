from torch.optim.lr_scheduler import CosineAnnealingLR
from src.deepaudiox.config.scheduling_config import CosineAnnealingConfig
from src.deepaudiox.schedulers.scheduler_registry import register_scheduler
from dataclasses import asdict


@register_scheduler("CosineAnnealingLR")
class CosineAnnealingScheduler(CosineAnnealingLR):
    """ A wrapper of the PyTorch CosineAnnealingLR scheduler.
        https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html

    """
    def __init__(self, optimizer: object, config: CosineAnnealingConfig):
        """Initialize the scheduler.
        
        Arguments:
            config (CosineAnnealingConfig): The parameters required for configuring the scheduler
            
        """
        kwargs = asdict(config)
        kwargs.pop("name", None)
        kwargs.pop("epochs", None)
        super().__init__(optimizer=optimizer, **kwargs)
