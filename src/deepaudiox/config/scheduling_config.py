from dataclasses import dataclass

from deepaudiox.config.base_config import SchedulingConfig
from deepaudiox.config.scheduling_config_registry import register_scheduling_config


@dataclass 
@register_scheduling_config("CosineAnnealingLR")
class CosineAnnealingConfig(SchedulingConfig):
    """ Configuration for setting up an CosineAnnealingLR scheduler.
        https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html

    Attributes:
        eta_min (float): Minimum learning rate.  Defaults to 0.0.
        last_epoch (int): The index of the last epoch. Defaults to -1.
        T_max (int): Maximum number of iterations. Defaults to 10.

    """
    eta_min: float = 0.0
    last_epoch: int = -1
    T_max: int = None

    def __post_init__(self):
        if self.T_max is None:
            self.T_max = self.epochs
