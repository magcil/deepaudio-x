from dataclasses import dataclass
from .scheduling_config_registry import register_scheduling_config

@dataclass
class SchedulingConfig:
    """Configuration for setting up a scheduler.

    Attributes:
        name (str): The name of the scheduler. Defaults to CosineAnnealing.

    """
    name: str = "CosineAnnealingLR"

@dataclass 
@register_scheduling_config("CosineAnnealingLR")
class CosineAnnealingConfig(SchedulingConfig):
    """Configuration for setting up an CosineAnnealingLR scheduler.

    Attributes:
        pending

    """
    eta_min: float = 0.0
    last_epoch: int = -1
    T_max: int = 10
