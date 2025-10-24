import torch.nn as nn

from src.deepaudiox.config.base_config import SchedulingConfig

_SCHEDULER_REGISTRY = {}

def register_scheduler(name):
    """Add a scheduler to the registry.

    Arguments:
        name: The name of the scheduler.
    
    """
    def decorator(cls):
        """Access a scheduler by its name."""
        _SCHEDULER_REGISTRY[name] = cls
        return cls
    return decorator

def build_scheduler(optimizer: nn.Module, config: SchedulingConfig) -> nn.Module:
    """Build the scheduler given its name.

        Arguments:
            optimizer (object): The optimizer used during scheduling.
            config (SchedulingConfig): Parameters required for configuring the scheduler.
    
    """
    if config.name not in _SCHEDULER_REGISTRY:
        raise ValueError(f"Unknown scheduler: {config.name}")
    
    scheduler_cls = _SCHEDULER_REGISTRY[config.name]

    return scheduler_cls(optimizer, config)

def list_schedulers() -> list:
    """List all registered schedulers."""
    return list(_SCHEDULER_REGISTRY.keys())