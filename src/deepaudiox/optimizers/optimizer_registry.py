import torch.nn as nn

from src.deepaudiox.config.base_config import OptimizationConfig

_OPTIMIZER_REGISTRY = {}

def register_optimizer(name):
    """Add an optimizer to the registry.

    Arguments:
        name: The name of the optimizer.
    
    """
    def decorator(cls):
        """Access an optimizer by its name."""
        _OPTIMIZER_REGISTRY[name] = cls
        return cls
    return decorator

def build_optimizer(model_params: list, config: OptimizationConfig) -> nn.Module: 
    """Build the optimizer given its name.

        Arguments:
            model_params (list): The model parameters to be optimized.
            config (OptimizationConfig): Parameters required for setting up an optimizer.
    
    """
    if config.name not in _OPTIMIZER_REGISTRY:
        raise ValueError(f"Unknown optimizer: {config.name}")
    
    optimizer_cls = _OPTIMIZER_REGISTRY[config.name]

    return optimizer_cls(model_params, config)

def list_optimizers() -> list:
    """List all registered optimizers."""
    return list(_OPTIMIZER_REGISTRY.keys())