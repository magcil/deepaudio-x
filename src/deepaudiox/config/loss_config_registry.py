from src.deepaudiox.config.base_config import LossConfig

_LOSS_CONFIG_REGISTRY = {}

def register_loss_config(name: str):
    """Add the configuration of a loss function to the registry.

    Arguments:
        name: The name of the loss function.
    
    """
    def decorator(cls):
        """Access the configuration of a loss function by its name."""
        _LOSS_CONFIG_REGISTRY[name] = cls
        return cls
    
    return decorator

def build_loss_config(params: dict, *args, **kwargs) -> LossConfig:
    """Build the configuration of a loss function given its name.

        Arguments:
            params (dict): Parameters for setting up an optimizer
            *args: Positional arguments to pass to the loss function config constructor.
            **kwargs: Keyword arguments to customize the loss function config parameters.
    
    """
    if not isinstance(params, dict):
        raise TypeError(f"'params' must be a dict, got {type(params).__name__}")

    name = params.get("name")
    if name is None:
        raise ValueError("Missing required key 'name' in params.")
    
    if name not in _LOSS_CONFIG_REGISTRY:
        raise ValueError(f"Unknown loss function: {name}")
        
    loss_config_cls = _LOSS_CONFIG_REGISTRY[name]
    return loss_config_cls(*args, **{**params, **kwargs})

def list_loss_configs() -> list:
    """List all registered loss function configurations"""
    return list(_LOSS_CONFIG_REGISTRY.keys())