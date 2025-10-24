from src.deepaudiox.config.base_config import OptimizationConfig

_OPTIMIZER_CONFIG_REGISTRY = {}

def register_optimizer_config(name):
    """Add the configuration of an optimizer to the registry.

    Arguments:
        name: The name of the optimizer.
    
    """
    def decorator(cls):
        """Access the configuration of an optimizer by its name."""
        _OPTIMIZER_CONFIG_REGISTRY[name] = cls
        return cls
    return decorator

def build_optimizer_config(params: dict, *args, **kwargs) -> OptimizationConfig:
    """Build the configuration of an optimizer given its name.

        Arguments:
            params (dict): Parameters for setting up an optimizer
            *args: Positional arguments to pass to the optimizer config constructor.
            **kwargs: Keyword arguments to customize the optimizer config parameters.
    
    """
    if not isinstance(params, dict):
        raise TypeError(f"'params' must be a dict, got {type(params).__name__}")

    name = params.get("name")
    if name is None:
        raise ValueError("Missing required key 'name' in params.")
    
    if name not in _OPTIMIZER_CONFIG_REGISTRY:
        raise ValueError(f"Unknown optimizer: {name}")
        
    optimizer_config_cls = _OPTIMIZER_CONFIG_REGISTRY[name]
    return optimizer_config_cls(*args, **{**params, **kwargs})

def list_optimizer_configs() -> list:
    """List all registered optimizer configurations."""
    return list(_OPTIMIZER_CONFIG_REGISTRY.keys())