from src.deepaudiox.config.base_config import SchedulingConfig

_SCHEDULING_CONFIG_REGISTRY = {}

def register_scheduling_config(name: str):
    """Add the configuration of a scheduler to the registry.

    Arguments:
        name: The name of the scheduler.
    
    """
    def decorator(cls):
        """Access the configuration of a scheduler by its name."""
        _SCHEDULING_CONFIG_REGISTRY[name] = cls
        return cls
    return decorator

def build_scheduling_config(params: dict, *args, **kwargs) -> SchedulingConfig:
    """Build the configuration of a scheduler given its name.

        Arguments:
            params (dict): Parameters for setting up a scheduler
            *args: Positional arguments to pass to the scheduler config constructor.
            **kwargs: Keyword arguments to customize the scheduler config parameters.
    
    """
    if not isinstance(params, dict):
        raise TypeError(f"'params' must be a dict, got {type(params).__name__}")

    name = params.get("name")
    if name is None:
        raise ValueError("Missing required key 'name' in params.")
    
    if name not in _SCHEDULING_CONFIG_REGISTRY:
        raise ValueError(f"Unknown scheduler: '{name}'. Available: {list(_SCHEDULING_CONFIG_REGISTRY.keys())}")

    scheduling_config_cls = _SCHEDULING_CONFIG_REGISTRY[name]
    return scheduling_config_cls(*args, **{**params, **kwargs})

def list_scheduling_configs() -> list:
    """List all registered scheduler configurations."""
    return list(_SCHEDULING_CONFIG_REGISTRY.keys())