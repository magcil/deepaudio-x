_LOSS_REGISTRY = {}

def register_loss_function(name):
    """Add a loss function to the registry.

    Arguments:
        name: The name of the loss function.
    
    """
    def decorator(cls):
        """Access a loss function by its name."""
        _LOSS_REGISTRY[name] = cls
        return cls
    return decorator

def build_loss_function(config):
    """Build the loss function given its name.

        Arguments:
            config (LossConfig): Parameters required for setting up a loss function.
    
    """
    if config.name not in _LOSS_REGISTRY:
        raise ValueError(f"Unknown loss function: {config.name}")
    
    loss_cls = _LOSS_REGISTRY[config.name]

    return loss_cls(config)

def list_loss_functions():
    """List all registered loss functions."""
    return list(_LOSS_REGISTRY.keys())