_LOSS_CONFIG_REGISTRY = {}

def register_loss_config(name):
    def decorator(cls):
        _LOSS_CONFIG_REGISTRY[name] = cls
        return cls
    return decorator

def build_loss_config(name: str, *args, **kwargs):
    if name not in _LOSS_CONFIG_REGISTRY:
        raise ValueError(f"Unknown loss function: {name}")
        
    loss_config_cls = _LOSS_CONFIG_REGISTRY[name]

    return loss_config_cls(name=name, *args, **kwargs)

def list_loss_configs():
    return list(_LOSS_CONFIG_REGISTRY.keys())