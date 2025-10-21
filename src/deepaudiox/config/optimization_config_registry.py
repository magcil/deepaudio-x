_OPTIMIZER_CONFIG_REGISTRY = {}

def register_optimizer_config(name):
    def decorator(cls):
        _OPTIMIZER_CONFIG_REGISTRY[name] = cls
        return cls
    return decorator

def build_optimizer_config(name: str, *args, **kwargs):
    if name not in _OPTIMIZER_CONFIG_REGISTRY:
        raise ValueError(f"Unknown optimizer: {name}")
        
    optimizer_config_cls = _OPTIMIZER_CONFIG_REGISTRY[name]

    return optimizer_config_cls(name=name, *args, **kwargs)


def list_optimizer_configs():
    return list(_OPTIMIZER_CONFIG_REGISTRY.keys())