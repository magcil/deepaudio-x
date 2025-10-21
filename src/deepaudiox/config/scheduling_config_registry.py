_SCHEDULING_CONFIG_REGISTRY = {}

def register_scheduling_config(name):
    def decorator(cls):
        _SCHEDULING_CONFIG_REGISTRY[name] = cls
        return cls
    return decorator

def build_scheduling_config(name: str, *args, **kwargs):
    if name not in _SCHEDULING_CONFIG_REGISTRY:
        raise ValueError(f"Unknown SCHEDULER: {name}")
        
    scheduling_config_cls = _SCHEDULING_CONFIG_REGISTRY[name]

    return scheduling_config_cls(name=name, *args, **kwargs)

def list_scheduling_configs():
    return list(_SCHEDULING_CONFIG_REGISTRY.keys())