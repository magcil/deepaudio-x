from src.deepaudiox.config.scheduling_config import SchedulingConfig


_SCHEDULER_REGISTRY = {}

def register_scheduler(name):
    def decorator(cls):
        _SCHEDULER_REGISTRY[name] = cls
        return cls
    return decorator

def build_scheduler(optimizer: object, config: SchedulingConfig):
    if config.name not in _SCHEDULER_REGISTRY:
        raise ValueError(f"Unknown scheduler: {config.name}")
    
    scheduler_cls = _SCHEDULER_REGISTRY[config.name]

    return scheduler_cls(optimizer, config)

def list_schedulers():
    return list(_SCHEDULER_REGISTRY.keys())