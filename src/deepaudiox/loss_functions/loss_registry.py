from src.deepaudiox.config.loss_config import LossConfig


_LOSS_REGISTRY = {}

def register_loss_function(name):
    def decorator(cls):
        _LOSS_REGISTRY[name] = cls
        return cls
    return decorator

def build_loss_function(config: LossConfig):
    if config.name not in _LOSS_REGISTRY:
        raise ValueError(f"Unknown loss function: {config.name}")
    
    loss_cls = _LOSS_REGISTRY[config.name]

    return loss_cls(config)

def list_loss_functions():
    return list(_LOSS_REGISTRY.keys())