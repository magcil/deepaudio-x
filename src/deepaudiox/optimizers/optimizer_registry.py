from src.deepaudiox.config.optimization_config import OptimizationConfig


_OPTIMIZER_REGISTRY = {}

def register_optimizer(name):
    def decorator(cls):
        _OPTIMIZER_REGISTRY[name] = cls
        return cls
    return decorator

def build_optimizer(model_params, config: OptimizationConfig):
    if config.name not in _OPTIMIZER_REGISTRY:
        raise ValueError(f"Unknown optimizer: {config.name}")
    
    optimizer_cls = _OPTIMIZER_REGISTRY[config.name]

    return optimizer_cls(model_params, config)


def list_optimizers():
    return list(_OPTIMIZER_REGISTRY.keys())