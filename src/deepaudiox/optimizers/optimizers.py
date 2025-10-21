from torch.optim import Adam, AdamW
from ..config.optimization_config import AdamOptimizationConfig, AdamwOptimizationConfig
from .optimizer_registry import register_optimizer
from dataclasses import asdict


@register_optimizer("ADAM")
class AdamOptimizer(Adam):
    def __init__(self, model_params: list, config: AdamOptimizationConfig):
        kwargs = asdict(config)
        kwargs.pop("name", None)
        super().__init__(model_params, **kwargs)


@register_optimizer("ADAMW")
class AdamwOptimizer(AdamW):
    def __init__(self, model_params: list, config: AdamwOptimizationConfig):
        kwargs = asdict(config)
        kwargs.pop("name", None)
        super().__init__(model_params, **kwargs)