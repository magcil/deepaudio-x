from dataclasses import asdict

from torch.optim import Adam, AdamW

from deepaudiox.config.optimization_config import AdamOptimizationConfig, AdamwOptimizationConfig
from deepaudiox.optimizers.optimizer_registry import register_optimizer


@register_optimizer("ADAM")
class AdamOptimizer(Adam):
    """ A wrapper of the PyTorch Adam optimizer.
        https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html

    """
    def __init__(self, model_params: list, config: AdamOptimizationConfig):
        """Initialize the optimizer.
        
        Arguments:
            config (AdamOptimizationConfig): The parameters required for configuring the optimizer.
            
        """
        kwargs = asdict(config)
        kwargs.pop("name", None)
        super().__init__(model_params, **kwargs)


@register_optimizer("ADAMW")
class AdamwOptimizer(AdamW):
    """ A wrapper of the PyTorch AdamW optimizer.
        https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html

    """
    def __init__(self, model_params: list, config: AdamwOptimizationConfig):
        """Initialize the optimizer.
        
        Arguments:
            config (AdamwOptimizationConfig): The parameters required for configuring the optimizer.
            
        """
        kwargs = asdict(config)
        kwargs.pop("name", None)
        super().__init__(model_params, **kwargs)