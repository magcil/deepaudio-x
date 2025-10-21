from dataclasses import dataclass
from .optimization_config_registry import register_optimizer_config


@dataclass
class OptimizationConfig:
    name: str = "ADAM"


@dataclass 
@register_optimizer_config("ADAM")
class AdamOptimizationConfig(OptimizationConfig):
    """Configuration for setting up an ADAM optimizer.

    Attributes:
        weight_decay (float): The L2 regularization coefficient. Defaults to 1e-2.

    """
    lr: float = 0.001  
    weight_decay: float = 0.0
    betas: tuple = (0.9, 0.999)
    eps: float = 0.00000001 #1e-08
    amsgrad: bool = False
    foreach: bool = None
    maximize: bool = False
    capturable: bool = False
    differentiable: bool = False
    fused: bool = None


@dataclass 
@register_optimizer_config("ADAMW")
class AdamwOptimizationConfig(OptimizationConfig):
    """Configuration for setting up an ADAMW optimizer.

    Attributes:
        weight_decay (float): The L2 regularization coefficient. Defaults to 1e-2.

    """
    lr: float = 0.001  
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)
    eps: float = 0.00000001 #1e-08
    amsgrad: bool = False
    foreach: bool = None
    maximize: bool = False
    capturable: bool = False
    differentiable: bool = False
    fused: bool = None