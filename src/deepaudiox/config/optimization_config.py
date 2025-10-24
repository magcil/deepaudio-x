from dataclasses import dataclass

from deepaudiox.config.base_config import OptimizationConfig
from deepaudiox.config.optimization_config_registry import register_optimizer_config


@dataclass 
@register_optimizer_config("ADAM")
class AdamOptimizationConfig(OptimizationConfig):
    """Configuration for setting up an Adam optimizer.
    https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html

    Attributes:
        lr (float): The learning rate of the optimizer.
            Defaults to 1e-3.
        weight_decay (float): Specifies the amount of smoothing when computing the loss.
            Defaults to 0.0.
        betas (tuple): Used for computing running averages of gradient and its square.
            Defaults to (0.9, 0.999).
        eps (float): Term added to the denominator to improve numerical stability.
            Defaults to 1e-8.
        amsgrad (bool): Whether to use the AMSGrad variant of this algorithm.
            Defaults to False.
        foreach (bool): Whether the foreach implementation of the optimizer is used.
            Defaults to False.
        maximize (bool): Maximize the objective with respect to the params, instead of minimizing.
            Defaults to False.
        capturable (bool): Whether this instance is safe to capture in a graph.
            Defaults to False.
        differentiable (bool): Whether autograd should occur through the optimizer step in training.
            Defaults to False.
        fused (bool): Whether the fused implementation is used.
            Defaults to None.

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
    """ Configuration for setting up an AdamW optimizer.
        https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html

    Attributes:
        lr (float): The learning rate of the optimizer.
            Defaults to 1e-3.
        weight_decay (float): Specifies the amount of smoothing when computing the loss.
            Defaults to 0.0.
        betas (tuple): Used for computing running averages of gradient and its square.
            Defaults to (0.9, 0.999).
        eps (float): Term added to the denominator to improve numerical stability.
            Defaults to 1e-8.
        amsgrad (bool): Whether to use the AMSGrad variant of this algorithm.
            Defaults to False.
        foreach (bool): Whether the foreach implementation of the optimizer is used.
            Defaults to False.
        maximize (bool): Maximize the objective with respect to the params, instead of minimizing.
            Defaults to False.
        capturable (bool): Whether this instance is safe to capture in a graph.
            Defaults to False.
        differentiable (bool): Whether autograd should occur through the optimizer step in training.
            Defaults to False.
        fused (bool): Whether the fused implementation is used.
            Defaults to None.

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