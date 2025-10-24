from dataclasses import asdict

from torch.nn import CrossEntropyLoss

from deepaudiox.config.loss_config import CrossEntropyLossConfig
from deepaudiox.loss_functions.loss_registry import register_loss_function


@register_loss_function("CrossEntropyLoss")
class CrossEntropy(CrossEntropyLoss):
    """ A wrapper of the PyTorch CrossEntropyLoss class.
        https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

    """
    def __init__(self, config: CrossEntropyLossConfig):
        """Initialize the loss function.
        
        Arguments:
            config (CrossEntropyLossConfig): The parameters required for configuring the loss

        """
        kwargs = asdict(config)
        kwargs.pop("name", None)
        super().__init__(**kwargs)