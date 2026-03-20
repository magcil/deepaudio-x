import functools
from collections.abc import Callable
from typing import TypeVar, cast

F = TypeVar("F", bound=Callable)


def eval_mode(method: F) -> F:
    """Decorator that runs a method in evaluation mode and restores the original training state afterward.

    Calls ``self.eval()`` before the method executes and restores the model to
    training mode via ``self.train()`` in a ``finally`` block if the model was
    training prior to the call. This guarantees the model state is always
    restored, even if the method raises an exception.

    Intended for use on inference methods of ``torch.nn.Module`` subclasses
    where eval mode (disabled dropout, frozen BatchNorm statistics) is required
    but the caller should not be responsible for managing model state.

    Args:
        method (Callable): The instance method to wrap.

    Returns:
        Callable: The wrapped method with automatic eval/train mode management.

    Example:
        >>> from deepaudiox.utils.decorators import eval_mode
        >>> class MyModel(nn.Module):
        ...     @eval_mode
        ...     def predict(self, x):
        ...         return self.forward(x)
    """

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        was_training = self.training
        self.eval()
        try:
            return method(self, *args, **kwargs)
        finally:
            if was_training:
                self.train()

    return cast(F, wrapper)
