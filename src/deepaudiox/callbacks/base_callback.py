class BaseCallback:
    """Base class for all training callbacks.

    Includes methods that are called throught the training process
    and are responsible for saving checkpoints, logging messages,
    handling early stopping, and reporting training results.

    """

    def on_train_start(self, trainer):
        pass

    def on_train_end(self, trainer):
        pass

    def on_epoch_start(self, trainer):
        pass

    def on_epoch_end(self, trainer):
        pass

    def on_step_end(self, trainer):
        pass
