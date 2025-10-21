import time
from .base_callback import BaseCallback

class ConsoleLogger(BaseCallback):
    def __init__(self, log_interval: int = 10):
        self.log_interval = log_interval
        self.last_time = time.time()

    def on_epoch_start(self, trainer):
        print(f"\nEpoch {trainer.state.current_epoch}/{trainer.epochs}")

    def on_step_end(self, trainer):
        step = trainer.state.step
        if step % self.log_interval == 0:
            elapsed = time.time() - self.last_time
            # loss = trainer.state.loss
            loss = 20000
            print(f"  Step {step:<5d} | Loss: {loss:.4f} | Time: {elapsed:.2f}s")
            self.last_time = time.time()

    def on_epoch_end(self, trainer):
        print(f"Epoch {trainer.state.current_epoch} finished.")

    def on_train_end(self, trainer):
        print("Training has finished.")
