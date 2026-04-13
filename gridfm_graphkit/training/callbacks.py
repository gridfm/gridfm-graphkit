from lightning.pytorch.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from lightning.pytorch.loggers import MLFlowLogger
import os
import time
import torch


class EpochTimerCallback(Callback):
    """Records wall-clock duration and iteration rate of every training epoch."""

    def __init__(self):
        self.epoch_times: list[float] = []
        self._epoch_start: float | None = None
        self._batch_count: int = 0
        self._last_batch_count: int = 0

    def on_train_epoch_start(self, trainer, pl_module):
        self._epoch_start = time.perf_counter()
        self._batch_count = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._batch_count += 1

    def on_train_epoch_end(self, trainer, pl_module):
        if self._epoch_start is not None:
            self.epoch_times.append(time.perf_counter() - self._epoch_start)
            self._last_batch_count = self._batch_count
            self._epoch_start = None

    @property
    def last_epoch_time(self) -> float | None:
        return self.epoch_times[-1] if self.epoch_times else None

    @property
    def last_epoch_iters_per_sec(self) -> float | None:
        t = self.last_epoch_time
        if t is None or t == 0 or self._last_batch_count == 0:
            return None
        return self._last_batch_count / t


class SaveBestModelStateDict(Callback):
    def __init__(
        self,
        monitor: str,
        mode: str = "min",
        filename: str = "best_model_state_dict.pt",
    ):
        self.monitor = monitor
        self.mode = mode
        self.filename = filename
        self.best_score = float("inf") if mode == "min" else -float("inf")

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        current = trainer.callback_metrics.get(self.monitor)
        if current is None:
            return  # Metric not available yet

        # Check if this is the best score so far
        if (self.mode == "min" and current < self.best_score) or (
            self.mode == "max" and current > self.best_score
        ):
            self.best_score = current

            # Determine artifact directory
            logger = trainer.logger
            if isinstance(logger, MLFlowLogger):
                model_dir = os.path.join(
                    logger.save_dir,
                    logger.experiment_id,
                    logger.run_id,
                    "artifacts",
                    "model",
                )
            else:
                model_dir = os.path.join(logger.save_dir, "model")

            os.makedirs(model_dir, exist_ok=True)

            # Save the model's state_dict
            model_path = os.path.join(model_dir, self.filename)
            torch.save(pl_module.state_dict(), model_path)
