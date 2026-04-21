from lightning.pytorch.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import os
import time
import torch
from gridfm_graphkit.utils.mlflow_artifact_utils import artifact_context


class EpochTimerCallback(Callback):
    """Records wall-clock duration and iteration rate of every training epoch.

    Logs the following metrics to the Lightning logger (e.g. MLflow) so that
    throughput and timing are tracked as time-series over the full training run:

    * ``perf/train_epoch_time_s``  – wall-clock seconds for the epoch
    * ``perf/train_it_s``          – training batches per second
    * ``perf/val_epoch_time_s``    – wall-clock seconds for the validation loop
    """

    def __init__(self):
        self.epoch_times: list[float] = []
        self._epoch_start: float | None = None
        self._batch_count: int = 0
        self._last_batch_count: int = 0
        self._val_start: float | None = None

    def on_train_epoch_start(self, trainer, pl_module):
        self._epoch_start = time.perf_counter()
        self._batch_count = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._batch_count += 1

    def on_train_epoch_end(self, trainer, pl_module):
        if self._epoch_start is not None:
            elapsed = time.perf_counter() - self._epoch_start
            self.epoch_times.append(elapsed)
            self._last_batch_count = self._batch_count
            self._epoch_start = None

            # Log epoch timing + throughput to MLflow (one point per epoch)
            pl_module.log(
                "perf/train_epoch_time_s",
                elapsed,
                on_epoch=True,
                on_step=False,
                logger=True,
                prog_bar=False,
                sync_dist=False,
            )
            if elapsed > 0 and self._batch_count > 0:
                pl_module.log(
                    "perf/train_it_s",
                    self._batch_count / elapsed,
                    on_epoch=True,
                    on_step=False,
                    logger=True,
                    prog_bar=False,
                    sync_dist=False,
                )

    def on_validation_epoch_start(self, trainer, pl_module):
        self._val_start = time.perf_counter()

    def on_validation_epoch_end(self, trainer, pl_module):
        if self._val_start is not None:
            val_elapsed = time.perf_counter() - self._val_start
            self._val_start = None
            pl_module.log(
                "perf/val_epoch_time_s",
                val_elapsed,
                on_epoch=True,
                on_step=False,
                logger=True,
                prog_bar=False,
                sync_dist=False,
            )

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
    """Persist the best model state_dict according to a monitored validation metric."""
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

    @staticmethod
    def _canonical_state_dict(pl_module):
        """Return a state dict with compile wrappers removed from key names."""
        state_dict = pl_module.state_dict()
        return {
            key.replace("model._orig_mod.", "model."): value
            for key, value in state_dict.items()
        }

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

            # Save the model's state_dict.
            # artifact_context handles both local and remote MLflow servers.
            logger = trainer.logger
            with artifact_context(logger, "model") as model_dir:
                model_path = os.path.join(model_dir, self.filename)
                torch.save(self._canonical_state_dict(pl_module), model_path)
