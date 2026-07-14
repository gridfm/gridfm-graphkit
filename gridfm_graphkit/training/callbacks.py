from lightning.pytorch.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from lightning.pytorch.loggers import MLFlowLogger
import os
import time
import torch

# Metric logged in validation_step as f"Validation {metric}" for layer_11_residual.
BEST_CHECKPOINT_MONITOR = "Validation layer_11_residual"
LR_SCHEDULER_MONITOR = "Validation loss"
BEST_MODEL_FILENAME = "best_model_state_dict.pt"
COMPILE_STATE_DICT_PREFIX = "model._orig_mod."
MLFLOW_FILE_URI_PREFIX = "file:"


def canonicalize_state_dict_keys(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Strip torch.compile wrappers from checkpoint key names."""
    return {
        key.replace(COMPILE_STATE_DICT_PREFIX, "model."): value
        for key, value in state_dict.items()
    }


def add_compile_state_dict_prefix(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Add torch.compile wrappers to canonical model.* checkpoint keys."""
    return {
        (
            key.replace("model.", COMPILE_STATE_DICT_PREFIX, 1)
            if key.startswith("model.") and not key.startswith(COMPILE_STATE_DICT_PREFIX)
            else key
        ): value
        for key, value in state_dict.items()
    }


def adapt_state_dict_for_model(
    state_dict: dict[str, torch.Tensor],
    model,
) -> dict[str, torch.Tensor]:
    """Remap checkpoint keys to match the target module's state_dict namespace."""
    target_keys = set(model.state_dict().keys())
    source_keys = set(state_dict.keys())
    if source_keys == target_keys:
        return state_dict

    canonical = canonicalize_state_dict_keys(state_dict)
    if set(canonical.keys()) == target_keys:
        return canonical

    compiled = add_compile_state_dict_prefix(canonical)
    if set(compiled.keys()) == target_keys:
        return compiled

    return canonical


def mlflow_local_root(logger) -> str | None:
    """Local MLflow store root; None when tracking URI is not a file: path."""
    save_dir = getattr(logger, "save_dir", None)
    if save_dir:
        return save_dir
    if isinstance(logger, MLFlowLogger):
        tracking_uri = getattr(logger, "_tracking_uri", None) or ""
        if tracking_uri.startswith(MLFLOW_FILE_URI_PREFIX):
            return tracking_uri[len(MLFLOW_FILE_URI_PREFIX) :]
    return None


def mlflow_run_artifacts_dir(logger, *parts: str) -> str | None:
    """Return <store>/<exp_id>/<run_id>/artifacts/... when resolvable."""
    if isinstance(logger, MLFlowLogger):
        experiment_id = logger.experiment_id
        run_id = logger.run_id
        root = mlflow_local_root(logger)
        if root and experiment_id and run_id:
            return os.path.join(root, experiment_id, run_id, "artifacts", *parts)
        return None
    root = getattr(logger, "save_dir", None)
    if root:
        return os.path.join(root, *parts)
    return None


def best_model_artifact_path(
    logger,
    filename: str = BEST_MODEL_FILENAME,
) -> str:
    """Return the on-disk path where SaveBestModelStateDict writes the checkpoint."""
    path = mlflow_run_artifacts_dir(logger, "model", filename)
    if path:
        return path
    root = mlflow_local_root(logger) or getattr(logger, "save_dir", None)
    return os.path.join(root, "model", filename)


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
    """Persist the best model state_dict according to a monitored validation metric."""
    def __init__(
        self,
        monitor: str,
        mode: str = "min",
        filename: str = BEST_MODEL_FILENAME,
    ):
        self.monitor = monitor
        self.mode = mode
        self.filename = filename
        self.best_score = float("inf") if mode == "min" else -float("inf")

    @staticmethod
    def _canonical_state_dict(pl_module):
        """Return a state dict with compile wrappers removed from key names."""
        return canonicalize_state_dict_keys(pl_module.state_dict())

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

            logger = trainer.logger
            model_path = best_model_artifact_path(logger, self.filename)
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

            # Save the model's state_dict
            torch.save(self._canonical_state_dict(pl_module), model_path)
