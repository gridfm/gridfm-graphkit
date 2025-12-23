from lightning.pytorch.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import (
    Timer,
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
    )
import os
import torch


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




def get_training_callbacks(callbacks):
    #TODO: make monitored metric configurable 
    callback_list = []
    if callbacks.tol is not None and callbacks.patience is not None:
        early_stop_callback = EarlyStopping(
            monitor="Validation loss",
            min_delta=callbacks.tol,
            patience=callbacks.patience,
            verbose=False,
            mode="min",
        )
        callback_list.append(early_stop_callback)

    save_best_model_callback = SaveBestModelStateDict(
        monitor="Validation loss",
        mode="min",
        filename="best_model_state_dict.pt",
    )
    callback_list.append(save_best_model_callback)

    checkpoint_callback = ModelCheckpoint(
        monitor="Validation loss",  # or whichever metric you track
        mode="min",
        save_last=True,
        save_top_k=0,
    )
    callback_list.append(checkpoint_callback)

    if callbacks.max_run_duration is not None:
        max_run_callback = Timer(duration=callbacks.max_run_duration)
        callback_list.append(max_run_callback)

    if callbacks.monitor_learning_rate:
        learning_rate_callback = LearningRateMonitor(logging_interval="epoch")
        callback_list.append(learning_rate_callback)

    return callback_list
