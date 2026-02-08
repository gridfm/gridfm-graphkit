from lightning.pytorch.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from lightning.pytorch.loggers import MLFlowLogger
import os
import torch
import shutil


class SaveBestModelStateDict(Callback):
    def __init__(
        self,
        monitor: str,
        mode: str = "min",
        filename: str = "best_model_state_dict",
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
        model_path = os.path.join(model_dir, f"{self.filename}.pt")

        # Check if this is the best score so far
        if (self.mode == "min" and current < self.best_score) or (
            self.mode == "max" and current > self.best_score
        ):
            self.best_score = current

            torch.save(pl_module.state_dict(), model_path)
            
        if (trainer.current_epoch+1) % 50 == 0 or (trainer.current_epoch+1) == 10:
                # copy model at model_path to model_path.epoch_<current_epoch>
            shutil.copy(model_path, os.path.join(model_dir, f"{self.filename}.epoch_{trainer.current_epoch}.pt"))
            
