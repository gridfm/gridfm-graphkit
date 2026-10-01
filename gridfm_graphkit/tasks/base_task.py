import os
from abc import ABC, abstractmethod
import lightning as L
from pytorch_lightning.utilities import rank_zero_only
from lightning.pytorch.loggers import MLFlowLogger
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau


class BaseTask(L.LightningModule, ABC):
    """
    Abstract base LightningModule for feature reconstruction tasks.
    Contains shared training/validation/test logic, logging, and optimizer setup.
    """

    def __init__(self, args, data_normalizers):
        super().__init__()
        self.args = args
        self.data_normalizers = data_normalizers
        self.save_hyperparameters()

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """Pre-cast float64 tensors before moving batches onto MPS.

        PyTorch MPS does not support float64 tensors. Some PyG metadata fields can
        get collated as float64 even when model inputs are float32, so coerce them
        first and then delegate to Lightning's standard device transfer.
        """
        if getattr(device, "type", None) == "mps" and hasattr(batch, "stores"):
            for store in batch.stores:
                for key, val in store.items():
                    if isinstance(val, torch.Tensor) and val.dtype == torch.float64:
                        store[key] = val.to(torch.float32)
        return super().transfer_batch_to_device(batch, device, dataloader_idx)

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Forward pass"""
        pass

    @abstractmethod
    def training_step(self, batch):
        pass

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        pass

    @abstractmethod
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        pass

    @abstractmethod
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        pass

    @rank_zero_only
    def on_fit_start(self):
        # Save normalization statistics
        if isinstance(self.logger, MLFlowLogger):
            log_dir = os.path.join(
                self.logger.save_dir,
                self.logger.experiment_id,
                self.logger.run_id,
                "artifacts",
                "stats",
            )
        else:
            log_dir = os.path.join(self.logger.save_dir, "stats")

        os.makedirs(log_dir, exist_ok=True)
        log_stats_path = os.path.join(log_dir, "normalization_stats.txt")
        with open(log_stats_path, "w") as log_file:
            for i, normalizer in enumerate(self.data_normalizers):
                # For task-based structure, use generic identifier
                normalizer_name = getattr(self.args.data, "task", f"normalizer_{i}")
                log_file.write(
                    f"Data Normalizer {normalizer_name} stats:\n{normalizer.get_stats()}\n\n",
                )

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.optimizer.learning_rate,
            betas=(self.args.optimizer.beta1, self.args.optimizer.beta2),
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.args.optimizer.lr_decay,
            patience=self.args.optimizer.lr_patience,
        )
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "monitor": "Validation loss",
                "reduce_on_plateau": True,
            },
        }
