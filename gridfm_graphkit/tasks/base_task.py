import os
import time
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

    def on_after_batch_transfer(self, batch, dataloader_idx: int):
        """Cast float tensors in HeteroData batches to the model's parameter dtype.

        Lightning's automatic mixed-precision casting does not handle PyG
        HeteroData objects, so we do it manually here to avoid dtype mismatches
        when --bfloat16 (precision='bf16-true') is used.
        """
        if not hasattr(self, "model"):
            return batch
        try:
            target_dtype = next(self.model.parameters()).dtype
        except StopIteration:
            return batch
        if target_dtype == torch.float32:
            # No casting needed for the default precision.
            return batch
        # Walk all node- and edge-store tensors in a HeteroData/Data object.
        for store in batch.stores:
            for key, val in store.items():
                if isinstance(val, torch.Tensor) and val.is_floating_point():
                    store[key] = val.to(target_dtype)
        return batch

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Forward pass"""
        pass

    def on_train_batch_start(self, batch, batch_idx):
        self._batch_start_time = time.perf_counter()

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

        # Human-readable log
        log_stats_path = os.path.join(log_dir, "normalization_stats.txt")
        with open(log_stats_path, "w") as log_file:
            for i, normalizer in enumerate(self.data_normalizers):
                log_file.write(
                    f"Data Normalizer {self.args.data.networks[i]} stats:\n{normalizer.get_stats()}\n\n",
                )

        # Machine-loadable stats (one file per network, keyed by network name)
        stats_dict = {}
        for i, normalizer in enumerate(self.data_normalizers):
            stats_dict[self.args.data.networks[i]] = normalizer.get_stats()
        torch.save(stats_dict, os.path.join(log_dir, "normalizer_stats.pt"))

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
