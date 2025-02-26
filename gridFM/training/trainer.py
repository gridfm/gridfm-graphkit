from gridFM.training.plugins import TrainerPlugin
from gridFM.training.callbacks import EarlyStopper

from typing import List
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        device,
        loss_fn: nn.Module,
        early_stopper: EarlyStopper,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader = None,
        lr_scheduler=None,
        plugins: List[TrainerPlugin] = [],
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.early_stopper = early_stopper
        self.loss_fn = loss_fn
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.lr_scheduler = lr_scheduler
        self.plugins = plugins

    def __one_step(
        self,
        input: torch.Tensor,
        edge_index: torch.Tensor,
        label: torch.Tensor,
        edge_attr: torch.Tensor,
        mask: torch.Tensor = None,
        batch: torch.Tensor = None,
        pe: torch.Tensor = None,
        val: bool = False,
    ):

        # expand the learnable mask to the input shape
        mask_value_expanded = self.model.mask_value.expand(input.shape[0], -1)
        # The line below will overwrite the last mask values, which is fine as long as the features which are masked do not change between batches
        # set the learnable mask to the inout where it should be masked
        input[:, : mask.shape[1]][mask] = mask_value_expanded[mask]
        output = self.model(input, pe, edge_index, edge_attr, batch)

        loss_dict = self.loss_fn(output, label, edge_index, edge_attr, mask)

        if not val:
            self.optimizer.zero_grad()
            loss_dict["loss"].backward()
            self.optimizer.step()

        return loss_dict

    def __one_epoch(self, epoch: int, prev_step: int):
        self.model.train()

        highest_step = prev_step
        for step, batch in enumerate(self.train_dataloader):
            step = prev_step + step + 1
            highest_step = step
            batch = batch.to(self.device)

            mask = getattr(batch, "mask", None)

            loss_dict = self.__one_step(
                batch.x,
                batch.edge_index,
                batch.y,
                batch.edge_attr,
                mask,
                batch.batch,
                batch.pe,
            )
            current_lr = self.optimizer.param_groups[0]["lr"]
            metrics = {}
            metrics["Training Loss"] = loss_dict["loss"].item()
            metrics["Learning Rate"] = current_lr

            if self.model.learn_mask:
                metrics["Mask Gradient Norm"] = self.model.mask_value.grad.norm().item()

            for plugin in self.plugins:
                plugin.step(epoch, step, metrics=metrics)

        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in self.val_dataloader:
                batch = batch.to(self.device)
                mask = getattr(batch, "mask", None)
                metrics = self.__one_step(
                    batch.x,
                    batch.edge_index,
                    batch.y,
                    batch.edge_attr,
                    mask,
                    batch.batch,
                    batch.pe,
                    True,
                )
                val_loss += metrics["loss"].item()
                metrics["Validation Loss"] = metrics.pop("loss").item()

                for plugin in self.plugins:
                    plugin.step(epoch, step, metrics=metrics)
        val_loss /= len(self.val_dataloader)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(val_loss)
        for plugin in self.plugins:
            plugin.step(
                epoch,
                step=highest_step,
                end_of_epoch=True,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.lr_scheduler,
            )
        return val_loss

    def train(self, start_epoch: int = 0, epochs: int = 1, prev_step: int = -1):
        for epoch in tqdm(range(start_epoch, start_epoch + epochs), desc="Epochs"):
            val_loss = self.__one_epoch(epoch, prev_step)
            if self.early_stopper.early_stop(val_loss, self.model):
                break
