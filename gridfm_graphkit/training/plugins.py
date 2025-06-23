from abc import abstractmethod
from typing import Dict, Optional
import mlflow
import os
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class TrainerPlugin:
    """
    Base class for training plugins.

    A `TrainerPlugin` is invoked during the training process either at regular step intervals,
    at the end of each epoch, or both. It can be extended to perform actions like logging,
    checkpointing, or validation.

    Args:
        steps (int, optional): Interval (in steps) to run the plugin. If `None`, only runs at end of epoch
    """

    def __init__(self, steps: Optional[int] = None):
        self.steps = steps

    def run(self, step: int, end_of_epoch: bool) -> bool:
        """
        Determines whether to execute the plugin at the current step.

        Args:
            step (int): The current step number.
            end_of_epoch (bool): Whether this is the end of the epoch.

        Returns:
            bool: True if the plugin should run; False otherwise.
        """
        # By default we always run for epoch ends.
        if end_of_epoch:
            return True
        # If self.steps is None, we're only recording epoch ends and this isn't one.
        if self.steps is None:
            return False
        # record every `step` steps, starting from step `step`
        if step != 0 and (step + 1) % self.steps == 0:
            return True
        return False

    @abstractmethod
    def step(
        self,
        epoch: int,
        step: int,
        metrics: Dict = {},
        end_of_epoch: bool = False,
        **kwargs,
    ):
        """
        This method is called on every step of training, or with step=None
        at the end of each epoch. Implementations can use the passed in
        parameters for validation, checkpointing, logging, etc.

        Args:
            epoch (int): The current epoch number.
            step (int): The current step within the epoch.
            metrics (dict): Dictionary of training metrics (e.g., loss).
            end_of_epoch (bool): Indicates if this call is at the end of an epoch.
            **kwargs (Any): Additional parameters such as model, optimizer, scheduler.
        """
        pass


class MLflowLoggerPlugin(TrainerPlugin):
    """
    Plugin to log training metrics to MLflow.

    Logs metrics dynamically during training at defined step intervals and/or
    at the end of each epoch. Also logs initial training parameters once.

    Args:
        steps (int, optional): Interval in steps to log metrics.
        params (dict, optional): Parameters to log to MLflow at the start.
    """

    def __init__(self, steps: Optional[int] = None, params: dict = None):
        super().__init__(steps=steps)  # Initialize the steps from the base class
        self.steps = steps
        self.metrics_history = {}  # Dictionary to hold lists of all metrics over time
        if params:
            # Log parameters to MLflow at the beginning of training
            mlflow.log_params(params)

    def step(
        self,
        epoch: int,
        step: int,
        metrics: Dict = {},
        end_of_epoch: bool = False,
        **kwargs,
    ):
        """
        Logs metrics to MLflow dynamically at each specified step and at the end of each epoch.

        Args:
            epoch (int): The current epoch number.
            step (int): The current step within the epoch.
            metrics (Dict): Dictionary of metrics to log, e.g., {'train_loss': value}.
            end_of_epoch (bool): Flag indicating whether this is the end of the epoch.
        """
        for metric_name, metric_value in metrics.items():
            # Add metric to history
            if metric_name not in self.metrics_history:
                self.metrics_history[metric_name] = []
            self.metrics_history[metric_name].append(metric_value)

        if end_of_epoch:
            for metric_name, values in self.metrics_history.items():
                if values:  # Avoid division by zero or empty lists
                    avg_value = sum(values) / len(values)
                    mlflow.log_metric(f"{metric_name}", avg_value, step=epoch)

            # Clear metrics for the next epoch
            self.metrics_history = {}


class CheckpointerPlugin(TrainerPlugin):
    """
    Plugin to periodically save model checkpoints.

    Stores the model, optimizer, and scheduler states to a given directory
    at specified step intervals or at the end of each epoch.

    Args:
        checkpoint_dir (str): Directory where checkpoints will be saved.
        steps (int, optional): Interval in steps for checkpointing.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        steps: Optional[int] = None,
    ):
        super().__init__(steps=steps)
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def step(
        self,
        epoch: int,
        step: int,
        metrics: Dict = {},
        end_of_epoch: bool = False,
        model: Optional[nn.Module] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[LRScheduler] = None,
    ):
        """
        Saves a checkpoint if the conditions to run the plugin are met.

        Args:
            epoch (int): Current epoch number.
            step (int): Current training step.
            metrics (dict): Optional metrics dictionary (unused here).
            end_of_epoch (bool): Whether this is the end of the epoch.
            model (nn.Module, optional): Model to be checkpointed.
            optimizer (Optimizer, optional): Optimizer to save.
            scheduler (LRScheduler, optional): Scheduler to save.
        """
        # Check if we should save at this step or end of epoch
        if not self.run(step, end_of_epoch):
            return

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict() if model else None,
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        }

        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            "checkpoint_last_epoch.pth",
        )
        torch.save(checkpoint, checkpoint_path)


class MetricsTrackerPlugin(TrainerPlugin):
    """
    Logs metrics at the end of each epoch. Currently only returning the validation loss.
    """

    def __init__(self):
        super().__init__()
        self.validation_losses = []
        self.metrics_history = {}

    def step(
        self,
        epoch: int,
        step: int,
        metrics: Dict = {},
        end_of_epoch: bool = False,
        **kwargs,
    ):
        for metric_name, metric_value in metrics.items():
            # Add metric to history
            if metric_name not in self.metrics_history:
                self.metrics_history[metric_name] = []
            self.metrics_history[metric_name].append(metric_value)

        if end_of_epoch:
            for metric_name, values in self.metrics_history.items():
                if values:  # Avoid division by zero or empty lists
                    avg_value = sum(values) / len(values)
                    if metric_name == "Validation Loss":
                        self.validation_losses.append(avg_value)

    def get_losses(self):
        return self.validation_losses
