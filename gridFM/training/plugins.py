from abc import abstractmethod
from typing import Dict, Optional
import mlflow
import os
import torch


class TrainerPlugin:
    """
    A TrainerPlugin runs once every epoch, and possibly every `steps` steps.
    It is passed relevant objects that can be used for checkpointing, logging,
    or validation.
    """

    def __init__(self, steps: Optional[int] = None):
        self.steps = steps

    def run(self, step: int, end_of_epoch: bool):
        """
        Whether or not to run this plugin on the current step.
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
        model: The model being trained.
        step: The step in training, re-starting from zero each epoch. None at
                 epoch end.
        metrics: a dictionary of metrics that might be useful for
                logging/reporting. E.g. 'loss'. Specific metrics subject
                to change.
        """
        pass


class MLflowLoggerPlugin(TrainerPlugin):
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
        model=None,
        optimizer=None,
        scheduler=None,
    ):
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
            self.checkpoint_dir, f"checkpoint_last_epoch.pth"
        )
        torch.save(checkpoint, checkpoint_path)