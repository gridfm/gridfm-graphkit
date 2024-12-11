from abc import abstractmethod
from typing import Dict, Optional
import mlflow


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
        self, epoch: int, step: int, metrics: Dict = {}, end_of_epoch: bool = False
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
        self.train_loss = []
        self.val_loss = []
        self.mask_grad_norm = []
        self.current_lr = None
        if params:
            # Log parameters to MLflow at the beginning of training
            mlflow.log_params(params)

    def step(
        self, epoch: int, step: int, metrics: Dict = {}, end_of_epoch: bool = False
    ):
        """
        Logs metrics to MLflow at each specified step and at the end of each epoch.

        Args:
            epoch (int): The current epoch number.
            step (int): The current step within the epoch.
            metrics (Dict): Dictionary of metrics to log, e.g., {'train_loss': value}.
            end_of_epoch (bool): Flag indicating whether this is the end of the epoch.
        """
        if "train_loss" in metrics:
            self.train_loss.append(metrics["train_loss"])
            self.current_lr = metrics["learning_rate"]
        if "val_loss" in metrics:
            self.val_loss.append(metrics["val_loss"])
        if "mask_grad_norm" in metrics:
            self.mask_grad_norm.append(metrics["mask_grad_norm"])

        if end_of_epoch:
            mlflow.log_metric(
                "train_loss_epoch",
                sum(self.train_loss) / len(self.train_loss),
                step=epoch,
            )
            mlflow.log_metric(
                "val_loss_epoch",
                sum(self.val_loss) / len(self.val_loss),
                step=epoch,
            )
            mlflow.log_metric("learning_rate", self.current_lr, step=epoch)
            if len(self.mask_grad_norm) > 0:
                mlflow.log_metric(
                "mask_grad_norm",
                sum(self.mask_grad_norm) / len(self.mask_grad_norm),
                step=epoch,
            )
            self.train_loss = []
            self.val_loss = []
            self.mask_grad_norm = []
