from lightning.pytorch.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from lightning.pytorch.loggers import MLFlowLogger
import os
import torch
import torch.profiler


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


class TorchProfilerCallback(Callback):
    """
    Lightning callback that wraps :class:`torch.profiler.profile`.

    Profiles the first few training batches (``wait`` + ``warmup`` + ``active``
    steps, repeated ``repeat`` times) and exports a TensorBoard-compatible trace
    and a human-readable key-averages summary to *output_dir*.

    Args:
        output_dir: Directory where trace files are written (default: ``"profiler_output"``).
        wait: Steps to skip at the start of each profiling cycle.
        warmup: Warm-up steps (profiler runs but data is discarded).
        active: Steps that are actually recorded per cycle.
        repeat: Number of wait/warmup/active cycles (0 = run until training ends).
        with_stack: Whether to capture Python call stacks.
        row_limit: Number of rows in the key-averages summary table.
    """

    def __init__(
        self,
        output_dir: str = "profiler_output",
        wait: int = 1,
        warmup: int = 1,
        active: int = 3,
        repeat: int = 2,
        with_stack: bool = False,
        row_limit: int = 20,
    ):
        self.output_dir = output_dir
        self.wait = wait
        self.warmup = warmup
        self.active = active
        self.repeat = repeat
        self.with_stack = with_stack
        self.row_limit = row_limit
        self._prof = None

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        os.makedirs(self.output_dir, exist_ok=True)
        schedule = torch.profiler.schedule(
            wait=self.wait,
            warmup=self.warmup,
            active=self.active,
            repeat=self.repeat,
        )
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        self._prof = torch.profiler.profile(
            schedule=schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(self.output_dir),
            record_shapes=True,
            with_stack=self.with_stack,
            activities=activities,
        )
        self._prof.__enter__()

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._prof is not None:
            self._prof.step()

    @rank_zero_only
    def on_train_end(self, trainer, pl_module):
        if self._prof is not None:
            self._prof.__exit__(None, None, None)
            summary_path = os.path.join(self.output_dir, "profiler_summary.txt")
            with open(summary_path, "w") as f:
                f.write(
                    self._prof.key_averages().table(
                        sort_by="cpu_time_total",
                        row_limit=self.row_limit,
                    )
                )
            self._prof = None
