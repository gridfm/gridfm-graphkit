"""Train one GENCO on power-flow and OPF together.

Each training step gets one batch per task (mixed grids). Each batch is a
plain GENCO forward with that task's decoder and loss.
"""

import torch
from torch import nn

from gridfm_graphkit.io.param_handler import NestedNamespace, get_loss_function
from gridfm_graphkit.io.registries import TASK_REGISTRY
from gridfm_graphkit.tasks.opf_task import OptimalPowerFlowTask
from gridfm_graphkit.tasks.pf_task import PowerFlowTask
from gridfm_graphkit.tasks.reconstruction_tasks import ReconstructionTask

_PREFIX = {"PowerFlow": "PF", "OptimalPowerFlow": "OPF"}
_PF_LOSSES = ["LayeredWeightedPhysics", "MaskedBusMSE"]
_PF_WEIGHTS = [0.1, 0.9]
_PF_LOSS_ARGS = [{"base_weight": 0.5}, {}]
# Graph-weighted PF loss name for every name in _PF_LOSSES, used instead of
# _PF_LOSSES when the OPF-side config (``training.losses``) opts into the
# graph-weighted variants. Keeps the PF side's weighting family consistent
# with whatever the config specifies for OPF, without a second YAML field.
_PF_LOSSES_GRAPH_WEIGHTED = ["LayeredWeightedPhysics", "MaskedBusMSEGraphWeighted"]


def _loss_from_spec(args, names, weights, loss_arg_dicts):
    saved = (
        args.training.losses,
        args.training.loss_weights,
        args.training.loss_args,
    )
    args.training.losses = list(names)
    args.training.loss_weights = list(weights)
    args.training.loss_args = [NestedNamespace(**spec) for spec in loss_arg_dicts]
    try:
        return get_loss_function(args)
    finally:
        (
            args.training.losses,
            args.training.loss_weights,
            args.training.loss_args,
        ) = saved


@TASK_REGISTRY.register("PowerFlowAndOPF")
class PowerFlowAndOPFTask(ReconstructionTask):
    """Mixed power-flow and OPF training with per-task evaluation."""

    def __init__(self, args, data_normalizers):
        super().__init__(args, data_normalizers)
        self.tasks = list(args.data.tasks)
        opf_loss = self.loss_fn
        # The OPF-side config opts into the graph-weighted loss variants by
        # naming them in training.losses (e.g. "MaskedBusMSEGraphWeighted").
        # Mirror that choice on the PF side so both tasks use the same
        # weighting family; default (no "GraphWeighted" names) is unchanged.
        graph_weighted = any(
            name.endswith("GraphWeighted") for name in args.training.losses
        )
        pf_losses = _PF_LOSSES_GRAPH_WEIGHTED if graph_weighted else _PF_LOSSES
        pf_loss = _loss_from_spec(args, pf_losses, _PF_WEIGHTS, _PF_LOSS_ARGS)
        self.losses = nn.ModuleDict(
            {
                "PowerFlow": pf_loss,
                "OptimalPowerFlow": opf_loss,
            },
        )
        # Both losses are fixed weights, so this only keeps them on the module.
        self.loss_fn = self.losses
        # Validation loaders come one per task, in first-seen order.
        self.val_tasks = list(dict.fromkeys(self.tasks))

    def on_train_epoch_start(self):
        for loss in self.losses.values():
            loss.set_epoch(self.current_epoch)

    def _task_loss(self, batch, task_name):
        self.model.set_task(task_name)
        return self.shared_step(batch)

    def shared_step(self, batch):
        """Loss for the task last set on the model (PF/OPF test steps call this)."""
        output = self.model(batch)
        loss_dict = self.losses[self.model.task](
            output,
            batch.y_dict,
            batch.edge_index_dict,
            batch.edge_attr_dict,
            batch.mask_dict,
            model=self.model,
            x_dict=batch.x_dict,
            batch_dict={"bus": batch["bus"].batch, "gen": batch["gen"].batch},
        )
        return output, loss_dict

    def training_step(self, batch):
        # ``batch`` maps each task to a single-task batch. Each microbatch
        # gets weight 1 / number of tasks, then one optimizer step.
        num_graphs = sum(b.num_graphs for b in batch.values())
        losses = []
        for task_name, task_batch in batch.items():
            _, loss_dict = self._task_loss(task_batch, task_name)
            losses.append(loss_dict["loss"])
            self.log(
                f"Training {_PREFIX[task_name]} loss",
                loss_dict["loss"].detach(),
                batch_size=task_batch.num_graphs,
                on_step=True,
                on_epoch=False,
                logger=True,
            )
        loss = torch.stack(losses).mean()
        self.log(
            "Training Loss",
            loss.detach(),
            batch_size=num_graphs,
            on_step=True,
            on_epoch=False,
            logger=True,
        )
        self.log(
            "Learning Rate",
            self.optimizer.param_groups[0]["lr"],
            batch_size=num_graphs,
            on_step=True,
            on_epoch=False,
            logger=True,
        )
        return loss

    def on_validation_epoch_start(self):
        self._val_sum = torch.zeros(2, device=self.device)  # loss * graphs, graphs

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        task_name = self.val_tasks[dataloader_idx]
        _, loss_dict = self._task_loss(batch, task_name)
        loss_dict["loss"] = loss_dict["loss"].detach()
        n = loss_dict["loss"].new_tensor(batch.num_graphs)
        self._val_sum += torch.stack([loss_dict["loss"] * n, n])
        for metric, value in loss_dict.items():
            self.log(
                f"Validation {_PREFIX[task_name]} {metric}",
                value,
                batch_size=batch.num_graphs,
                sync_dist=True,
                on_epoch=True,
                on_step=False,
                add_dataloader_idx=False,
            )
        return loss_dict["loss"]

    def on_validation_epoch_end(self):
        # Lightning will not log one key from two loaders, so the monitored
        # loss is averaged here over every validation graph of both tasks.
        total = self.all_gather(self._val_sum)
        if total.ndim > 1:
            total = total.sum(dim=0)
        self.log("Validation loss", total[0] / total[1], on_epoch=True, logger=True)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        task_name = self.tasks[dataloader_idx]
        self.model.set_task(task_name)
        if task_name == "PowerFlow":
            return PowerFlowTask.test_step(self, batch, batch_idx, dataloader_idx)
        return OptimalPowerFlowTask.test_step(self, batch, batch_idx, dataloader_idx)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        task_name = self.tasks[dataloader_idx]
        self.model.set_task(task_name)
        if task_name == "PowerFlow":
            return PowerFlowTask.predict_step(self, batch, batch_idx, dataloader_idx)
        return OptimalPowerFlowTask.predict_step(
            self,
            batch,
            batch_idx,
            dataloader_idx,
        )

    def on_test_end(self):
        saved_outputs = {index: list(rows) for index, rows in self.test_outputs.items()}
        metrics = self.trainer.callback_metrics
        saved_metrics = dict(metrics)

        def _run(task_cls, task_name):
            prefixes = {
                name
                for name, task in zip(self.args.data.networks, self.tasks)
                if task == task_name
            }
            indexes = [i for i, task in enumerate(self.tasks) if task == task_name]
            metrics.clear()
            for key, value in saved_metrics.items():
                if "/" not in key or key.split("/", 1)[0] in prefixes:
                    metrics[key] = value
            self.test_outputs = {
                index: saved_outputs.get(index, []) for index in indexes
            }
            task_cls.on_test_end(self)

        _run(PowerFlowTask, "PowerFlow")
        _run(OptimalPowerFlowTask, "OptimalPowerFlow")
        metrics.clear()
        metrics.update(saved_metrics)
