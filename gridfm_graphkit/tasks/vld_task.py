import os
import torch
import torch.distributed as dist
import pandas as pd
from lightning.pytorch.loggers import MLFlowLogger

from gridfm_graphkit.io.registries import TASK_REGISTRY
from gridfm_graphkit.tasks.reconstruction_tasks import ReconstructionTask
from gridfm_graphkit.datasets.globals import (
    VM_OUT,
    VM_H,
    BUS_STATUS_TARGET,
    BUS_STATUS_LOGIT_OUT,
)

@TASK_REGISTRY.register("VoltageLossDetection")
class VoltageLossDetectionTask(ReconstructionTask):
    """
    Topology-aware voltage loss detection task.

    Uses the standard ReconstructionTask training/validation flow and adds
    VLD-specific test/predict metrics for bus status and Vm behavior.
    """

    def __init__(self, args, data_normalizers):
        super().__init__(args, data_normalizers)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        output, loss_dict = self.shared_step(batch)
        dataset_name = self.args.data.networks[dataloader_idx]

        bus_pred = output["bus"]
        bus_target = batch.y_dict["bus"]

        status_prob = torch.sigmoid(bus_pred[:, BUS_STATUS_LOGIT_OUT])
        status_pred = (status_prob >= 0.5).float()
        status_true = bus_target[:, BUS_STATUS_TARGET].float()

        vm_pred = bus_pred[:, VM_OUT]
        vm_true = bus_target[:, VM_H]

        status_acc = (status_pred == status_true).float().mean()

        off_mask = status_true < 0.5
        on_mask = status_true >= 0.5

        off_vm_mae = (
            vm_pred[off_mask].abs().mean()
            if off_mask.any()
            else torch.tensor(0.0, device=vm_pred.device)
        )
        on_vm_rmse = (
            torch.sqrt(torch.mean((vm_pred[on_mask] - vm_true[on_mask]) ** 2))
            if on_mask.any()
            else torch.tensor(0.0, device=vm_pred.device)
        )

        loss_dict["Status Accuracy"] = status_acc.detach()
        loss_dict["OFF Vm MAE"] = off_vm_mae.detach()
        loss_dict["ON Vm RMSE"] = on_vm_rmse.detach()

        loss_dict["Test loss"] = loss_dict.pop("loss").detach()

        for metric, value in loss_dict.items():
            metric_name = f"{dataset_name}/{metric}"
            self.log(
                metric_name,
                value,
                batch_size=batch.num_graphs,
                add_dataloader_idx=False,
                sync_dist=True,
                logger=False,
            )

        self.test_outputs[dataloader_idx].append(
            {
                "dataset": dataset_name,
                "status_prob": status_prob.detach().cpu(),
                "status_pred": status_pred.detach().cpu(),
                "status_true": status_true.detach().cpu(),
                "vm_pred": vm_pred.detach().cpu(),
                "vm_true": vm_true.detach().cpu(),
            }
        )

    def on_test_end(self):
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            gathered = [None] * world_size if dist.get_rank() == 0 else None
            dist.gather_object(self.test_outputs, gathered, dst=0)
            if dist.get_rank() == 0:
                merged = {i: [] for i in range(len(self.args.data.networks))}
                for rank_data in gathered:
                    for dl_idx, batches in rank_data.items():
                        merged[dl_idx].extend(batches)
                self.test_outputs = merged

        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return

        if isinstance(self.logger, MLFlowLogger):
            artifact_dir = os.path.join(
                self.logger.save_dir,
                self.logger.experiment_id,
                self.logger.run_id,
                "artifacts",
            )
        else:
            artifact_dir = self.logger.save_dir

        test_dir = os.path.join(artifact_dir, "test")
        os.makedirs(test_dir, exist_ok=True)

        for dataset_idx, outputs in self.test_outputs.items():
            if not outputs:
                continue

            dataset_name = self.args.data.networks[dataset_idx]
            status_prob = torch.cat([o["status_prob"] for o in outputs]).numpy()
            status_pred = torch.cat([o["status_pred"] for o in outputs]).numpy()
            status_true = torch.cat([o["status_true"] for o in outputs]).numpy()
            vm_pred = torch.cat([o["vm_pred"] for o in outputs]).numpy()
            vm_true = torch.cat([o["vm_true"] for o in outputs]).numpy()

            df = pd.DataFrame(
                {
                    "status_prob": status_prob,
                    "status_pred": status_pred,
                    "status_true": status_true,
                    "vm_pred": vm_pred,
                    "vm_true": vm_true,
                }
            )
            df.to_csv(os.path.join(test_dir, f"{dataset_name}_vld_predictions.csv"), index=False)

        self.test_outputs.clear()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        output, _ = self.shared_step(batch)

        bus_pred = output["bus"]
        status_prob = torch.sigmoid(bus_pred[:, BUS_STATUS_LOGIT_OUT])

        bus_batch = batch.batch_dict["bus"]
        scenario_ids = batch["scenario_id"][bus_batch]

        local_bus_idx = torch.cat(
            [torch.arange(c, device=bus_batch.device) for c in torch.bincount(bus_batch)]
        )

        return {
            "scenario": scenario_ids.cpu().numpy(),
            "bus": local_bus_idx.cpu().numpy(),
            "vm_pred": bus_pred[:, VM_OUT].detach().cpu().numpy(),
            "status_prob": status_prob.detach().cpu().numpy(),
            "status_pred": (status_prob >= 0.5).float().detach().cpu().numpy(),
            "status_true": batch.y_dict["bus"][:, BUS_STATUS_TARGET].detach().cpu().numpy(),
        }