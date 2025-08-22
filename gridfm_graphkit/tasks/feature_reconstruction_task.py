import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import lightning as L
from pytorch_lightning.utilities import rank_zero_only
import numpy as np
import os
import pandas as pd

from lightning.pytorch.loggers import MLFlowLogger
from gridfm_graphkit.io.param_handler import load_model, get_loss_function
import torch.nn.functional as F
from gridfm_graphkit.datasets.globals import PQ, PV, REF, PD, QD, PG, QG, VM, VA


class FeatureReconstructionTask(L.LightningModule):
    """
    PyTorch Lightning task for node feature reconstruction on power grid graphs.

    This task wraps a GridFM model inside a LightningModule and defines the full
    training, validation, testing, and prediction logic. It is designed to
    reconstruct masked node features from graph-structured input data, using
    datasets and normalizers provided by `gridfm-graphkit`.

    Args:
        args (NestedNamespace): Experiment configuration. Expected fields include `training.batch_size`, `optimizer.*`, etc.
        node_normalizers (list): One normalizer per dataset to (de)normalize node features.
        edge_normalizers (list): One normalizer per dataset to (de)normalize edge features.

    Attributes:
        model (torch.nn.Module): model loaded via `load_model`.
        loss_fn (callable): Loss function resolved from configuration.
        batch_size (int): Training batch size. From ``args.training.batch_size``
        node_normalizers (list): Dataset-wise node feature normalizers.
        edge_normalizers (list): Dataset-wise edge feature normalizers.

    Methods:
        forward(x, pe, edge_index, edge_attr, batch, mask=None):
            Forward pass with optional feature masking.
        training_step(batch):
            One training step: computes loss, logs metrics, returns loss.
        validation_step(batch, batch_idx):
            One validation step: computes losses and logs metrics.
        test_step(batch, batch_idx, dataloader_idx=0):
            Evaluate on test data, compute per-node-type MSEs, and log per-dataset metrics.
        predict_step(batch, batch_idx, dataloader_idx=0):
            Run inference and return denormalized outputs + node masks.
        configure_optimizers():
            Setup Adam optimizer and ReduceLROnPlateau scheduler.
        on_fit_start():
            Save normalization statistics at the beginning of training.
        on_test_end():
            Collect test metrics across datasets and export summary CSV reports.

    Notes:
        - Node types are distinguished using the global constants (`PQ`, `PV`, `REF`).
        - The datamodule must provide `batch.mask` for masking node features.
        - Test metrics include per-node-type RMSE for [Pd, Qd, Pg, Qg, Vm, Va].
        - Reports are saved under `<mlflow_artifacts>/test/<dataset>.csv`.

    Example:
        ```python
        model = FeatureReconstructionTask(args, node_normalizers, edge_normalizers)
        output = model(batch.x, batch.pe, batch.edge_index, batch.edge_attr, batch.batch)
        ```
    """

    def __init__(self, args, node_normalizers, edge_normalizers):
        super().__init__()
        self.model = load_model(args=args)
        self.args = args
        self.loss_fn = get_loss_function(args)
        self.batch_size = int(args.training.batch_size)
        self.node_normalizers = node_normalizers
        self.edge_normalizers = edge_normalizers
        self.save_hyperparameters()

    def forward(self, x, pe, edge_index, edge_attr, batch, mask=None):
        if mask is not None:
            mask_value_expanded = self.model.mask_value.expand(x.shape[0], -1)
            x[:, : mask.shape[1]][mask] = mask_value_expanded[mask]
        return self.model(x, pe, edge_index, edge_attr, batch)

    @rank_zero_only
    def on_fit_start(self):
        # Determine save path
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

        # Collect normalization stats
        with open(log_stats_path, "w") as log_file:
            for i, normalizer in enumerate(self.node_normalizers):
                log_file.write(
                    f"Node Normalizer {self.args.data.networks[i]} stats:\n{normalizer.get_stats()}\n\n",
                )

            for i, normalizer in enumerate(self.edge_normalizers):
                log_file.write(
                    f"Edge Normalizer {self.args.data.networks[i]} stats:\n{normalizer.get_stats()}\n\n",
                )

    def shared_step(self, batch):
        output = self.forward(
            x=batch.x,
            pe=batch.pe,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            batch=batch.batch,
            mask=batch.mask,
        )

        loss_dict = self.loss_fn(
            output,
            batch.y,
            batch.edge_index,
            batch.edge_attr,
            batch.mask,
        )
        return output, loss_dict

    def training_step(self, batch):
        _, loss_dict = self.shared_step(batch)
        current_lr = self.optimizer.param_groups[0]["lr"]
        metrics = {}
        metrics["Training Loss"] = loss_dict["loss"].detach()
        metrics["Learning Rate"] = current_lr
        for metric, value in metrics.items():
            self.log(
                metric,
                value,
                batch_size=batch.num_graphs,
                sync_dist=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                on_step=False,
            )

        return loss_dict["loss"]

    def validation_step(self, batch, batch_idx):
        _, loss_dict = self.shared_step(batch)
        loss_dict["loss"] = loss_dict["loss"].detach()
        for metric, value in loss_dict.items():
            metric_name = f"Validation {metric}"
            self.log(
                metric_name,
                value,
                batch_size=batch.num_graphs,
                sync_dist=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                on_step=False,
            )

        return loss_dict["loss"]

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        output, loss_dict = self.shared_step(batch)

        dataset_name = self.args.data.networks[dataloader_idx]

        output_denorm = self.node_normalizers[dataloader_idx].inverse_transform(output)
        target_denorm = self.node_normalizers[dataloader_idx].inverse_transform(batch.y)

        mask_PQ = batch.x[:, PQ] == 1
        mask_PV = batch.x[:, PV] == 1
        mask_REF = batch.x[:, REF] == 1

        mse_PQ = F.mse_loss(
            output_denorm[mask_PQ],
            target_denorm[mask_PQ],
            reduction="none",
        )
        mse_PV = F.mse_loss(
            output_denorm[mask_PV],
            target_denorm[mask_PV],
            reduction="none",
        )
        mse_REF = F.mse_loss(
            output_denorm[mask_REF],
            target_denorm[mask_REF],
            reduction="none",
        )

        mse_PQ = mse_PQ.mean(dim=0)
        mse_PV = mse_PV.mean(dim=0)
        mse_REF = mse_REF.mean(dim=0)

        loss_dict["MSE PQ nodes - PD"] = mse_PQ[PD]
        loss_dict["MSE PV nodes - PD"] = mse_PV[PD]
        loss_dict["MSE REF nodes - PD"] = mse_REF[PD]

        loss_dict["MSE PQ nodes - QD"] = mse_PQ[QD]
        loss_dict["MSE PV nodes - QD"] = mse_PV[QD]
        loss_dict["MSE REF nodes - QD"] = mse_REF[QD]

        loss_dict["MSE PQ nodes - PG"] = mse_PQ[PG]
        loss_dict["MSE PV nodes - PG"] = mse_PV[PG]
        loss_dict["MSE REF nodes - PG"] = mse_REF[PG]

        loss_dict["MSE PQ nodes - QG"] = mse_PQ[QG]
        loss_dict["MSE PV nodes - QG"] = mse_PV[QG]
        loss_dict["MSE REF nodes - QG"] = mse_REF[QG]

        loss_dict["MSE PQ nodes - VM"] = mse_PQ[VM]
        loss_dict["MSE PV nodes - VM"] = mse_PV[VM]
        loss_dict["MSE REF nodes - VM"] = mse_REF[VM]

        loss_dict["MSE PQ nodes - VA"] = mse_PQ[VA]
        loss_dict["MSE PV nodes - VA"] = mse_PV[VA]
        loss_dict["MSE REF nodes - VA"] = mse_REF[VA]

        loss_dict["Test loss"] = loss_dict.pop("loss").detach()
        for metric, value in loss_dict.items():
            metric_name = f"{dataset_name}/{metric}"
            if "p.u." in metric:
                # Denormalize metrics expressed in p.u.
                value *= self.node_normalizers[dataloader_idx].baseMVA
                metric_name = metric_name.replace("in p.u.", "").strip()
            self.log(
                metric_name,
                value,
                batch_size=batch.num_graphs,
                add_dataloader_idx=False,
                sync_dist=True,
                logger=False,
            )
        return

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        output, _ = self.shared_step(batch)
        output_denorm = self.node_normalizers[dataloader_idx].inverse_transform(output)

        # Masks for node types
        mask_PQ = (batch.x[:, PQ] == 1).cpu()
        mask_PV = (batch.x[:, PV] == 1).cpu()
        mask_REF = (batch.x[:, REF] == 1).cpu()

        # Count buses and generate per-node scenario_id
        bus_counts = batch.batch.unique(return_counts=True)[1]
        scenario_ids = batch.scenario_id  # shape: [num_graphs]
        scenario_per_node = torch.cat(
            [
                torch.full((count,), sid, dtype=torch.int32)
                for count, sid in zip(bus_counts, scenario_ids)
            ],
        )

        bus_numbers = np.concatenate([np.arange(count.item()) for count in bus_counts])

        return {
            "output": output_denorm.cpu().numpy(),
            "mask_PQ": mask_PQ,
            "mask_PV": mask_PV,
            "mask_REF": mask_REF,
            "scenario_id": scenario_per_node,
            "bus_number": bus_numbers,
        }

    @rank_zero_only
    def on_test_end(self):
        if isinstance(self.logger, MLFlowLogger):
            artifact_dir = os.path.join(
                self.logger.save_dir,
                self.logger.experiment_id,
                self.logger.run_id,
                "artifacts",
            )
        else:
            artifact_dir = self.logger.save_dir

        final_metrics = self.trainer.callback_metrics
        grouped_metrics = {}

        for full_key, value in final_metrics.items():
            try:
                value = value.item()
            except AttributeError:
                pass

            if "/" in full_key:
                dataset_name, metric = full_key.split("/", 1)
                if dataset_name not in grouped_metrics:
                    grouped_metrics[dataset_name] = {}
                grouped_metrics[dataset_name][metric] = value

        for dataset, metrics in grouped_metrics.items():
            rmse_PQ = [
                metrics.get(f"MSE PQ nodes - {label}", float("nan")) ** 0.5
                for label in ["PD", "QD", "PG", "QG", "VM", "VA"]
            ]
            rmse_PV = [
                metrics.get(f"MSE PV nodes - {label}", float("nan")) ** 0.5
                for label in ["PD", "QD", "PG", "QG", "VM", "VA"]
            ]
            rmse_REF = [
                metrics.get(f"MSE REF nodes - {label}", float("nan")) ** 0.5
                for label in ["PD", "QD", "PG", "QG", "VM", "VA"]
            ]

            avg_active_res = metrics.get("Active Power Loss", " ")
            avg_reactive_res = metrics.get("Reactive Power Loss", " ")

            data = {
                "Metric": [
                    "RMSE-PQ",
                    "RMSE-PV",
                    "RMSE-REF",
                    "Avg. active res. (MW)",
                    "Avg. reactive res. (MVar)",
                ],
                "Pd (MW)": [
                    rmse_PQ[0],
                    rmse_PV[0],
                    rmse_REF[0],
                    avg_active_res,
                    avg_reactive_res,
                ],
                "Qd (MVar)": [rmse_PQ[1], rmse_PV[1], rmse_REF[1], " ", " "],
                "Pg (MW)": [rmse_PQ[2], rmse_PV[2], rmse_REF[2], " ", " "],
                "Qg (MVar)": [rmse_PQ[3], rmse_PV[3], rmse_REF[3], " ", " "],
                "Vm (p.u.)": [rmse_PQ[4], rmse_PV[4], rmse_REF[4], " ", " "],
                "Va (degree)": [rmse_PQ[5], rmse_PV[5], rmse_REF[5], " ", " "],
            }

            df = pd.DataFrame(data)

            test_dir = os.path.join(artifact_dir, "test")
            os.makedirs(test_dir, exist_ok=True)
            csv_path = os.path.join(test_dir, f"{dataset}.csv")
            df.to_csv(csv_path, index=False)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(
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
        config_optim = {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "monitor": "Validation loss",
                "reduce_on_plateau": True,
            },
        }
        return config_optim
