from gridfm_graphkit.datasets.globals import (
    # Bus feature indices
    PD_H,
    QD_H,
    QG_H,
    VM_H,
    VA_H,
    # Output feature indices
    VM_OUT,
    VA_OUT,
    PG_OUT,
    QG_OUT,
)

from gridfm_graphkit.tasks.reconstruction_tasks import ReconstructionTask
from gridfm_graphkit.io.registries import TASK_REGISTRY
from gridfm_graphkit.tasks.utils import (
    plot_correlation_by_node_type,
    plot_residuals_histograms,
    residual_stats_by_type,
)
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn import global_mean_pool
from gridfm_graphkit.models.utils import (
    ComputeBranchFlow,
    ComputeNodeInjection,
    ComputeNodeResiduals,
)
from lightning.pytorch.loggers import MLFlowLogger
import os
import pandas as pd


@TASK_REGISTRY.register("PowerFlow")
class PowerFlowTask(ReconstructionTask):
    """
    Concrete Optimal Power Flow task.
    Extends ReconstructionTask and adds OPF-specific metrics.
    """

    def __init__(self, args, data_normalizers):
        super().__init__(args, data_normalizers)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        output, loss_dict = self.shared_step(batch)
        dataset_name = self.args.data.networks[dataloader_idx]

        self.data_normalizers[dataloader_idx].inverse_transform(batch)
        self.data_normalizers[dataloader_idx].inverse_output(output, batch)

        branch_flow_layer = ComputeBranchFlow()
        node_injection_layer = ComputeNodeInjection()
        node_residuals_layer = ComputeNodeResiduals()

        num_bus = batch.x_dict["bus"].size(0)
        bus_edge_index = batch.edge_index_dict[("bus", "connects", "bus")]
        bus_edge_attr = batch.edge_attr_dict[("bus", "connects", "bus")]
        _, gen_to_bus_index = batch.edge_index_dict[("gen", "connected_to", "bus")]

        agg_gen_on_bus = scatter_add(
            batch.y_dict["gen"],
            gen_to_bus_index,
            dim=0,
            dim_size=num_bus,
        )
        # output_agg = torch.cat([batch.y_dict["bus"], agg_gen_on_bus], dim=1)
        target = torch.stack(
            [
                batch.y_dict["bus"][:, VM_H],
                batch.y_dict["bus"][:, VA_H],
                agg_gen_on_bus.squeeze(),
                batch.y_dict["bus"][:, QG_H],
            ],
            dim=1,
        )

        # UN-COMMENT THIS TO CHECK PBE ON GROUND TRUTH
        # output["bus"] = target

        Pft, Qft = branch_flow_layer(output["bus"], bus_edge_index, bus_edge_attr)
        P_in, Q_in = node_injection_layer(Pft, Qft, bus_edge_index, num_bus)
        residual_P, residual_Q = node_residuals_layer(
            P_in,
            Q_in,
            output["bus"],
            batch.x_dict["bus"],
        )

        bus_batch = batch.batch_dict["bus"]  # shape: [num_bus_total]

        mask_PQ = batch.mask_dict["PQ"]  # PQ buses
        mask_PV = batch.mask_dict["PV"]  # PV buses
        mask_REF = batch.mask_dict["REF"]  # Reference buses

        if self.args.verbose:
            mean_res_P_PQ, max_res_P_PQ = residual_stats_by_type(
                residual_P,
                mask_PQ,
                bus_batch,
            )
            mean_res_Q_PQ, max_res_Q_PQ = residual_stats_by_type(
                residual_Q,
                mask_PQ,
                bus_batch,
            )

            mean_res_P_PV, max_res_P_PV = residual_stats_by_type(
                residual_P,
                mask_PV,
                bus_batch,
            )
            mean_res_Q_PV, max_res_Q_PV = residual_stats_by_type(
                residual_Q,
                mask_PV,
                bus_batch,
            )

            mean_res_P_REF, max_res_P_REF = residual_stats_by_type(
                residual_P,
                mask_REF,
                bus_batch,
            )
            mean_res_Q_REF, max_res_Q_REF = residual_stats_by_type(
                residual_Q,
                mask_REF,
                bus_batch,
            )
            self.test_outputs[dataloader_idx].append(
                {
                    "dataset": dataset_name,
                    "pred": output["bus"].detach().cpu(),
                    "target": target.detach().cpu(),
                    "mask_PQ": mask_PQ.cpu(),
                    "mask_PV": mask_PV.cpu(),
                    "mask_REF": mask_REF.cpu(),
                    "mean_residual_P_PQ": mean_res_P_PQ.detach().cpu(),
                    "max_residual_P_PQ": max_res_P_PQ.detach().cpu(),
                    "mean_residual_Q_PQ": mean_res_Q_PQ.detach().cpu(),
                    "max_residual_Q_PQ": max_res_Q_PQ.detach().cpu(),
                    "mean_residual_P_PV": mean_res_P_PV.detach().cpu(),
                    "max_residual_P_PV": max_res_P_PV.detach().cpu(),
                    "mean_residual_Q_PV": mean_res_Q_PV.detach().cpu(),
                    "max_residual_Q_PV": max_res_Q_PV.detach().cpu(),
                    "mean_residual_P_REF": mean_res_P_REF.detach().cpu(),
                    "max_residual_P_REF": max_res_P_REF.detach().cpu(),
                    "mean_residual_Q_REF": mean_res_Q_REF.detach().cpu(),
                    "max_residual_Q_REF": max_res_Q_REF.detach().cpu(),
                },
            )

        final_residual_real_bus = torch.mean(torch.abs(residual_P))
        final_residual_imag_bus = torch.mean(torch.abs(residual_Q))

        loss_dict["Active Power Loss"] = final_residual_real_bus.detach()
        loss_dict["Reactive Power Loss"] = final_residual_imag_bus.detach()

        # Power Balance Error (PBE) metrics
        delta_PQ_2 = residual_P**2 + residual_Q**2
        delta_PQ_magn = torch.sqrt(delta_PQ_2)
        pbe_mean_per_graph = global_mean_pool(delta_PQ_magn, bus_batch)  # [num_graphs]
        pbe_mean = pbe_mean_per_graph.mean()
        pbe_max = delta_PQ_magn.max()

        loss_dict["PBE Mean"] = pbe_mean.detach()

        mse_PQ = F.mse_loss(
            output["bus"][mask_PQ],
            target[mask_PQ],
            reduction="none",
        )
        mse_PV = F.mse_loss(
            output["bus"][mask_PV],
            target[mask_PV],
            reduction="none",
        )
        mse_REF = F.mse_loss(
            output["bus"][mask_REF],
            target[mask_REF],
            reduction="none",
        )

        mse_PQ = mse_PQ.mean(dim=0)
        mse_PV = mse_PV.mean(dim=0)
        mse_REF = mse_REF.mean(dim=0)

        loss_dict["MSE PQ nodes - PG"] = mse_PQ[PG_OUT]
        loss_dict["MSE PV nodes - PG"] = mse_PV[PG_OUT]
        loss_dict["MSE REF nodes - PG"] = mse_REF[PG_OUT]

        loss_dict["MSE PQ nodes - QG"] = mse_PQ[QG_OUT]
        loss_dict["MSE PV nodes - QG"] = mse_PV[QG_OUT]
        loss_dict["MSE REF nodes - QG"] = mse_REF[QG_OUT]

        loss_dict["MSE PQ nodes - VM"] = mse_PQ[VM_OUT]
        loss_dict["MSE PV nodes - VM"] = mse_PV[VM_OUT]
        loss_dict["MSE REF nodes - VM"] = mse_REF[VM_OUT]

        loss_dict["MSE PQ nodes - VA"] = mse_PQ[VA_OUT]
        loss_dict["MSE PV nodes - VA"] = mse_PV[VA_OUT]
        loss_dict["MSE REF nodes - VA"] = mse_REF[VA_OUT]

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
        # Log PBE Max separately with max reduction across batches
        self.log(
            f"{dataset_name}/PBE Max",
            pbe_max.detach(),
            batch_size=batch.num_graphs,
            add_dataloader_idx=False,
            sync_dist=True,
            logger=False,
            reduce_fx="max",
        )
        return

    def on_test_end(self):
        # In DDP, gather verbose test outputs from all ranks to rank 0
        # so that plots and detailed analysis cover the full test set.
        if self.args.verbose and dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            gathered = [None] * world_size if dist.get_rank() == 0 else None
            dist.gather_object(self.test_outputs, gathered, dst=0)
            if dist.get_rank() == 0:
                merged = {i: [] for i in range(len(self.args.data.networks))}
                for rank_data in gathered:
                    for dl_idx, batches in rank_data.items():
                        merged[dl_idx].extend(batches)
                self.test_outputs = merged

        # Only rank 0 proceeds with logging, CSV writing, and plotting
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
            # RMSE metrics
            rmse_PQ = [
                metrics.get(f"MSE PQ nodes - {label}", float("nan")) ** 0.5
                for label in ["PG", "QG", "VM", "VA"]
            ]
            rmse_PV = [
                metrics.get(f"MSE PV nodes - {label}", float("nan")) ** 0.5
                for label in ["PG", "QG", "VM", "VA"]
            ]
            rmse_REF = [
                metrics.get(f"MSE REF nodes - {label}", float("nan")) ** 0.5
                for label in ["PG", "QG", "VM", "VA"]
            ]

            # Residuals and generator metrics
            avg_active_res = metrics.get("Active Power Loss", " ")
            avg_reactive_res = metrics.get("Reactive Power Loss", " ")
            pbe_mean_val = metrics.get("PBE Mean", " ")
            pbe_max_val = metrics.get("PBE Max", " ")

            # --- Main RMSE metrics file ---
            data_main = {
                "Metric": ["RMSE-PQ", "RMSE-PV", "RMSE-REF"],
                "Pg (MW)": [rmse_PQ[0], rmse_PV[0], rmse_REF[0]],
                "Qg (MVar)": [rmse_PQ[1], rmse_PV[1], rmse_REF[1]],
                "Vm (p.u.)": [rmse_PQ[2], rmse_PV[2], rmse_REF[2]],
                "Va (radians)": [rmse_PQ[3], rmse_PV[3], rmse_REF[3]],
            }
            df_main = pd.DataFrame(data_main)

            # --- Residuals / generator metrics file ---
            data_residuals = {
                "Metric": [
                    "Avg. active res. (MW)",
                    "Avg. reactive res. (MVar)",
                    "PBE Mean",
                    "PBE Max",
                ],
                "Value": [avg_active_res, avg_reactive_res, pbe_mean_val, pbe_max_val],
            }
            df_residuals = pd.DataFrame(data_residuals)

            # --- Save CSVs ---
            test_dir = os.path.join(artifact_dir, "test")
            os.makedirs(test_dir, exist_ok=True)

            main_csv_path = os.path.join(test_dir, f"{dataset}_RMSE.csv")
            residuals_csv_path = os.path.join(test_dir, f"{dataset}_metrics.csv")

            df_main.to_csv(main_csv_path, index=False)
            df_residuals.to_csv(residuals_csv_path, index=False)

        if self.args.verbose:
            for dataset_idx, outputs in self.test_outputs.items():
                dataset_name = self.args.data.networks[dataset_idx]

                plot_dir = os.path.join(artifact_dir, "test_plots", dataset_name)
                os.makedirs(plot_dir, exist_ok=True)

                # Concatenate predictions and targets across all batches
                all_preds = torch.cat([d["pred"] for d in outputs])
                all_targets = torch.cat([d["target"] for d in outputs])
                all_masks = {
                    "PQ": torch.cat([d["mask_PQ"] for d in outputs]),
                    "PV": torch.cat([d["mask_PV"] for d in outputs]),
                    "REF": torch.cat([d["mask_REF"] for d in outputs]),
                }

                plot_residuals_histograms(outputs, dataset_name, plot_dir)

                plot_correlation_by_node_type(
                    preds=all_preds,
                    targets=all_targets,
                    masks=all_masks,
                    feature_labels=["Vm", "Va", "Pg", "Qg"],
                    plot_dir=plot_dir,
                    prefix=dataset_name,
                )

        self.test_outputs.clear()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        output, _ = self.shared_step(batch)

        self.data_normalizers[dataloader_idx].inverse_transform(batch)
        self.data_normalizers[dataloader_idx].inverse_output(output, batch)

        branch_flow_layer = ComputeBranchFlow()
        node_injection_layer = ComputeNodeInjection()
        node_residuals_layer = ComputeNodeResiduals()

        num_bus = batch.x_dict["bus"].size(0)
        bus_edge_index = batch.edge_index_dict[("bus", "connects", "bus")]
        bus_edge_attr = batch.edge_attr_dict[("bus", "connects", "bus")]

        Pft, Qft = branch_flow_layer(output["bus"], bus_edge_index, bus_edge_attr)
        P_in, Q_in = node_injection_layer(Pft, Qft, bus_edge_index, num_bus)
        residual_P, residual_Q = node_residuals_layer(
            P_in,
            Q_in,
            output["bus"],
            batch.x_dict["bus"],
        )
        residual_P = torch.abs(residual_P)
        residual_Q = torch.abs(residual_Q)
        residual_mva = torch.sqrt(residual_P**2 + residual_Q**2)

        bus_batch = batch.batch_dict["bus"]
        scenario_ids = batch["scenario_id"][bus_batch]
        local_bus_idx = torch.cat(
            [
                torch.arange(c, device=bus_batch.device)
                for c in torch.bincount(bus_batch)
            ],
        )

        bus_x = batch.x_dict["bus"]
        bus_y = batch.y_dict["bus"]
        mask_PQ = batch.mask_dict["PQ"]
        mask_PV = batch.mask_dict["PV"]
        mask_REF = batch.mask_dict["REF"]

        _, gen_to_bus_index = batch.edge_index_dict[("gen", "connected_to", "bus")]
        agg_gen_on_bus = scatter_add(
            batch.y_dict["gen"],
            gen_to_bus_index,
            dim=0,
            dim_size=num_bus,
        )

        return {
            "scenario": scenario_ids.cpu().numpy(),
            "bus": local_bus_idx.cpu().numpy(),
            "pd_mw": bus_x[:, PD_H].cpu().numpy(),
            "qd_mvar": bus_x[:, QD_H].cpu().numpy(),
            "vm_pu_target": bus_y[:, VM_H].cpu().numpy(),
            "va_target": bus_y[:, VA_H].cpu().numpy(),
            "pg_mw_target": agg_gen_on_bus.squeeze().cpu().numpy(),
            "qg_mvar_target": bus_y[:, QG_H].cpu().numpy(),
            "is_pq": mask_PQ.cpu().numpy().astype(int),
            "is_pv": mask_PV.cpu().numpy().astype(int),
            "is_ref": mask_REF.cpu().numpy().astype(int),
            "vm_pu": output["bus"][:, VM_OUT].detach().cpu().numpy(),
            "va": output["bus"][:, VA_OUT].detach().cpu().numpy(),
            "pg_mw": output["bus"][:, PG_OUT].detach().cpu().numpy(),
            "qg_mvar": output["bus"][:, QG_OUT].detach().cpu().numpy(),
            "active res. (MW)": residual_P.detach().cpu().numpy(),
            "reactive res. (MVar)": residual_Q.detach().cpu().numpy(),
            "PBE": residual_mva.detach().cpu().numpy(),
        }
