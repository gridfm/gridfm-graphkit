from gridfm_graphkit.datasets.globals import (
    # Bus feature indices
    QG_H,
    VM_H,
    VA_H,
    MIN_QG_H,
    MAX_QG_H,
    # Output feature indices
    VM_OUT,
    VA_OUT,
    PG_OUT,
    QG_OUT,
    # Generator feature indices
    PG_H,
    C0_H,
    C1_H,
    C2_H,
    # Edge feature indices
    ANG_MIN,
    ANG_MAX,
    RATE_A,
)

from gridfm_graphkit.tasks.reconstruction_tasks import ReconstructionTask
from gridfm_graphkit.io.registries import TASK_REGISTRY
from gridfm_graphkit.tasks.utils import (
    plot_correlation_by_node_type,
    plot_residuals_histograms,
    residual_stats_by_type,
)
from pytorch_lightning.utilities import rank_zero_only
import torch
import torch.nn.functional as F
from torch_scatter import scatter_add
from gridfm_graphkit.models.utils import (
    ComputeBranchFlow,
    ComputeNodeInjection,
    ComputeNodeResiduals,
)
import matplotlib.pyplot as plt
import seaborn as sns
from lightning.pytorch.loggers import MLFlowLogger
import numpy as np
import os
import pandas as pd


@TASK_REGISTRY.register("OptimalPowerFlow")
class OptimalPowerFlowTask(ReconstructionTask):
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

        mse_PG = F.mse_loss(
            output["gen"],
            batch.y_dict["gen"],
            reduction="none",
        ).mean(dim=0)
        c0 = batch.x_dict["gen"][:, C0_H]
        c1 = batch.x_dict["gen"][:, C1_H]
        c2 = batch.x_dict["gen"][:, C2_H]
        target_pg = batch.y_dict["gen"].squeeze()
        pred_pg = output["gen"].squeeze()
        gen_cost_gt = c0 + c1 * target_pg + c2 * target_pg**2
        gen_cost_pred = c0 + c1 * pred_pg + c2 * pred_pg**2

        gen_batch = batch.batch_dict["gen"]  # shape: [N_gen_total]

        cost_gt = scatter_add(gen_cost_gt, gen_batch, dim=0)
        cost_pred = scatter_add(gen_cost_pred, gen_batch, dim=0)

        optimality_gap = torch.mean(torch.abs((cost_pred - cost_gt) / cost_gt * 100))

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
        # Compute branch termal limits violations
        Sft = torch.sqrt(Pft**2 + Qft**2)  # apparent power flow per branch
        branch_thermal_limits = bus_edge_attr[:, RATE_A]
        branch_thermal_excess = F.relu(Sft - branch_thermal_limits)

        num_edges = bus_edge_index.size(1)
        half_edges = num_edges // 2
        forward_excess = branch_thermal_excess[:half_edges]
        reverse_excess = branch_thermal_excess[half_edges:]

        mean_thermal_violation_forward = torch.mean(forward_excess)
        mean_thermal_violation_reverse = torch.mean(reverse_excess)

        # Compute branch angle difference violation
        angle_min = bus_edge_attr[:, ANG_MIN]
        angle_max = bus_edge_attr[:, ANG_MAX]

        bus_angles = output["bus"][:, VA_OUT]  # in degrees
        from_bus = bus_edge_index[0]
        to_bus = bus_edge_index[1]
        angle_diff = torch.abs(bus_angles[from_bus] - bus_angles[to_bus])

        angle_excess_low = F.relu(angle_min - angle_diff)  # violation if too small
        angle_excess_high = F.relu(angle_diff - angle_max)  # violation if too large
        branch_angle_violation_mean = (
            torch.mean(angle_excess_low + angle_excess_high) * 180.0 / torch.pi
        )

        P_in, Q_in = node_injection_layer(Pft, Qft, bus_edge_index, num_bus)
        residual_P, residual_Q = node_residuals_layer(
            P_in,
            Q_in,
            output["bus"],
            batch.x_dict["bus"],
        )

        # --- Qg limit violation mask ---
        Qg_pred = output["bus"][:, QG_OUT]
        Qg_max = batch.x_dict["bus"][:, MAX_QG_H]
        Qg_min = batch.x_dict["bus"][:, MIN_QG_H]

        mask_Qg_violation = (Qg_pred > Qg_max) | (Qg_pred < Qg_min)

        bus_batch = batch.batch_dict["bus"]  # shape: [num_bus_total]

        mask_PQ = batch.mask_dict["PQ"]  # PQ buses
        mask_PV = batch.mask_dict["PV"]  # PV buses
        mask_REF = batch.mask_dict["REF"]  # Reference buses

        Qg_over = F.relu(Qg_pred - Qg_max)  # amount above max limit
        Qg_under = F.relu(Qg_min - Qg_pred)  # amount below min limit
        Qg_violation_amount = Qg_over + Qg_under

        mean_Qg_violation_PV = Qg_violation_amount[mask_PV].mean()
        mean_Qg_violation_REF = Qg_violation_amount[mask_REF].mean()

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
                    "cost_predicted": cost_pred.detach().cpu(),
                    "cost_ground_truth": cost_gt.detach().cpu(),
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
                    "mask_Qg_violation": mask_Qg_violation.detach().cpu(),
                },
            )

        final_residual_real_bus = torch.mean(torch.abs(residual_P))
        final_residual_imag_bus = torch.mean(torch.abs(residual_Q))

        loss_dict["Active Power Loss"] = final_residual_real_bus.detach()
        loss_dict["Reactive Power Loss"] = final_residual_imag_bus.detach()

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

        loss_dict["Opt gap"] = optimality_gap
        loss_dict["MSE PG"] = mse_PG[PG_H]

        loss_dict["Branch thermal violation from"] = mean_thermal_violation_forward
        loss_dict["Branch thermal violation to"] = mean_thermal_violation_reverse
        loss_dict["Branch voltage angle difference violations"] = (
            branch_angle_violation_mean
        )
        loss_dict["Mean Qg violation PV buses"] = mean_Qg_violation_PV
        loss_dict["Mean Qg violation REF buses"] = mean_Qg_violation_REF

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
        return

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
            rmse_gen = metrics.get("MSE PG", 0) ** 0.5
            optimality_gap = metrics.get("Opt gap", " ")
            branch_thermal_violation_from = metrics.get(
                "Branch termal violation from",
                " ",
            )
            branch_thermal_violation_to = metrics.get("Branch termal violation to", " ")
            branch_angle_violation = metrics.get(
                "Branch voltage angle difference violations",
                " ",
            )
            mean_qg_violation_PV_buses = metrics.get("Mean Qg violation PV buses", " ")
            mean_qg_violation_REF_buses = metrics.get(
                "Mean Qg violation REF buses",
                " ",
            )

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
                    "RMSE PG generators (MW)",
                    "Mean optimality gap (%)",
                    "Mean branch termal violation from (MVA)",
                    "Mean branch termal violation to (MVA)",
                    "Mean branch angle difference violation (radians)",
                    "Mean Qg violation PV buses",
                    "Mean Qg violation REF buses",
                ],
                "Value": [
                    avg_active_res,
                    avg_reactive_res,
                    rmse_gen,
                    optimality_gap,
                    branch_thermal_violation_from,
                    branch_thermal_violation_to,
                    branch_angle_violation,
                    mean_qg_violation_PV_buses,
                    mean_qg_violation_REF_buses,
                ],
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
                all_cost_pred = torch.cat([d["cost_predicted"] for d in outputs])
                all_cost_ground_truth = torch.cat(
                    [d["cost_ground_truth"] for d in outputs],
                )

                # Convert to numpy for plotting
                y_pred = all_cost_pred.numpy()
                y_true = all_cost_ground_truth.numpy()

                # Compute correlation coefficient
                corr = np.corrcoef(y_true, y_pred)[0, 1]

                # Create scatter plot
                plt.figure(figsize=(6, 6))
                sns.scatterplot(x=y_true, y=y_pred, s=20, alpha=0.6)

                # Add y=x reference line
                min_val = min(y_true.min(), y_pred.min())
                max_val = max(y_true.max(), y_pred.max())
                plt.plot(
                    [min_val, max_val],
                    [min_val, max_val],
                    "k--",
                    linewidth=1.0,
                    alpha=0.7,
                )

                # Add correlation coefficient text
                plt.text(
                    0.05,
                    0.95,
                    f"R = {corr:.3f}",
                    transform=plt.gca().transAxes,
                    fontsize=12,
                    verticalalignment="top",
                    bbox=dict(facecolor="white", alpha=0.6),
                )

                plt.xlabel("Ground Truth Cost")
                plt.ylabel("Predicted Cost")
                plt.title(f"{dataset_name} – Predicted vs Ground Truth Cost")
                plt.tight_layout()
                plt.savefig(
                    os.path.join(plot_dir, f"{dataset_name}_objective.png"),
                    dpi=300,
                )
                plt.close()

                plot_residuals_histograms(outputs, dataset_name, plot_dir)

                plot_correlation_by_node_type(
                    preds=all_preds,
                    targets=all_targets,
                    masks=all_masks,
                    feature_labels=["Vm", "Va", "Pg", "Qg"],
                    plot_dir=plot_dir,
                    prefix=dataset_name,
                    qg_violation_mask=torch.cat(
                        [d["mask_Qg_violation"] for d in outputs],
                    ),
                )

        self.test_outputs.clear()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        raise NotImplementedError
