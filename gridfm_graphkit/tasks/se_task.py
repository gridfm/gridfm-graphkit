from gridfm_graphkit.datasets.globals import (
    # Bus feature indices
    PD_H,
    QG_H,
    QD_H,
    VM_H,
    VA_H,
    MIN_QG_H,
    MAX_QG_H,
    MIN_VM_H,
    MAX_VM_H,
    GS,
    BS,
    VN_KV,
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
    MIN_PG,
    MAX_PG,
    # Edge feature indices
    ANG_MIN,
    ANG_MAX,
    RATE_A,
    YFF_TT_R,
    YFT_TF_I,
)

from gridfm_graphkit.tasks.reconstruction_tasks import ReconstructionTask
from gridfm_graphkit.io.registries import TASK_REGISTRY
from gridfm_graphkit.tasks.utils import plot_correlation_by_node_type

import os
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from lightning.pytorch.loggers import MLFlowLogger
from pytorch_lightning.utilities import rank_zero_only
from torch_scatter import scatter_add

FEATURE_LABELS = ["Vm", "Va", "Pg", "Qg"]
MASK_NAMES = ["mask_bus", "outliers_bus", "non_outliers_bus"]


@TASK_REGISTRY.register("StateEstimation")
class StateEstimationTask(ReconstructionTask):
    """State estimation from simulated measurements.

    With `task.run_pp: true`, every test scenario is also solved with the WLS
    baseline (`PandapowerSE`). This needs `training.batch_size: 1` and runs on CPU.

    The test writes one metrics CSV per scenario set:

    - `c`: neural estimator, scenarios where WLS converged.
    - `pp`: WLS, scenarios where WLS converged.
    - `nc`: neural estimator, scenarios where WLS did not converge.
    - `cnc`: neural estimator, all scenarios.

    Without `run_pp`, every scenario counts as converged and `pp` is not written.
    Voltage angles are reported in degrees and powers in MW / MVAr.
    """

    def __init__(self, args, data_normalizers):
        super().__init__(args, data_normalizers)
        self.args = args
        self.parameter_estimation = getattr(args.task, "parameter_estimation", False)
        self.run_pp = getattr(args.task, "run_pp", False)
        if self.run_pp and self.parameter_estimation:
            raise ValueError(
                "The WLS baseline does not support parameter estimation. Set task.run_pp to false.",
            )
        if self.run_pp:
            from gridfm_graphkit.models.pandapower_se import PandapowerSE

            self.pp_se = PandapowerSE(args)

    def shared_step(self, batch):
        # OPF limits and costs are not inputs for SE. They are zeroed here and not in the
        # transforms, because the WLS baseline still needs them to build the network.
        batch.x_dict["bus"][:, [MIN_VM_H, MAX_VM_H, MIN_QG_H, MAX_QG_H, VN_KV]] = 0.0
        batch.x_dict["gen"][:, [MIN_PG, MAX_PG, C0_H, C1_H, C2_H]] = 0.0
        batch.edge_attr_dict[("bus", "connects", "bus")][
            :,
            [ANG_MAX, ANG_MIN, RATE_A],
        ] = 0.0
        return super().shared_step(batch)

    def run_wls(self, batch, dataloader_idx):
        """Run the WLS baseline on a denormalized copy of one scenario."""
        normalizer = self.data_normalizers[dataloader_idx]
        batch = deepcopy(batch)
        # Shunts and admittances back to p.u. on the case base, with one float32
        # multiplication by baseMVA / baseMVA_orig. inverse_transform scales them
        # for the physics layers instead, so they are restored first.
        edges = batch.edge_attr_dict[("bus", "connects", "bus")]
        admittance = edges[:, YFF_TT_R : YFT_TF_I + 1].clone()
        shunts = batch.x_dict["bus"][:, [GS, BS]].clone()
        normalizer.inverse_transform(batch)
        normalizer.inverse_std(batch)
        batch.x_dict["bus"][:, VA_H] *= 180 / torch.pi
        scale = normalizer.baseMVA / normalizer.baseMVA_orig
        batch.edge_attr_dict[("bus", "connects", "bus")][:, YFF_TT_R : YFT_TF_I + 1] = (
            admittance * scale
        )
        batch.x_dict["bus"][:, [GS, BS]] = shunts * scale
        return self.pp_se(
            x_dict=batch.x_dict,
            edge_index_dict=batch.edge_index_dict,
            edge_attr_dict=batch.edge_attr_dict,
            mask_dict=batch.mask_dict,
            baseMVA_orig=normalizer.baseMVA_orig,
        )

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        outputpp = self.run_wls(batch, dataloader_idx) if self.run_pp else None

        output, _ = self.shared_step(batch)
        dataset_name = self.args.data.networks[dataloader_idx]

        self.data_normalizers[dataloader_idx].inverse_transform(batch)
        self.data_normalizers[dataloader_idx].inverse_output(output, batch)

        num_bus = batch.x_dict["bus"].size(0)
        _, gen_to_bus_index = batch.edge_index_dict[("gen", "connected_to", "bus")]

        # Per bus: Vm (p.u.), Va (degrees), P_inj (MW), Q_inj (MVAr).
        agg_gen_on_bus = scatter_add(
            batch.y_dict["gen"],
            gen_to_bus_index,
            dim=0,
            dim_size=num_bus,
        ).squeeze(-1)
        target = torch.stack(
            [
                batch.y_dict["bus"][:, VM_H],
                batch.y_dict["bus"][:, VA_H],
                agg_gen_on_bus - batch.y_dict["bus"][:, PD_H],
                batch.y_dict["bus"][:, QG_H] - batch.y_dict["bus"][:, QD_H],
            ],
            dim=1,
        )
        agg_meas_gen_on_bus = scatter_add(
            batch.x_dict["gen"][:, [PG_H]],
            gen_to_bus_index,
            dim=0,
            dim_size=num_bus,
        ).squeeze(-1)
        measurement = torch.stack(
            [
                batch.x_dict["bus"][:, VM_H],
                batch.x_dict["bus"][:, VA_H],
                agg_meas_gen_on_bus - batch.x_dict["bus"][:, PD_H],
                batch.x_dict["bus"][:, QG_H] - batch.x_dict["bus"][:, QD_H],
            ],
            dim=1,
        )
        pred = output["bus"].clone()
        for tensor in (target, measurement, pred):
            tensor[:, VA_OUT] *= 180 / torch.pi

        # Bus masks per output column: masked (not measured), outlier, and measured non-outlier.
        outliers_bus = batch.mask_dict["outliers_bus"]
        mask_bus = batch.mask_dict["bus"][:, : outliers_bus.size(1)]
        non_outliers_bus = torch.logical_and(~outliers_bus, ~mask_bus)
        masks = {}
        for name, mask in zip(MASK_NAMES, [mask_bus, outliers_bus, non_outliers_bus]):
            new_mask = torch.zeros_like(target, dtype=torch.bool)
            new_mask[:, VM_OUT] = mask[:, VM_H]
            new_mask[:, VA_OUT] = mask[:, VA_H]
            new_mask[:, PG_OUT] = mask[:, PD_H]
            new_mask[:, QG_OUT] = mask[:, QD_H]
            masks[name] = new_mask.detach().cpu()

        res_dict = {
            "dataset": dataset_name,
            "num_graphs": batch.num_graphs,
            "pred": pred.detach().cpu(),
            "target": target.detach().cpu(),
            "measurement": measurement.detach().cpu(),
            "converged": outputpp["success"] if self.run_pp else True,
            **masks,
        }
        if self.run_pp:
            res_dict["pppred"] = outputpp["bus"].detach().float().cpu()

        if self.parameter_estimation:
            normalizer = self.data_normalizers[dataloader_idx]
            pred_impedance = normalizer.inverse_impedance(output["impedance"])
            target_impedance = normalizer.inverse_impedance(batch.y_dict["impedance"])
            measurement_impedance = normalizer.inverse_impedance(
                batch.x_dict["impedance"],
            )
            pred_impedance[target_impedance.abs() < 1e-6] = 0
            res_dict["pred_impedance"] = pred_impedance.detach().cpu()
            res_dict["target_impedance"] = target_impedance.detach().cpu()
            res_dict["measurement_impedance"] = measurement_impedance.detach().cpu()

        self.test_outputs[dataloader_idx].append(res_dict)

    @staticmethod
    def _concat(outputs, key_pred):
        """Concatenate predictions, targets, measurements and masks over test batches."""
        if len(outputs) == 0:
            return None
        return (
            torch.cat([d[key_pred] for d in outputs]),
            torch.cat([d["target"] for d in outputs]),
            torch.cat([d["measurement"] for d in outputs]),
            {m: torch.cat([d[m] for d in outputs]) for m in MASK_NAMES},
        )

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

        for dataset_idx, outputs in self.test_outputs.items():
            if len(outputs) == 0:
                continue
            dataset_name = self.args.data.networks[dataset_idx]
            plot_dir = os.path.join(artifact_dir, "test_plots", dataset_name)
            test_dir = os.path.join(artifact_dir, "test")
            os.makedirs(test_dir, exist_ok=True)

            # Every scenario of a grid has the same number of buses.
            self.buses_in_grid = outputs[0]["pred"].shape[0] // outputs[0]["num_graphs"]
            converged_outputs = [d for d in outputs if d["converged"]]
            failed_outputs = [d for d in outputs if not d["converged"]]
            converged = len(converged_outputs) / len(outputs)

            sets = {
                "c": self._concat(converged_outputs, "pred"),
                "pp": self._concat(converged_outputs, "pppred")
                if self.run_pp
                else None,
                "nc": self._concat(failed_outputs, "pred"),
                "cnc": self._concat(outputs, "pred"),
            }
            for label, values in sets.items():
                if values is None:
                    continue
                all_preds, all_targets, all_measurements, all_masks = values

                if self.args.verbose:
                    for preds, targets, name, xlabel, ylabel in [
                        (all_preds, all_targets, "pred_vs_target", "Target", "Pred"),
                        (
                            all_preds,
                            all_measurements,
                            "pred_vs_measured",
                            "Measured",
                            "Pred",
                        ),
                        (
                            all_measurements,
                            all_targets,
                            "measured_vs_target",
                            "Target",
                            "Measured",
                        ),
                    ]:
                        plot_correlation_by_node_type(
                            preds=preds,
                            targets=targets,
                            masks=all_masks,
                            feature_labels=FEATURE_LABELS,
                            plot_dir=plot_dir,
                            prefix=f"{dataset_name}_{name}_{label}",
                            xlabel=xlabel,
                            ylabel=ylabel,
                        )

                metrics = {}
                for i, feature_label in enumerate(FEATURE_LABELS):
                    feature_masks = {key: mask[:, i] for key, mask in all_masks.items()}
                    feature_masks["all"] = torch.ones_like(
                        all_targets[:, i],
                        dtype=torch.bool,
                    )
                    metrics[feature_label] = self.calculate_metrics(
                        (all_targets[:, i] - all_preds[:, i]).abs(),
                        feature_masks,
                    )
                metrics = pd.DataFrame.from_dict(metrics, orient="index")
                metrics["converged"] = converged
                print(f"{dataset_name} [{label}]\n{metrics}")
                metrics.to_csv(
                    os.path.join(test_dir, f"{dataset_name}_metrics_{label}.csv"),
                )

            if self.parameter_estimation:
                self.save_parameter_metrics(
                    outputs,
                    os.path.join(test_dir, f"{dataset_name}_parameters_cnc.csv"),
                )

        self.test_outputs.clear()

    @staticmethod
    def save_parameter_metrics(outputs, path):
        """Per-scenario branch r, x errors of the prediction and of the noisy input.

        Columns: `{,rel,meas,meas_rel}{mean,std,median}_grid`. `rel` is relative to the
        true value. `meas` compares the noisy input with the true value.
        """
        metrics = {}
        for i, feature_label in enumerate(["r", "x"]):
            targets = [d["target_impedance"][:, i] for d in outputs]
            preds = [d["pred_impedance"][:, i] for d in outputs]
            measurements = [d["measurement_impedance"][:, i] for d in outputs]
            keep = [t > 1e-6 for t in targets]
            targets = [t[k] for t, k in zip(targets, keep)]
            preds = [p[k] for p, k in zip(preds, keep)]
            measurements = [m[k] for m, k in zip(measurements, keep)]

            errors = {
                "": torch.tensor(
                    [(t - p).abs().mean() for t, p in zip(targets, preds)],
                ),
                "rel": torch.tensor(
                    [((t - p) / t).abs().mean() for t, p in zip(targets, preds)],
                ),
                "meas": torch.tensor(
                    [(m - t).abs().mean() for t, m in zip(targets, measurements)],
                ),
                "meas_rel": torch.tensor(
                    [((m - t) / t).abs().mean() for t, m in zip(targets, measurements)],
                ),
            }
            row = {}
            for prefix, error in errors.items():
                row[prefix + "mean_grid"] = error.mean().item()
                row[prefix + "std_grid"] = error.std().item()
                row[prefix + "median_grid"] = error.median().item()
            metrics[feature_label] = row
        pd.DataFrame.from_dict(metrics, orient="index").to_csv(path)

    def calculate_metrics(self, error, masks, prefix=""):
        """Absolute error statistics.

        `mean_<mask>`, `std_<mask>`, `median_<mask>` are taken over all buses in the mask.
        `*_grid` statistics are taken over scenarios, after averaging the error over the buses of each scenario.
        """
        metrics = {}
        for mask_name, mask in masks.items():
            errorm = error[mask]
            if len(errorm) == 0:
                continue
            metrics[prefix + f"mean_{mask_name}"] = errorm.mean().item()
            metrics[prefix + f"std_{mask_name}"] = errorm.std().item()
            metrics[prefix + f"median_{mask_name}"] = errorm.median().item()

        per_scenario = error.reshape(-1, self.buses_in_grid).mean(dim=1)
        metrics[prefix + "mean_grid"] = per_scenario.mean().item()
        metrics[prefix + "std_grid"] = (
            per_scenario.std().item() if len(per_scenario) > 1 else float(np.nan)
        )
        metrics[prefix + "median_grid"] = per_scenario.median().item()
        return metrics

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        pass
