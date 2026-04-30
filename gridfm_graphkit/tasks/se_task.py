from gridfm_graphkit.datasets.globals import (
    # Bus feature indices
    PD_H,
    QG_H,
    QD_H,
    VM_H,
    VA_H,
    # Output feature indices
    VM_OUT,
    VA_OUT,
    PG_OUT,
    QG_OUT,
    # Generator feature indices
    PG_H,
)

from gridfm_graphkit.tasks.reconstruction_tasks import ReconstructionTask
from gridfm_graphkit.io.registries import TASK_REGISTRY
from gridfm_graphkit.tasks.utils import plot_correlation_by_node_type
from pytorch_lightning.utilities import rank_zero_only
import torch
from torch_scatter import scatter_add
from lightning.pytorch.loggers import MLFlowLogger
import os


@TASK_REGISTRY.register("StateEstimation")
class StateEstimationTask(ReconstructionTask):
    def __init__(self, args, data_normalizers):
        super().__init__(args, data_normalizers)

    # TODO: add custom test and predict steps
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        output, loss_dict = self.shared_step(batch)
        dataset_name = self.args.data.networks[dataloader_idx]

        self.data_normalizers[dataloader_idx].inverse_transform(batch)
        self.data_normalizers[dataloader_idx].inverse_output(output, batch)

        num_bus = batch.x_dict["bus"].size(0)
        _, gen_to_bus_index = batch.edge_index_dict[("gen", "connected_to", "bus")]

        agg_gen_on_bus = scatter_add(
            batch.y_dict["gen"],
            gen_to_bus_index,
            dim=0,
            dim_size=num_bus,
        )
        target = torch.stack(
            [
                batch.y_dict["bus"][:, VM_H],
                batch.y_dict["bus"][:, VA_H],
                agg_gen_on_bus.squeeze() - batch.y_dict["bus"][:, PD_H],
                batch.y_dict["bus"][:, QG_H] - batch.y_dict["bus"][:, QD_H],
            ],
            dim=1,
        )

        agg_meas_gen_on_bus = scatter_add(
            batch.x_dict["gen"][:, [PG_H]],
            gen_to_bus_index,
            dim=0,
            dim_size=num_bus,
        )

        # fig, ax = plt.subplots()
        # mask = batch.mask_dict['gen'][:, PG_H]
        # ax.hist((batch.x_dict['gen'][~mask, PG_H] - batch.y_dict['gen'][~mask, PG_H]).cpu().numpy(), bins=100)
        # fig.savefig('gen.png')

        # fig, ax = plt.subplots()
        # mask = batch.mask_dict['bus'][:, PD_H]
        # ax.hist((batch.x_dict['bus'][~mask, PD_H] - batch.y_dict['bus'][~mask, PD_H]).cpu().numpy(), bins=100)
        # fig.savefig('pd.png')

        measurements = torch.stack(
            [
                batch.x_dict["bus"][:, VM_H],
                batch.x_dict["bus"][:, VA_H],
                agg_meas_gen_on_bus.squeeze() - batch.x_dict["bus"][:, PD_H],
                batch.x_dict["bus"][:, QG_H] - batch.x_dict["bus"][:, QD_H],
            ],
            dim=1,
        )

        # fig, ax = plt.subplots()
        # mask = batch.mask_dict['bus'][:, PG_H]
        # ax.hist((agg_meas_gen_on_bus.squeeze()[~mask] - agg_gen_on_bus.squeeze()[~mask]).cpu().numpy(), bins=100)
        # fig.savefig('gen_to_bus.png')

        outliers_bus = batch.mask_dict["outliers_bus"]
        mask_bus = batch.mask_dict["bus"][:, : outliers_bus.size(1)]
        non_outliers_bus = torch.logical_and(~outliers_bus, ~mask_bus)
        masks = [outliers_bus, mask_bus, non_outliers_bus]
        for i, mask in enumerate(masks):
            new_mask = torch.zeros_like(target, dtype=bool)
            new_mask[:, VM_OUT] = mask[:, VM_H]
            new_mask[:, VA_OUT] = mask[:, VA_H]
            new_mask[:, PG_OUT] = mask[:, PD_H]
            new_mask[:, QG_OUT] = mask[:, QD_H]
            masks[i] = new_mask
        outliers_bus, mask_bus, non_outliers_bus = masks

        # fig, ax = plt.subplots()
        # ax.hist((measurements[~mask_bus[:, PG_OUT], PG_OUT] - target[~mask_bus[:, PG_OUT], PG_OUT]).cpu().numpy(), bins=100)
        # fig.savefig('p_inj.png')
        # assert False

        self.test_outputs[dataloader_idx].append(
            {
                "dataset": dataset_name,
                "pred": output["bus"].detach().cpu(),
                "target": target.detach().cpu(),
                "measurement": measurements.cpu(),
                "mask_bus": mask_bus.detach().cpu(),
                "outliers_bus": outliers_bus.detach().cpu(),
                "non_outliers_bus": non_outliers_bus.detach().cpu(),
            },
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

        if self.args.verbose:
            for dataset_idx, outputs in self.test_outputs.items():
                dataset_name = self.args.data.networks[dataset_idx]

                plot_dir = os.path.join(artifact_dir, "test_plots", dataset_name)
                os.makedirs(plot_dir, exist_ok=True)

                # Concatenate predictions and targets across all batches
                all_preds = torch.cat([d["pred"] for d in outputs])
                all_targets = torch.cat([d["target"] for d in outputs])
                all_measurements = torch.cat([d["measurement"] for d in outputs])

                all_masks = {
                    m: torch.cat([d[m] for d in outputs])
                    for m in ["mask_bus", "outliers_bus", "non_outliers_bus"]
                }

                plot_correlation_by_node_type(
                    preds=all_preds,
                    targets=all_targets,
                    masks=all_masks,
                    feature_labels=["Vm", "Va", "Pg", "Qg"],
                    plot_dir=plot_dir,
                    prefix=dataset_name + "_pred_vs_target_",
                    xlabel="Target",
                    ylabel="Pred",
                )

                plot_correlation_by_node_type(
                    preds=all_preds,
                    targets=all_measurements,
                    masks=all_masks,
                    feature_labels=["Vm", "Va", "Pg", "Qg"],
                    plot_dir=plot_dir,
                    prefix=dataset_name + "_pred_vs_measured_",
                    xlabel="Measured",
                    ylabel="Pred",
                )

                plot_correlation_by_node_type(
                    preds=all_measurements,
                    targets=all_targets,
                    masks=all_masks,
                    feature_labels=["Vm", "Va", "Pg", "Qg"],
                    plot_dir=plot_dir,
                    prefix=dataset_name + "_measured_vs_target_",
                    xlabel="Target",
                    ylabel="Measured",
                )

        self.test_outputs.clear()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        pass
