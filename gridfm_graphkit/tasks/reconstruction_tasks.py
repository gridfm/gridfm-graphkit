from gridfm_graphkit.io.param_handler import load_model, get_loss_function
from gridfm_graphkit.tasks.base_task import BaseTask
from pytorch_lightning.utilities import rank_zero_only


class ReconstructionTask(BaseTask):
    """
    PyTorch Lightning task for node feature reconstruction on power grid graphs.

    This task wraps a GridFM model inside a LightningModule and defines the full
    training, validation, testing, and prediction logic. It is designed to
    reconstruct masked node features from graph-structured input data, using
    datasets and normalizers provided by `gridfm-graphkit`.

    Args:
        args (NestedNamespace): Experiment configuration. Expected fields include `training.batch_size`, `optimizer.*`, etc.
        data_normalizers (list): One normalizer per dataset to (de)normalize features.

    Attributes:
        model (torch.nn.Module): model loaded via `load_model`.
        loss_fn (callable): Loss function resolved from configuration.
        batch_size (int): Training batch size. From ``args.training.batch_size``
        data_normalizers (list): Dataset-wise feature normalizers.

    Methods:
        forward(x, pe, edge_index, edge_attr, batch, mask=None):
            Forward pass with optional feature masking.
        training_step(batch):
            One training step: computes loss, logs metrics, returns loss.
        validation_step(batch, batch_idx):
            One validation step: computes losses and logs metrics.

    """

    def __init__(self, args, data_normalizers):
        super().__init__(args, data_normalizers)
        self.model = load_model(args=args)
        self.loss_fn = get_loss_function(args)
        self.batch_size = int(args.training.batch_size)
        self.test_outputs = {i: [] for i in range(len(args.data.networks))}

    def forward(self, x_dict, edge_index_dict, edge_attr_dict, mask_dict):
        return self.model(x_dict, edge_index_dict, edge_attr_dict, mask_dict)

    def shared_step(self, batch):
        output = self.forward(
            x_dict=batch.x_dict,
            edge_index_dict=batch.edge_index_dict,
            edge_attr_dict=batch.edge_attr_dict,
            mask_dict=batch.mask_dict,
        )

        loss_dict = self.loss_fn(
            output,
            batch.y_dict,
            batch.edge_index_dict,
            batch.edge_attr_dict,
            batch.mask_dict,
            model=self.model,
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
                sync_dist=False,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                on_step=True,
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
                logger=True,
                on_step=False,
            )

        return loss_dict["loss"]

    @rank_zero_only
    def on_test_end(self):
        """Optional shared test end logic, like clearing stored outputs"""
        self.test_outputs.clear()
