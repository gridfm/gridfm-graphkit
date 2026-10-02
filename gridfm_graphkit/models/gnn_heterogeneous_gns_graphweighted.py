import torch
from torch_scatter import scatter_mean

from gridfm_graphkit.io.registries import MODELS_REGISTRY
from gridfm_graphkit.models.gnn_heterogeneous_gns import GNS_heterogeneous


@MODELS_REGISTRY.register("GNS_heterogeneous_GraphWeighted")
class GNS_heterogeneous_GraphWeighted(GNS_heterogeneous):
    """``GNS_heterogeneous``, but the per-layer physics-residual term that
    feeds ``LayeredWeightedPhysicsLoss`` gives every graph in the batch equal
    weight, instead of weighting by bus count.

    The only change from the parent class: ``_residual_norm_mean`` averages
    each graph's own residual-norm mean, rather than taking one flat mean
    over every bus row pooled across the batch. Everything else (forward
    pass, decoders, other loss terms) is inherited unchanged from
    ``GNS_heterogeneous`` — this class exists so the node-weighted default
    path (``GNS_heterogeneous``) is never touched; pick this type in
    ``model.type`` to opt in.

    Pairs with the graph-weighted loss variants
    (``MaskedBusMSEGraphWeighted``, ``MaskedGenMSEGraphWeighted``,
    ``QgViolationPenaltyGraphWeighted``) in ``training.losses`` to make every
    term in the mixed loss equal-weight-per-graph, not just the physics term.
    """

    def _residual_norm_mean(self, bus_residuals, batch=None):
        if batch is None:
            # No batch index available (e.g. called outside the normal
            # training/eval path): fall back to the parent's flat mean.
            return super()._residual_norm_mean(bus_residuals, batch)

        batch_index = batch["bus"].batch
        residual_norm = torch.linalg.norm(bus_residuals, dim=-1)
        num_graphs = int(batch_index.max().item()) + 1 if batch_index.numel() else 0
        if num_graphs == 0:
            return super()._residual_norm_mean(bus_residuals, batch)
        per_graph_mean = scatter_mean(residual_norm, batch_index, dim=0, dim_size=num_graphs)
        return per_graph_mean.mean()
