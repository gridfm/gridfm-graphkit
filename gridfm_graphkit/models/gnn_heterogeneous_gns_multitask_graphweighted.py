from gridfm_graphkit.io.registries import MODELS_REGISTRY
from gridfm_graphkit.models.gnn_heterogeneous_gns_graphweighted import (
    GNS_heterogeneous_GraphWeighted,
)
from gridfm_graphkit.models.gnn_heterogeneous_gns_multitask import (
    GNS_heterogeneous_MultiTask,
)


@MODELS_REGISTRY.register("GNS_heterogeneous_MultiTask_GraphWeighted")
class GNS_heterogeneous_MultiTask_GraphWeighted(
    GNS_heterogeneous_GraphWeighted,
    GNS_heterogeneous_MultiTask,
):
    """``GNS_heterogeneous_MultiTask`` with the graph-weighted physics-residual
    reduction from ``GNS_heterogeneous_GraphWeighted``.

    Pure composition, no new logic: inherits the mixed PF/OPF task-decoder
    switching from ``GNS_heterogeneous_MultiTask`` and the equal-weight-per-
    graph ``_residual_norm_mean`` override from ``GNS_heterogeneous_GraphWeighted``.
    MRO puts the graph-weighted override first, so it wins over the parent's
    default; everything else (``__init__``, ``set_task``, ``forward``) comes
    from ``GNS_heterogeneous_MultiTask`` as before.
    """
