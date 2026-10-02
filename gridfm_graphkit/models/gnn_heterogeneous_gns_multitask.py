from gridfm_graphkit.io.registries import MODELS_REGISTRY, PHYSICS_DECODER_REGISTRY
from gridfm_graphkit.models.gnn_heterogeneous_gns import GNS_heterogeneous


@MODELS_REGISTRY.register("GNS_heterogeneous_MultiTask")
class GNS_heterogeneous_MultiTask(GNS_heterogeneous):
    """Plain GENCO with one shared trunk and a switchable task decoder.

    Every forward is the unchanged GENCO forward on a single-task batch.
    Call ``set_task`` before the forward to pick the power-flow or OPF path.
    """

    def __init__(self, args) -> None:
        original_task = args.task.task_name
        # The parent looks up a decoder from the task name. Build the trunk
        # as OPF, then restore the caller's task name.
        args.task.task_name = "OptimalPowerFlow"
        super().__init__(args)
        args.task.task_name = original_task
        self.opf_decoder = self.physics_decoder
        self.pf_decoder = PHYSICS_DECODER_REGISTRY.create("PowerFlow")
        self._decoders = {
            "OptimalPowerFlow": self.opf_decoder,
            "PowerFlow": self.pf_decoder,
        }

    def set_task(self, task_name: str) -> None:
        if task_name not in self._decoders:
            raise KeyError(f"Unsupported task: {task_name}")
        self.task = task_name
        self.physics_decoder = self._decoders[task_name]
