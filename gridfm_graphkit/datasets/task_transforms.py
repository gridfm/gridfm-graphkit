from torch_geometric.transforms import Compose
from gridfm_graphkit.datasets.transforms import (
    RemoveInactiveBranches,
    RemoveInactiveGenerators,
    ApplyMasking,
    LoadGridParamsFromPath,
)
from gridfm_graphkit.datasets.masking import (
    AddOPFHeteroMask,
    AddPFHeteroMask,
    AddRandomHeteroMask,
    SimulateMeasurements,
)
from gridfm_graphkit.io.registries import TRANSFORM_REGISTRY


@TRANSFORM_REGISTRY.register("PowerFlow")
class PowerFlowTransforms(Compose):
    """Compose preprocessing and masking transforms for PowerFlow datasets."""

    def __init__(self, args):
        transforms = []

        transforms.append(RemoveInactiveBranches())
        transforms.append(RemoveInactiveGenerators())

        mask_type = getattr(args.data, "mask_type", None)
        if mask_type == "rnd":
            transforms.append(AddRandomHeteroMask(mask_ratio=args.data.mask_ratio))
        else:
            transforms.append(
                AddPFHeteroMask(
                    mask_ref_vm=getattr(args.data, "mask_ref_vm", False),
                ),
            )

        transforms.append(ApplyMasking(args=args))

        # Pass the list of transforms to Compose
        super().__init__(transforms)


@TRANSFORM_REGISTRY.register("OptimalPowerFlow")
class OptimalPowerFlowTransforms(Compose):
    """Compose preprocessing and masking transforms for OptimalPowerFlow datasets."""

    def __init__(self, args):
        transforms = []

        transforms.append(RemoveInactiveBranches())
        transforms.append(RemoveInactiveGenerators())

        mask_type = getattr(args.data, "mask_type", None)
        if mask_type == "rnd":
            transforms.append(AddRandomHeteroMask(mask_ratio=args.data.mask_ratio))
        else:
            transforms.append(AddOPFHeteroMask())

        transforms.append(ApplyMasking(args=args))

        # Pass the list of transforms to Compose
        super().__init__(transforms)


@TRANSFORM_REGISTRY.register("StateEstimation")
class StateEstimationTransforms(Compose):
    """Compose preprocessing and measurement transforms for StateEstimation datasets."""

    def __init__(self, args):
        transforms = []

        if hasattr(args.task, "grid_path"):
            transforms.append(LoadGridParamsFromPath(args))
        transforms.append(RemoveInactiveBranches())
        transforms.append(RemoveInactiveGenerators())
        transforms.append(SimulateMeasurements(args=args))
        transforms.append(ApplyMasking(args=args))

        # Pass the list of transforms to Compose
        super().__init__(transforms)
