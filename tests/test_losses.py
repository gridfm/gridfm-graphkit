import pytest
import torch
from gridfm_graphkit.datasets.hetero_powergrid_datamodule import LitGridHeteroDataModule
from gridfm_graphkit.io.param_handler import NestedNamespace, get_loss_function
from gridfm_graphkit.training.loss import MixedLoss
from gridfm_graphkit.datasets.globals import VM_H, VA_H, QG_H
from torch_scatter import scatter_add
from gridfm_graphkit.models.utils import (
    ComputeBranchFlow,
    ComputeNodeInjection,
    ComputeNodeResiduals,
)


@pytest.fixture
def small_grid_data_module():
    # Load config
    import yaml

    with open("tests/config/datamodule_test_base_config.yaml") as f:
        config_dict = yaml.safe_load(f)

    args = NestedNamespace(**config_dict)
    dm = LitGridHeteroDataModule(args, data_dir="tests/data")

    # Fake trainer for setup
    class DummyTrainer:
        is_global_zero = True

    dm.trainer = DummyTrainer()
    dm.setup("train")
    return dm


def test_pbe_loss_zero_with_real_data(small_grid_data_module):
    loader = small_grid_data_module.train_dataloader()
    batch = next(iter(loader))

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

    Pft, Qft = branch_flow_layer(target, bus_edge_index, bus_edge_attr)
    P_in, Q_in = node_injection_layer(Pft, Qft, bus_edge_index, num_bus)
    residual_P, residual_Q = node_residuals_layer(
        P_in,
        Q_in,
        target,
        batch.x_dict["bus"],
    )
    assert torch.max(torch.abs(residual_P)) < 1e-4, (
        f"Active Residuals are not zero! {torch.max(torch.abs(residual_P))}"
    )
    assert torch.max(torch.abs(residual_Q)) < 1e-4, (
        f"Reactive Residuals not zero! {torch.max(torch.abs(residual_Q))}"
    )


class ConstantLoss(torch.nn.Module):
    """Loss stub returning a fixed value, to check how MixedLoss weights terms."""

    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, pred, target, *args, **kwargs):
        return {"loss": torch.tensor(self.value)}


def test_mixed_loss_weights_constant_without_warmup():
    loss_fn = MixedLoss(
        loss_functions=[ConstantLoss(1.0), ConstantLoss(2.0)],
        weights=[0.1, 0.9],
    )

    for epoch in range(5):
        loss_fn.set_epoch(epoch)
        assert loss_fn.weights == [0.1, 0.9]


def test_mixed_loss_warmup_ramps_linearly_then_clamps():
    loss_fn = MixedLoss(
        loss_functions=[ConstantLoss(1.0), ConstantLoss(2.0)],
        weights=[1.0, 0.5],
        warmup_indices=[0],
        warmup_epochs=4,
    )

    # alpha = (epoch + 1) / warmup_epochs, so the ramp starts at one step of the
    # ramp rather than at 0 and reaches its target on the last warmup epoch.
    expected = [0.25, 0.5, 0.75, 1.0]
    for epoch, target in enumerate(expected):
        loss_fn.set_epoch(epoch)
        assert loss_fn.weights[0] == pytest.approx(target)
        # weights outside warmup_indices are never touched
        assert loss_fn.weights[1] == pytest.approx(0.5)

    # past the warmup window the weight stays at its target value
    for epoch in (4, 10, 200):
        loss_fn.set_epoch(epoch)
        assert loss_fn.weights[0] == pytest.approx(1.0)


def test_mixed_loss_warmup_scales_total_loss():
    loss_fn = MixedLoss(
        loss_functions=[ConstantLoss(4.0), ConstantLoss(1.0)],
        weights=[1.0, 2.0],
        warmup_indices=[0],
        warmup_epochs=2,
    )

    loss_fn.set_epoch(0)  # alpha = 0.5 -> 0.5 * 4.0 + 2.0 * 1.0
    assert loss_fn(None, None)["loss"].item() == pytest.approx(4.0)

    loss_fn.set_epoch(1)  # alpha = 1.0 -> 1.0 * 4.0 + 2.0 * 1.0
    assert loss_fn(None, None)["loss"].item() == pytest.approx(6.0)


@pytest.fixture
def loss_config():
    import yaml

    with open("tests/config/datamodule_test_base_config.yaml") as f:
        return yaml.safe_load(f)


def test_get_loss_function_ramps_physics_terms(loss_config):
    # config declares losses [LayeredWeightedPhysics, MaskedBusMSE]
    loss_config["training"]["physics_warmup_epochs"] = 10
    loss_fn = get_loss_function(NestedNamespace(**loss_config))

    assert loss_fn.warmup_indices == [0]
    assert loss_fn.warmup_epochs == 10

    loss_fn.set_epoch(4)
    assert loss_fn.weights[0] == pytest.approx(0.1 * 0.5)
    assert loss_fn.weights[1] == pytest.approx(0.9)


def test_get_loss_function_without_warmup_key_is_unchanged(loss_config):
    loss_fn = get_loss_function(NestedNamespace(**loss_config))

    assert loss_fn.warmup_epochs == 0
    loss_fn.set_epoch(3)
    assert loss_fn.weights == loss_fn.target_weights
