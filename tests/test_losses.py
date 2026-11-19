import pytest
import torch
from gridfm_graphkit.datasets.hetero_powergrid_datamodule import LitGridHeteroDataModule
from gridfm_graphkit.io.param_handler import NestedNamespace
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
