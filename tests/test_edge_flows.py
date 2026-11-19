import torch
from gridfm_graphkit.datasets.normalizers import HeteroDataMVANormalizer
from gridfm_graphkit.io.param_handler import NestedNamespace
import yaml
from gridfm_graphkit.models.utils import (
    ComputeBranchFlow,
    ComputeNodeInjection,
    compute_shunt_power,
)
from gridfm_graphkit.training.loss import LossPerDim
from gridfm_graphkit.datasets.globals import (
    VM_H,
    VA_H,
    P_E,
    Q_E,
)
from torch_geometric.data import HeteroData


def test_edge_flows():
    data_dict = torch.load(
        "tests/data/case14_ieee/processed/data_index_0.pt",
        weights_only=True,
    )
    data = HeteroData.from_dict(data_dict)

    with open("tests/config/datamodule_test_base_config.yaml", "r") as f:
        args = yaml.safe_load(f)
    args = NestedNamespace(**args)
    normalizer = HeteroDataMVANormalizer(args)
    normalizer.fit("tests/data/case14_ieee/raw", list(range(72)))
    normalizer.transform(data)

    bus_edge_index = data[("bus", "connects", "bus")].edge_index
    bus_edge_attr = data[("bus", "connects", "bus")].edge_attr
    branch_flow_layer = ComputeBranchFlow()
    node_injection_layer = ComputeNodeInjection()

    Pft, Qft = branch_flow_layer(
        data["bus"].x[:, [VM_H, VA_H]],
        bus_edge_index,
        bus_edge_attr,
    )

    assert torch.isclose(Pft, bus_edge_attr[:, P_E], atol=1e-4).all()
    assert torch.isclose(Qft, bus_edge_attr[:, Q_E], atol=1e-4).all()

    P_in, Q_in = node_injection_layer(
        Pft,
        Qft,
        bus_edge_index,
        data["bus"].x.size(0),
    )

    # Final SE output: voltages and the injections they imply, as in GNS_heterogeneous.
    voltages = data["bus"].y[:, [VM_H, VA_H]]
    p_shunt, q_shunt = compute_shunt_power(voltages, data["bus"].x)
    output_temp = torch.stack(
        [voltages[:, 0], voltages[:, 1], P_in - p_shunt, Q_in - q_shunt],
        dim=1,
    )

    pred_dict = {"bus": output_temp, "gen": data["gen"].y}
    target_dict = {"bus": data["bus"].y, "gen": data["gen"].y}
    edge_index = {
        ("gen", "connected_to", "bus"): data[("gen", "connected_to", "bus")].edge_index,
    }
    assert (
        LossPerDim(NestedNamespace(**{"loss_str": "MAE", "dim": "VM"}), None)(
            pred_dict,
            target_dict,
            edge_index,
            None,
            None,
        )["loss"]
        < 1e-4
    )
    assert (
        LossPerDim(NestedNamespace(**{"loss_str": "MAE", "dim": "VA"}), None)(
            pred_dict,
            target_dict,
            edge_index,
            None,
            None,
        )["loss"]
        < 1e-4
    )
    assert (
        LossPerDim(NestedNamespace(**{"loss_str": "MAE", "dim": "P_in"}), None)(
            pred_dict,
            target_dict,
            edge_index,
            None,
            None,
        )["loss"]
        < 1e-4
    )
    assert (
        LossPerDim(NestedNamespace(**{"loss_str": "MAE", "dim": "Q_in"}), None)(
            pred_dict,
            target_dict,
            edge_index,
            None,
            None,
        )["loss"]
        < 1e-4
    )


if __name__ == "__main__":
    test_edge_flows()
