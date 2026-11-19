import torch
import yaml
from gridfm_graphkit.datasets.masking import SimulateMeasurements
from gridfm_graphkit.datasets.normalizers import HeteroDataMVANormalizer
from gridfm_graphkit.io.param_handler import NestedNamespace
from copy import deepcopy
from torch_geometric.data import HeteroData
from gridfm_graphkit.datasets.globals import (
    # Bus feature indices
    PD_H,
    QD_H,
    QG_H,
    VA_H,
    VM_H,
    # Generator feature indices
    PG_H,
    # Edge feature indices
    P_E,
    Q_E,
)


def test_simulate_measurements():
    data_dict = torch.load(
        "tests/data/case14_ieee/processed/data_index_0.pt",
        weights_only=True,
    )
    data = HeteroData.from_dict(data_dict)
    with open("tests/config/SE_simulate_measurements.yaml", "r") as f:
        args = yaml.safe_load(f)
    args = NestedNamespace(**args)
    normalizer = HeteroDataMVANormalizer(args)
    normalizer.fit("tests/data/case14_ieee/raw", list(range(72)))
    normalizer.transform(data)
    transform = SimulateMeasurements(args)
    out = transform(deepcopy(data))

    # Check that targets are not modified
    assert torch.isclose(out["bus"].y, data["bus"].y).all()
    assert torch.isclose(out["gen"].y, data["gen"].y).all()

    # Check that values different from PD, QD, QG, VM, VA (bus), PG (Gen) and
    # P_E, Q_E (edge_attr) have not been modified
    assert torch.isclose(
        out["bus"].x[:, VA_H + 1 :],
        data["bus"].x[:, VA_H + 1 :],
    ).all()
    assert torch.isclose(
        out["gen"].x[:, PG_H + 1 :],
        data["gen"].x[:, PG_H + 1 :],
    ).all()
    assert torch.isclose(
        out[("bus", "connects", "bus")]["edge_attr"][:, Q_E + 1 :],
        data[("bus", "connects", "bus")]["edge_attr"][:, Q_E + 1 :],
    ).all()

    # Check that masked values have not been modified
    mask_bus = out.mask_dict["bus"][:, [PD_H, QD_H, QG_H, VM_H, VA_H]]
    assert torch.isclose(
        out["bus"].x[:, [PD_H, QD_H, QG_H, VM_H, VA_H]][mask_bus],
        data["bus"].x[:, [PD_H, QD_H, QG_H, VM_H, VA_H]][mask_bus],
    ).all()

    mask_gen = out.mask_dict["gen"][:, [PG_H]]
    assert torch.isclose(
        out["gen"].x[:, [PG_H]][mask_gen],
        data["gen"].x[:, [PG_H]][mask_gen],
    ).all()

    mask_branch = out.mask_dict["branch"][:, [P_E, Q_E]]
    assert torch.isclose(
        out[("bus", "connects", "bus")]["edge_attr"][:, [P_E, Q_E]][mask_branch],
        data[("bus", "connects", "bus")]["edge_attr"][:, [P_E, Q_E]][mask_branch],
    ).all()

    # Check that the masked values have inf std associated
    mask_bus_small = out.mask_dict["bus"][:, : out.mask_dict["std_bus"].size(1)]
    mask_gen_small = out.mask_dict["gen"][:, : out.mask_dict["std_gen"].size(1)]
    mask_branch_small = out.mask_dict["branch"][
        :,
        : out.mask_dict["std_branch"].size(1),
    ]
    assert (mask_bus_small == torch.isinf(out.mask_dict["std_bus"])).all()
    assert (mask_gen_small == torch.isinf(out.mask_dict["std_gen"])).all()
    assert (mask_branch_small == torch.isinf(out.mask_dict["std_branch"])).all()

    # Check that outliers are not masked
    assert (~mask_bus_small >= out.mask_dict["outliers_bus"]).all()
    assert (~mask_branch_small >= out.mask_dict["outliers_branch"]).all()


if __name__ == "__main__":
    test_simulate_measurements()
