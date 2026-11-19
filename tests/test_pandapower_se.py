"""WLS baseline on the bundled IEEE 14-bus scenarios."""

from copy import deepcopy

import pytest
import torch
import yaml
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_add

from gridfm_graphkit.datasets.globals import (
    BS,
    GS,
    PD_H,
    PG_H,
    QD_H,
    QG_H,
    VA_H,
    VM_H,
    YFF_TT_R,
    YFT_TF_I,
)
from gridfm_graphkit.datasets.normalizers import HeteroDataMVANormalizer
from gridfm_graphkit.datasets.task_transforms import StateEstimationTransforms
from gridfm_graphkit.io.param_handler import NestedNamespace

pytest.importorskip("pandapower")
from gridfm_graphkit.models.pandapower_se import PandapowerSE  # noqa: E402

DATA = "tests/data/case14_ieee"
CONFIG = "scripts/state_estimation/configs/case14_ieee_base_0.yaml"
NUM_SCENARIOS = 72


def make_args(mask_ratio, outlier_ratio):
    with open(CONFIG) as f:
        config = yaml.safe_load(f)
    for name in ["vm", "power_inj", "power_flow"]:
        config["task"]["measurements"][name]["mask_ratio"] = mask_ratio
        config["task"]["measurements"][name]["outlier_ratio"] = outlier_ratio
    return NestedNamespace(**config)


def run_wls(args, scenario):
    """Measurements for one scenario, denormalized the same way as StateEstimationTask."""
    normalizer = HeteroDataMVANormalizer(args)
    normalizer.fit(f"{DATA}/raw", list(range(NUM_SCENARIOS)))
    data = HeteroData.from_dict(
        torch.load(f"{DATA}/processed/data_index_{scenario}.pt", weights_only=True),
    )
    truth = deepcopy(data)
    normalizer.transform(data)
    data = StateEstimationTransforms(args)(data)

    batch = next(iter(DataLoader([data], batch_size=1)))
    normalizer.inverse_transform(batch)
    normalizer.inverse_std(batch)
    batch.x_dict["bus"][:, VA_H] *= 180 / torch.pi
    batch.edge_attr_dict[("bus", "connects", "bus")][:, YFF_TT_R : YFT_TF_I + 1] /= (
        normalizer.baseMVA_orig
    )
    batch.x_dict["bus"][:, [GS, BS]] /= normalizer.baseMVA_orig
    out = PandapowerSE(args)(
        x_dict=batch.x_dict,
        edge_index_dict=batch.edge_index_dict,
        edge_attr_dict=batch.edge_attr_dict,
        mask_dict=batch.mask_dict,
        baseMVA_orig=normalizer.baseMVA_orig,
    )
    return out, truth


def target(truth):
    num_bus = truth["bus"].y.shape[0]
    gen_to_bus = truth[("gen", "connected_to", "bus")].edge_index[1]
    pg = scatter_add(truth["gen"].y[:, PG_H], gen_to_bus, dim=0, dim_size=num_bus)
    return torch.stack(
        [
            truth["bus"].y[:, VM_H],
            truth["bus"].y[:, VA_H],
            pg - truth["bus"].y[:, PD_H],
            truth["bus"].y[:, QG_H] - truth["bus"].y[:, QD_H],
        ],
        dim=1,
    ).double()


@pytest.mark.parametrize("scenario", [0, 1])
def test_wls_recovers_state_with_full_measurements(scenario):
    out, truth = run_wls(make_args(0.0, 0.0), scenario)
    assert out["success"]
    error = (out["bus"] - target(truth)).abs().mean(dim=0)
    # Measurement noise is 1% on Vm and 2% on powers.
    assert error[0] < 5e-3  # Vm, p.u.
    assert error[1] < 0.2  # Va, degrees
    assert error[2] < 1.0  # P_inj, MW
    # Q_inj includes the bus shunt injections.
    assert error[3] < 1.0  # Q_inj, MVAr


def test_wls_is_deterministic():
    args = make_args(0.2, 0.1)
    first, _ = run_wls(args, 3)
    second, _ = run_wls(args, 3)
    assert first["success"] == second["success"]
    assert torch.equal(first["bus"], second["bus"])


def test_wls_removes_outliers():
    out, truth = run_wls(make_args(0.0, 0.2), 0)
    assert out["success"]
    assert out["bad_data_cycles"] > 0
    assert (out["bus"][:, 0] - target(truth)[:, 0]).abs().mean() < 1e-2
