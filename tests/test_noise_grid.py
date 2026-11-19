"""Branch r, x noise used for joint state and parameter estimation."""

import pytest
import torch
import yaml
from torch_geometric.data import HeteroData

from gridfm_graphkit.datasets.globals import YFF_TT_I, YFF_TT_R, YFT_TF_I, YFT_TF_R
from gridfm_graphkit.datasets.normalizers import HeteroDataMVANormalizer
from gridfm_graphkit.datasets.transforms import (
    NoiseGridParameters,
    RemoveInactiveBranches,
    RemoveInactiveGenerators,
)
from gridfm_graphkit.io.param_handler import NestedNamespace

ADMITTANCE = [YFF_TT_R, YFF_TT_I, YFT_TF_R, YFT_TF_I]


def load(config_path, noise):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config["task"]["grid_noise_r"] = noise
    config["task"]["grid_noise_x"] = noise
    args = NestedNamespace(**config)
    data = HeteroData.from_dict(
        torch.load(
            "tests/data/case14_ieee/processed/data_index_0.pt",
            weights_only=True,
        ),
    )
    normalizer = HeteroDataMVANormalizer(args)
    normalizer.fit("tests/data/case14_ieee/raw", list(range(72)))
    normalizer.transform(data)
    data = RemoveInactiveGenerators()(RemoveInactiveBranches()(data))
    return args, data


CONFIG = "scripts/state_estimation/configs/case14_ieee_pe_0.yaml"


def test_zero_noise_leaves_data_unchanged():
    args, data = load(CONFIG, 0.0)
    before = data[("bus", "connects", "bus")].edge_attr.clone()
    out = NoiseGridParameters(args)(data)
    assert torch.equal(out[("bus", "connects", "bus")].edge_attr, before)
    assert "impedance" not in out.node_types


def test_noise_round_trip_and_range():
    args, data = load(CONFIG, 0.2)
    before = data[("bus", "connects", "bus")].edge_attr.clone()
    torch.manual_seed(0)
    out = NoiseGridParameters(args)(data)

    r, x = out["impedance"].y[:, 0], out["impedance"].y[:, 1]
    r_n, x_n = out["impedance"].x[:, 0], out["impedance"].x[:, 1]
    keep = r.abs() > 1e-8
    assert ((r_n[keep] / r[keep] - 1).abs() <= 0.2 + 1e-5).all()
    assert ((x_n / x - 1).abs() <= 0.2 + 1e-5).all()
    assert not torch.allclose(
        out[("bus", "connects", "bus")].edge_attr[:, ADMITTANCE],
        before[:, ADMITTANCE],
    )

    # Rebuilding the admittances from the true r, x gives the input back.
    _, data_check = load(CONFIG, 0.2)
    noise = NoiseGridParameters(args)
    noise.add_noise_admittance = lambda adm, sigma: adm
    rebuilt = noise(data_check)[("bus", "connects", "bus")].edge_attr
    assert torch.allclose(
        rebuilt[:, ADMITTANCE],
        before[:, ADMITTANCE],
        atol=1e-4,
        rtol=1e-4,
    )


@pytest.mark.parametrize("noise", [0.0, 0.2])
def test_noise_keeps_flows(noise):
    args, data = load(CONFIG, noise)
    before = data[("bus", "connects", "bus")].edge_attr[:, :2].clone()
    out = NoiseGridParameters(args)(data)
    assert torch.equal(out[("bus", "connects", "bus")].edge_attr[:, :2], before)
