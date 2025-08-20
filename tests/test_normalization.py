import torch
import pytest
from gridfm_graphkit.io.registries import NORMALIZERS_REGISTRY
from gridfm_graphkit.datasets.normalizers import BaseMVANormalizer
from gridfm_graphkit.io.param_handler import NestedNamespace


@pytest.mark.parametrize("norm_name", list(NORMALIZERS_REGISTRY))
@pytest.mark.parametrize("node_data", [True, False])
def test_normalizer_roundtrip_registry(norm_name, node_data):
    # Example input data
    data = torch.randn(3, 6)

    # Dummy args as a dictionary and converted to NestedNamespace
    args_dict = {"data": {"baseMVA": 100}}
    args = NestedNamespace(**args_dict)

    norm = NORMALIZERS_REGISTRY.create(norm_name, node_data, args)

    # Fit and transform
    if isinstance(norm, BaseMVANormalizer) and not node_data:
        norm.fit(data, baseMVA=100)
    else:
        norm.fit(data)

    transformed = norm.transform(data.clone())
    restored = norm.inverse_transform(transformed.clone())

    # Check round-trip equality
    assert torch.allclose(restored, data, atol=1e-6), (
        f"Failed round-trip for {norm_name}"
    )

    # Check stats dict is present
    stats = norm.get_stats()
    assert isinstance(stats, dict), f"Stats not dict for {norm_name}"
