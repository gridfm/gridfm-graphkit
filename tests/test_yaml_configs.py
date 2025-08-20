import yaml
import glob
import pytest
from gridfm_graphkit.io.param_handler import (
    load_normalizer,
    get_loss_function,
    load_model,
    get_transform,
    NestedNamespace,
)


@pytest.mark.parametrize("yaml_path", glob.glob("examples/config/*.yaml"))
def test_yaml_config_valid(yaml_path):
    with open(yaml_path) as f:
        config_dict = yaml.safe_load(f)

    args = NestedNamespace(**config_dict)
    # Call your param handler functions; they should not raise exceptions
    load_normalizer(args)
    get_transform(args)
    if hasattr(args, "model"):
        load_model(args)
    if hasattr(args, "training") and hasattr(args.training, "losses"):
        get_loss_function(args)


def test_nested_namespace_with_list_and_flatten():
    cfg = {
        "seed": 42,
        "data": {"networks": ["case30_ieee", "case118_ieee"]},
        "training": {"batch_size": 16},
    }

    ns = NestedNamespace(**cfg)

    # Direct attribute and list access
    assert ns.seed == 42
    assert ns.data.networks[0] == "case30_ieee"

    # Round-trip back to dict
    assert ns.to_dict() == cfg

    # Flatten with default and custom separators
    flat = ns.flatten()
    assert flat["training.batch_size"] == 16
    assert flat["data.networks"] == ["case30_ieee", "case118_ieee"]
    assert ns.flatten(sep="/")["data/networks"] == ["case30_ieee", "case118_ieee"]
