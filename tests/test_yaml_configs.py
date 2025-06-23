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
