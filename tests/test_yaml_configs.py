import yaml
import glob
import pytest
from gridfm_graphkit.io.param_handler import (
    load_normalizer,
    get_loss_function,
    load_model,
    get_task,
    get_task_transforms,
    get_physics_decoder,
    NestedNamespace,
)


@pytest.mark.parametrize("yaml_path", glob.glob("scripts/config/*.yaml"))
def test_yaml_config_valid(yaml_path):
    with open(yaml_path) as f:
        config_dict = yaml.safe_load(f)

    args = NestedNamespace(**config_dict)
    # Call your param handler functions; they should not raise exceptions
    normalizer = load_normalizer(args)
    get_task(args, normalizer)
    load_model(args)
    get_loss_function(args)
    get_task_transforms(args)
    get_physics_decoder(args)
