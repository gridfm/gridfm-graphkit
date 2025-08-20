import pytest
import torch
from gridfm_graphkit.datasets.powergrid_datamodule import LitGridDataModule
from gridfm_graphkit.io.param_handler import NestedNamespace
from gridfm_graphkit.training.loss import PBELoss


@pytest.fixture
def small_grid_data_module():
    # Load config
    import yaml

    with open("tests/config/datamodule_test_base_config.yaml") as f:
        config_dict = yaml.safe_load(f)

    args = NestedNamespace(**config_dict)
    dm = LitGridDataModule(args, data_dir="tests/data")

    # Fake trainer for setup
    class DummyTrainer:
        is_global_zero = True

    dm.trainer = DummyTrainer()
    dm.setup("train")
    return dm


def test_pbe_loss_zero_with_real_data(small_grid_data_module):
    loader = small_grid_data_module.train_dataloader()
    batch = next(iter(loader))

    loss_fn = PBELoss()
    out = loss_fn(batch.y, batch.y, batch.edge_index, batch.edge_attr, batch.mask)
    assert torch.allclose(out["loss"], torch.tensor(0.0), atol=1e-5), (
        f"PBELoss is not zero! Got {out['loss'].item()}"
    )
