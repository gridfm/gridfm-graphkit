from gridfm_graphkit.datasets.powergrid_datamodule import LitGridDataModule
from gridfm_graphkit.io.param_handler import NestedNamespace
import yaml
import copy
import pytest

with open("tests/config/datamodule_test_base_config.yaml") as f:
    BASE_CONFIG = yaml.safe_load(f)


@pytest.mark.parametrize(
    "normalization",
    ["minmax", "standard", "baseMVAnorm", "identity"],
)
@pytest.mark.parametrize("mask_type", ["rnd", "pf", "opf", "none"])
def test_dataloaders(normalization, mask_type):
    cfg = copy.deepcopy(BASE_CONFIG)

    # Override values
    cfg["data"]["normalization"] = normalization
    cfg["data"]["mask_type"] = mask_type

    args = NestedNamespace(**cfg)

    dm = LitGridDataModule(args, data_dir="tests/data")

    # Lightning will inject trainer, but for testing we can fake it
    class DummyTrainer:
        is_global_zero = True

    dm.trainer = DummyTrainer()

    # Run setup
    dm.setup("test")

    # Check datasets exist
    assert len(dm.train_datasets) > 0
    assert len(dm.val_datasets) > 0
    assert len(dm.test_datasets) > 0

    # Train dataloader should yield batches
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    assert batch is not None

    # Val dataloader should yield batches
    val_loader = dm.val_dataloader()
    batch = next(iter(val_loader))
    assert batch is not None

    # Test dataloader should yield batches
    test_loaders = dm.test_dataloader()
    assert isinstance(test_loaders, list)
    batch = next(iter(test_loaders[0]))
    assert batch is not None

    # Predict dataloader should mirror test
    predict_loaders = dm.predict_dataloader()
    batch = next(iter(predict_loaders[0]))
    assert batch is not None
