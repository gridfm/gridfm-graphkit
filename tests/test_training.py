import yaml
import glob
import pytest
import torch
import os
import tempfile
from torch import optim
from torch_geometric.loader import DataLoader

from gridfm_graphkit.io.param_handler import (
    load_normalizer,
    get_loss_function,
    load_model,
    get_transform,
    NestedNamespace,
)
from gridfm_graphkit.datasets.utils import split_dataset
from gridfm_graphkit.datasets.powergrid import GridDatasetMem
from gridfm_graphkit.training.callbacks import EarlyStopper
from gridfm_graphkit.training.trainer import Trainer


@pytest.mark.parametrize("yaml_path", glob.glob("tests/config/*.yaml"))
def test_training_loop(yaml_path):
    with open(yaml_path) as f:
        config_dict = yaml.safe_load(f)

    args = NestedNamespace(**config_dict)

    # Set device early
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load components
    model = load_model(args).to(device)
    loss_fn = get_loss_function(args)
    node_normalizer, edge_normalizer = load_normalizer(args)

    # Dataset
    data_path_network = os.path.join("tests/data", args.data.network)
    dataset = GridDatasetMem(
        root=data_path_network,
        norm_method=args.data.normalization,
        node_normalizer=node_normalizer,
        edge_normalizer=edge_normalizer,
        pe_dim=args.model.pe_dim,
        mask_dim=args.data.mask_dim,
        transform=get_transform(args=args),
    )

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.optimizer.learning_rate,
        betas=(args.optimizer.beta1, args.optimizer.beta2),
    )

    # Use temporary directory for checkpoint saving
    with tempfile.TemporaryDirectory() as tmpdir:
        train_dataset, val_dataset, _ = split_dataset(dataset=dataset, log_dir=tmpdir)

        dataloader_train = DataLoader(
            train_dataset,
            batch_size=args.training.batch_size,
        )
        dataloader_val = DataLoader(val_dataset, batch_size=args.training.batch_size)
        path_to_save = os.path.join(tmpdir, "checkpoint.pt")

        early_stopper = EarlyStopper(
            path_to_save,
            args.callbacks.patience,
            args.callbacks.tol,
        )

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            device=device,
            loss_fn=loss_fn,
            early_stopper=early_stopper,
            train_dataloader=dataloader_train,
            val_dataloader=dataloader_val,
        )

        trainer.train(start_epoch=0, epochs=args.training.epochs)

        # Optionally check that checkpoint exists
        if not os.path.exists(path_to_save):
            raise AssertionError(
                f"Expected checkpoint at {path_to_save} was not found.",
            )
