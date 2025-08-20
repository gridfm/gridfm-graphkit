import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import ConcatDataset
from torch.utils.data import Subset
from gridfm_graphkit.io.param_handler import (
    NestedNamespace,
    load_normalizer,
    get_transform,
)
from gridfm_graphkit.datasets.utils import split_dataset
from gridfm_graphkit.datasets.powergrid_dataset import GridDatasetDisk
import numpy as np
import random
import warnings
import os
import lightning as L


class LitGridDataModule(L.LightningDataModule):
    """ """

    def __init__(self, args: NestedNamespace, data_dir: str = "./data"):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = int(args.training.batch_size)
        self.args = args
        self.node_normalizers = []
        self.edge_normalizers = []
        self.datasets = []
        self.train_datasets = []
        self.val_datasets = []
        self.test_datasets = []
        self._is_setup_done = False

    def setup(self, stage: str):
        if self._is_setup_done:
            print(f"Setup already done for stage={stage}, skipping...")
            return

        for i, network in enumerate(self.args.data.networks):
            node_normalizer, edge_normalizer = load_normalizer(args=self.args)
            self.node_normalizers.append(node_normalizer)
            self.edge_normalizers.append(edge_normalizer)

            # Create torch dataset and split
            data_path_network = os.path.join(self.data_dir, network)

            # Run preprocessing only on rank 0
            if self.trainer.is_global_zero:
                print(f"Pre-processing of {network} dataset on rank 0")
                _ = GridDatasetDisk(  # just to trigger processing
                    root=data_path_network,
                    norm_method=self.args.data.normalization,
                    node_normalizer=node_normalizer,
                    edge_normalizer=edge_normalizer,
                    pe_dim=self.args.model.pe_dim,
                    mask_dim=self.args.data.mask_dim,
                    transform=get_transform(args=self.args),
                )

            # All ranks wait here until processing is done
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()

            dataset = GridDatasetDisk(
                root=data_path_network,
                norm_method=self.args.data.normalization,
                node_normalizer=node_normalizer,
                edge_normalizer=edge_normalizer,
                pe_dim=self.args.model.pe_dim,
                mask_dim=self.args.data.mask_dim,
                transform=get_transform(args=self.args),
            )
            self.datasets.append(dataset)

            num_scenarios = self.args.data.scenarios[i]
            if num_scenarios > len(dataset):
                warnings.warn(
                    f"Requested number of scenarios ({num_scenarios}) exceeds dataset size ({len(dataset)}). "
                    "Using the full dataset instead.",
                )
                num_scenarios = len(dataset)

            # Create a subset
            all_indices = list(range(len(dataset)))
            # Random seed set before every shuffle for reproducibility in case the power grid datasets are analyzed in a different order
            random.seed(self.args.seed)
            random.shuffle(all_indices)
            subset_indices = all_indices[:num_scenarios]
            dataset = Subset(dataset, subset_indices)

            # Random seed set before every split, same as above
            np.random.seed(self.args.seed)
            train_dataset, val_dataset, test_dataset = split_dataset(
                dataset,
                self.data_dir,
                self.args.data.val_ratio,
                self.args.data.test_ratio,
            )

            self.train_datasets.append(train_dataset)
            self.val_datasets.append(val_dataset)
            self.test_datasets.append(test_dataset)

        self.train_dataset_multi = ConcatDataset(self.train_datasets)
        self.val_dataset_multi = ConcatDataset(self.val_datasets)
        self._is_setup_done = True

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset_multi,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.args.data.workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset_multi,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.args.data.workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return [
            DataLoader(
                i,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.args.data.workers,
                pin_memory=True,
            )
            for i in self.test_datasets
        ]

    def predict_dataloader(self):
        return [
            DataLoader(
                i,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.args.data.workers,
                pin_memory=True,
            )
            for i in self.test_datasets
        ]
