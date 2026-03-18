import json
import torch
import os
from torch_geometric.loader import DataLoader
from torch.utils.data import ConcatDataset
from torch.utils.data import Subset
import torch.distributed as dist
from gridfm_graphkit.io.registries import DATASET_WRAPPER_REGISTRY
from gridfm_graphkit.io.param_handler import (
    NestedNamespace,
    load_normalizer,
    get_task_transforms,
)
from gridfm_graphkit.datasets.utils import (
    split_dataset,
    split_dataset_by_load_scenario_idx,
)
from gridfm_graphkit.datasets.powergrid_hetero_dataset import HeteroGridDatasetDisk
import numpy as np
import random
import warnings
import os
import lightning as L
from typing import List
from lightning.pytorch.loggers import MLFlowLogger


class LitGridHeteroDataModule(L.LightningDataModule):
    """
    PyTorch Lightning DataModule for power grid datasets.

    This datamodule handles loading, preprocessing, splitting, and batching
    of power grid graph datasets (`GridDatasetDisk`) for training, validation,
    testing, and prediction. It ensures reproducibility through fixed seeds.

    Args:
        args (NestedNamespace): Experiment configuration.
        data_dir (str, optional): Root directory for datasets. Defaults to "./data".

    Attributes:
        batch_size (int): Batch size for all dataloaders. From ``args.training.batch_size``
        data_normalizers (list): List of data normalizers, one per dataset.
        datasets (list): Original datasets for each network.
        train_datasets (list): Train splits for each network.
        val_datasets (list): Validation splits for each network.
        test_datasets (list): Test splits for each network.
        train_dataset_multi (ConcatDataset): Concatenated train datasets for multi-network training.
        val_dataset_multi (ConcatDataset): Concatenated validation datasets for multi-network validation.
        _is_setup_done (bool): Tracks whether `setup` has been executed to avoid repeated processing.

    Methods:
        setup(stage):
            Load and preprocess datasets, split into train/val/test, and store normalizers.
            Handles distributed preprocessing safely.
        train_dataloader():
            Returns a DataLoader for concatenated training datasets.
        val_dataloader():
            Returns a DataLoader for concatenated validation datasets.
        test_dataloader():
            Returns a list of DataLoaders, one per test dataset.
        predict_dataloader():
            Returns a list of DataLoaders, one per test dataset for prediction.

    Notes:
        - Preprocessing is only performed on rank 0 in distributed settings.
        - Subsets and splits are deterministic based on the provided random seed.
        - Normalizers are loaded for each network independently.
        - Test and predict dataloaders are returned as lists, one per dataset.

    Example:
        ```python
        from gridfm_graphkit.datasets.powergrid_datamodule import LitGridDataModule
        from gridfm_graphkit.io.param_handler import NestedNamespace
        import yaml

        with open("config/config.yaml") as f:
            base_config = yaml.safe_load(f)
        args = NestedNamespace(**base_config)

        datamodule = LitGridDataModule(args, data_dir="./data")

        datamodule.setup("fit")
        train_loader = datamodule.train_dataloader()
        ```
    """

    def __init__(
        self,
        args: NestedNamespace,
        data_dir: str = "./data",
        normalizer_stats_path: str = None,
        dataset_wrapper: str = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.dataset_wrapper = dataset_wrapper
        self.batch_size = int(args.training.batch_size)
        self.split_by_load_scenario_idx = getattr(
            args.data,
            "split_by_load_scenario_idx",
            False,
        )
        self.args = args
        self.normalizer_stats_path = normalizer_stats_path
        self.data_normalizers = []
        self.datasets = []
        self.train_datasets = []
        self.val_datasets = []
        self.test_datasets = []
        self.train_scenario_ids: List[List[int]] = []
        self.val_scenario_ids: List[List[int]] = []
        self.test_scenario_ids: List[List[int]] = []
        self._is_setup_done = False

    def setup(self, stage: str):
        if self._is_setup_done:
            print(f"Setup already done for stage={stage}, skipping...")
            return

        # Load pre-fitted normalizer stats if provided (e.g. from a training run)
        saved_stats = None
        if self.normalizer_stats_path is not None:
            saved_stats = torch.load(
                self.normalizer_stats_path,
                map_location="cpu",
                weights_only=True,
            )
            print(f"Loaded normalizer stats from {self.normalizer_stats_path}")

        for i, network in enumerate(self.args.data.networks):
            data_normalizer = load_normalizer(args=self.args)
            self.data_normalizers.append(data_normalizer)

            # Create torch dataset (normalizer is NOT yet fitted)
            data_path_network = os.path.join(self.data_dir, network)

            is_distributed = dist.is_available() and dist.is_initialized()

            if not is_distributed or dist.get_rank() == 0:
                dataset = HeteroGridDatasetDisk(
                    root=data_path_network,
                    data_normalizer=data_normalizer,
                    transform=get_task_transforms(args=self.args),
                )

            # All ranks wait here until rank 0 processing is done
            if is_distributed:
                dist.barrier()

            if is_distributed and dist.get_rank() != 0:
                dataset = HeteroGridDatasetDisk(
                    root=data_path_network,
                    data_normalizer=data_normalizer,
                    transform=get_task_transforms(args=self.args),
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

            # load_scenario for each scenario in the subset
            load_scenarios = dataset.load_scenarios[subset_indices]

            dataset = Subset(dataset, subset_indices)

            if self.dataset_wrapper is not None:
                wrapper_cls = DATASET_WRAPPER_REGISTRY.get(self.dataset_wrapper)
                dataset = wrapper_cls(dataset)

            # Random seed set before every split, same as above
            np.random.seed(self.args.seed)
            if self.split_by_load_scenario_idx:
                train_dataset, val_dataset, test_dataset = (
                    split_dataset_by_load_scenario_idx(
                        dataset,
                        self.data_dir,
                        load_scenarios,
                        self.args.data.val_ratio,
                        self.args.data.test_ratio,
                    )
                )
            else:
                train_dataset, val_dataset, test_dataset = split_dataset(
                    dataset,
                    self.data_dir,
                    self.args.data.val_ratio,
                    self.args.data.test_ratio,
                )

            # Extract scenario IDs for each split
            train_scenario_ids = self._extract_scenario_ids(
                train_dataset,
                subset_indices,
            )
            val_scenario_ids = self._extract_scenario_ids(
                val_dataset,
                subset_indices,
            )
            test_scenario_ids = self._extract_scenario_ids(
                test_dataset,
                subset_indices,
            )

            # Fit normalizer: restore from saved stats only for fit_on_train
            # normalizers (global baseMVA must match the model's training run).
            # fit_on_dataset normalizers compute per-scenario stats and must
            # always fit on the actual scenarios being used.
            use_saved = (
                saved_stats is not None
                and network in saved_stats
                and data_normalizer.fit_strategy == "fit_on_train"
            )
            if use_saved:
                print(f"Restoring normalizer for {network} from saved stats")
                data_normalizer.fit_from_dict(saved_stats[network])
            else:
                self._fit_normalizer(
                    data_normalizer,
                    data_path_network,
                    network,
                    train_scenario_ids,
                    val_scenario_ids,
                    test_scenario_ids,
                    num_scenarios,
                    saved_stats,
                )

            self.train_datasets.append(train_dataset)
            self.val_datasets.append(val_dataset)
            self.test_datasets.append(test_dataset)
            self.train_scenario_ids.append(train_scenario_ids)
            self.val_scenario_ids.append(val_scenario_ids)
            self.test_scenario_ids.append(test_scenario_ids)

        self.train_dataset_multi = ConcatDataset(self.train_datasets)
        self.val_dataset_multi = ConcatDataset(self.val_datasets)
        self._is_setup_done = True

        # Save scenario splits (rank 0 only in DDP)
        is_rank0 = (
            not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0
        )
        if is_rank0 and self.trainer is not None and self.trainer.logger is not None:
            logger = self.trainer.logger
            if isinstance(logger, MLFlowLogger):
                log_dir = os.path.join(
                    logger.save_dir,
                    logger.experiment_id,
                    logger.run_id,
                    "artifacts",
                    "stats",
                )
            else:
                log_dir = os.path.join(logger.save_dir, "stats")
            self.save_scenario_splits(log_dir)

    @staticmethod
    def _fit_normalizer(
        data_normalizer,
        data_path_network,
        network,
        train_scenario_ids,
        val_scenario_ids,
        test_scenario_ids,
        num_scenarios,
        saved_stats,
    ):
        """
        Fit normalizer from raw data. In distributed settings, only rank 0
        reads the parquet files and computes stats; the result is broadcast
        to all other ranks via fit_from_dict.
        """
        is_distributed = dist.is_available() and dist.is_initialized()
        is_rank0 = not is_distributed or dist.get_rank() == 0

        raw_data_path = os.path.join(data_path_network, "raw")
        stats = None

        if is_rank0:
            if data_normalizer.fit_strategy == "fit_on_train":
                if saved_stats is not None and network not in saved_stats:
                    warnings.warn(
                        f"No saved normalizer stats found for network '{network}'. "
                        "Fitting from data instead.",
                    )
                print(
                    f"Fitting normalizer on train set ({len(train_scenario_ids)} scenarios)",
                )
                stats = data_normalizer.fit(raw_data_path, train_scenario_ids)
            elif data_normalizer.fit_strategy == "fit_on_dataset":
                all_scenario_ids = (
                    train_scenario_ids + val_scenario_ids + test_scenario_ids
                )
                assert np.unique(all_scenario_ids).shape[0] == num_scenarios
                print(
                    f"Fitting normalizer on full dataset ({len(all_scenario_ids)} scenarios)",
                )
                stats = data_normalizer.fit(raw_data_path, all_scenario_ids)
            else:
                raise ValueError(
                    f"Unknown fit_strategy: {data_normalizer.fit_strategy}",
                )

        if is_distributed:
            stats_list = [stats]
            dist.broadcast_object_list(stats_list, src=0)
            stats = stats_list[0]
            if dist.get_rank() != 0:
                data_normalizer.fit_from_dict(stats)

    @staticmethod
    def _extract_scenario_ids(
        subset: Subset,
        subset_indices: List[int],
    ) -> List[int]:
        """
        Extract original scenario IDs from a Subset.

        The subset's indices point into an outer Subset defined by subset_indices,
        so we map: original_scenario_id = subset_indices[subset_idx].
        """
        indices = subset.indices
        if isinstance(indices, torch.Tensor):
            indices = indices.flatten().tolist()
        elif not isinstance(indices, list):
            indices = list(indices)
        return [subset_indices[idx] for idx in indices]

    def save_scenario_splits(self, log_dir: str):
        """Save train/val/test scenario ID splits to JSON files."""
        os.makedirs(log_dir, exist_ok=True)
        for i, network in enumerate(self.args.data.networks):
            splits = {
                "train": self.train_scenario_ids[i],
                "val": self.val_scenario_ids[i],
                "test": self.test_scenario_ids[i],
            }
            splits_path = os.path.join(log_dir, f"{network}_scenario_splits.json")
            with open(splits_path, "w") as f:
                json.dump(splits, f, indent=2)

    def _dataloader_kwargs(self):
        num_workers = self.args.data.workers
        kwargs = dict(
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )
        return kwargs

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset_multi,
            batch_size=self.batch_size,
            shuffle=True,
            **self._dataloader_kwargs(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset_multi,
            batch_size=self.batch_size,
            shuffle=False,
            **self._dataloader_kwargs(),
        )

    def test_dataloader(self):
        return [
            DataLoader(
                i,
                batch_size=self.batch_size,
                shuffle=False,
                **self._dataloader_kwargs(),
            )
            for i in self.test_datasets
        ]

    def predict_dataloader(self):
        return [
            DataLoader(
                i,
                batch_size=self.batch_size,
                shuffle=False,
                **self._dataloader_kwargs(),
            )
            for i in self.test_datasets
        ]
