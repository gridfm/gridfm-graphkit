import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import ConcatDataset
from torch.utils.data import Subset
import torch.distributed as dist
from gridfm_graphkit.io.param_handler import (
    NestedNamespace,
    load_normalizer,
    get_task_transforms,
)
from gridfm_graphkit.datasets.utils import split_dataset
from gridfm_graphkit.datasets.powergrid_hetero_dataset import HeteroGridDatasetDisk
import numpy as np
import random
import warnings
import os
import lightning as L


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

    def __init__(self, args: NestedNamespace, data_dir: str = "./data"):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = int(args.training.batch_size)
        self.args = args
        self.data_normalizers = []
        self.datasets = []
        self.train_datasets = []
        self.val_datasets = []
        self.test_datasets = []
        self._is_setup_done = False


    def setup(self, stage: str):
        """
        Build train/val/test datasets by:
        - Scanning all grids in self.data_dir
        - Excluding explicit validation/test networks from the pool
        - Selecting 'training_networks' count from the remaining pool (seeded shuffle)
        - Preprocessing each network on rank-0 with barrier synchronization
        - Concatenating all training datasets into self.train_dataset_multi
        - Concatenating all validation datasets into self.val_dataset_multi
        - Keeping per-network test datasets in self.test_datasets (same concept as before)
        """
        if self._is_setup_done:
            print(f"Setup already done for stage={stage}, skipping...")
            return

        # Alias for convenience (handles both import styles)
        dist = torch.distributed

        def is_main_rank() -> bool:
            return dist.is_available() and dist.is_initialized() and dist.get_rank() == 0

        # --- 1) Discover all networks present in data_dir ---
        try:
            all_entries = os.listdir(self.data_dir)
        except FileNotFoundError:
            raise RuntimeError(f"Data directory not found: {self.data_dir}")

        all_networks = sorted(
            d for d in all_entries if os.path.isdir(os.path.join(self.data_dir, d))
        )
        if not all_networks:
            raise RuntimeError(f"No grid folders found under: {self.data_dir}")

        # --- 2) Read desired splits from args ---
        # Expect:
        #   self.args.data.training_networks: int
        #   self.args.data.validation_networks: list[str] | None
        #   self.args.data.test_networks: list[str] | None
        training_count = int(getattr(self.args.data, "training_networks", 0) or 0)
        requested_val = list(getattr(self.args.data, "validation_networks", []) or [])
        requested_test = list(getattr(self.args.data, "test_networks", []) or [])

        # Keep only networks that actually exist; warn on missing
        val_networks = [n for n in requested_val if n in all_networks]
        missing_val = sorted(set(requested_val) - set(val_networks))
        if missing_val:
            warnings.warn(
                f"The following validation networks are not present in {self.data_dir} and will be ignored: {missing_val}"
            )

        test_networks = [n for n in requested_test if n in all_networks]
        missing_test = sorted(set(requested_test) - set(test_networks))
        if missing_test:
            warnings.warn(
                f"The following test networks are not present in {self.data_dir} and will be ignored: {missing_test}"
            )

        # --- 3) Build training pool by excluding val/test and sampling deterministically ---
        excluded = set(val_networks) | set(test_networks)
        train_candidates = [n for n in all_networks if n not in excluded]

        if training_count > len(train_candidates):
            warnings.warn(
                f"Requested training_networks={training_count} exceeds available train candidates "
                f"({len(train_candidates)}). Using all available."
            )
            training_count = len(train_candidates)

        # Deterministic selection with your seed
        random.seed(self.args.seed)
        train_candidates_shuffled = train_candidates[:]
        random.shuffle(train_candidates_shuffled)
        training_networks = train_candidates_shuffled[:training_count]

        # --- 5) Iterate splits and build datasets ---
        # Training: split into train/val/test (as before) and later concat train & val
        # Validation: use full (subsetted by scenarios) datasets and concat together
        # Test: keep per-network datasets in self.test_datasets (same concept as now)

        splits = {
            "train": training_networks,
            "val": val_networks,
            "test": test_networks,
        }
        
        for split_name, networks in splits.items():
            for network in networks:
                data_normalizer = load_normalizer(args=self.args)
                if split_name == "test":
                    self.data_normalizers.append(data_normalizer)

                data_path_network = os.path.join(self.data_dir, network)

                # --- Preprocess on rank 0 only, then barrier for all ranks ---
                if is_main_rank():
                    print(f"Pre-processing of {network} dataset ({split_name}) on rank 0")
                    _ = HeteroGridDatasetDisk(  # just to trigger processing
                        root=data_path_network,
                        norm_method=self.args.data.normalization,
                        data_normalizer=data_normalizer,
                        transform=get_task_transforms(args=self.args),
                    )

                if dist.is_available() and dist.is_initialized():
                    dist.barrier()

                # Materialize dataset (post-preprocessing)
                full_dataset = HeteroGridDatasetDisk(
                    root=data_path_network,
                    norm_method=self.args.data.normalization,
                    data_normalizer=data_normalizer,
                    transform=get_task_transforms(args=self.args),
                )

                if split_name == "train":
                    self.train_datasets.append(full_dataset)
                elif split_name == "val":
                    self.val_datasets.append(full_dataset)
                else:  # split_name == "test"
                    self.test_datasets.append(full_dataset)

        # --- 6) Build multi-dataset concatenations ---
        self.train_dataset_multi = ConcatDataset(self.train_datasets) if len(self.train_datasets) > 0 else None
        self.val_dataset_multi = ConcatDataset(self.val_datasets) if len(self.val_datasets) > 0 else None

        self._is_setup_done = True


    # def setup(self, stage: str):
    #     if self._is_setup_done:
    #         print(f"Setup already done for stage={stage}, skipping...")
    #         return

    #     # --------------------------------------------------
    #     # 1. TRAIN DATASETS (ID)
    #     # --------------------------------------------------
    #     for i, network in enumerate(self.args.data.networks):
    #         data_normalizer = load_normalizer(args=self.args)
    #         self.data_normalizers.append(data_normalizer)

    #         data_path_network = os.path.join(self.data_dir, network)

    #         # Preprocess on rank 0
    #         if dist.is_available() and dist.is_initialized() and dist.get_rank() == 0:
    #             _ = HeteroGridDatasetDisk(
    #                 root=data_path_network,
    #                 norm_method=self.args.data.normalization,
    #                 data_normalizer=data_normalizer,
    #                 transform=get_task_transforms(args=self.args),
    #             )

    #         if dist.is_available() and dist.is_initialized():
    #             dist.barrier()

    #         dataset = HeteroGridDatasetDisk(
    #             root=data_path_network,
    #             norm_method=self.args.data.normalization,
    #             data_normalizer=data_normalizer,
    #             transform=get_task_transforms(args=self.args),
    #         )

    #         # Limit scenarios
    #         num_scenarios = self.args.data.scenarios[i]
    #         num_scenarios = min(num_scenarios, len(dataset))

    #         indices = list(range(len(dataset)))
    #         random.seed(self.args.seed)
    #         random.shuffle(indices)
    #         dataset = Subset(dataset, indices[:num_scenarios])

    #         self.train_datasets.append(dataset)

    #     self.train_dataset_multi = ConcatDataset(self.train_datasets)

    #     # --------------------------------------------------
    #     # 2. OOD VALIDATION / TEST DATASETS
    #     # --------------------------------------------------
    #     self.val_datasets = []
    #     self.test_datasets = []

    #     for i, network in enumerate(self.args.data.test_networks):
    #         data_normalizer = load_normalizer(args=self.args)
    #         test_grids_path = "/dccstor/gridfm/powermodels_data/v1/pf"
    #         data_path_network = os.path.join(test_grids_path, network)

    #         if dist.is_available() and dist.is_initialized() and dist.get_rank() == 0:
    #             _ = HeteroGridDatasetDisk(
    #                 root=data_path_network,
    #                 norm_method=self.args.data.normalization,
    #                 data_normalizer=data_normalizer,
    #                 transform=get_task_transforms(args=self.args),
    #             )

    #         if dist.is_available() and dist.is_initialized():
    #             dist.barrier()

    #         dataset = HeteroGridDatasetDisk(
    #             root=data_path_network,
    #             norm_method=self.args.data.normalization,
    #             data_normalizer=data_normalizer,
    #             transform=get_task_transforms(args=self.args),
    #         )

    #         num_scenarios = self.args.data.test_scenarios[i]
    #         num_scenarios = min(num_scenarios, len(dataset))

    #         indices = list(range(len(dataset)))
    #         random.seed(self.args.seed)
    #         random.shuffle(indices)
    #         dataset = Subset(dataset, indices[:num_scenarios])

    #         # 🔹 Validation and test can be identical
    #         self.val_datasets.append(dataset)
    #         self.test_datasets.append(dataset)

    #     # Validation concatenated
    #     self.val_dataset_multi = ConcatDataset(self.val_datasets)

    #     # Test kept separate (OOD analysis per network)
    #     # self.test_datasets is already a list of datasets

    #     self._is_setup_done = True

    # def setup(self, stage: str):
    #     if self._is_setup_done:
    #         print(f"Setup already done for stage={stage}, skipping...")
    #         return

    #     for i, network in enumerate(self.args.data.test_networks):
    #         data_normalizer = load_normalizer(args=self.args)
    #         self.data_normalizers.append(data_normalizer)

    #         # Create torch dataset and split
    #         data_path_network = os.path.join(self.data_dir, network)

    #         # Run preprocessing only on rank 0
    #         if dist.is_available() and dist.is_initialized() and dist.get_rank() == 0:
    #             print(f"Pre-processing of {network} dataset on rank 0")
    #             _ = HeteroGridDatasetDisk(  # just to trigger processing
    #                 root=data_path_network,
    #                 norm_method=self.args.data.normalization,
    #                 data_normalizer=data_normalizer,
    #                 transform=get_task_transforms(args=self.args),
    #             )

    #         # All ranks wait here until processing is done
    #         if torch.distributed.is_available() and torch.distributed.is_initialized():
    #             torch.distributed.barrier()

    #         dataset = HeteroGridDatasetDisk(
    #             root=data_path_network,
    #             norm_method=self.args.data.normalization,
    #             data_normalizer=data_normalizer,
    #             transform=get_task_transforms(args=self.args),
    #         )
    #         self.datasets.append(dataset)

    #         num_scenarios = self.args.data.scenarios[i]
    #         if num_scenarios > len(dataset):
    #             warnings.warn(
    #                 f"Requested number of scenarios ({num_scenarios}) exceeds dataset size ({len(dataset)}). "
    #                 "Using the full dataset instead.",
    #             )
    #             num_scenarios = len(dataset)

    #         # Create a subset
    #         all_indices = list(range(len(dataset)))
    #         # Random seed set before every shuffle for reproducibility in case the power grid datasets are analyzed in a different order
    #         random.seed(self.args.seed)
    #         random.shuffle(all_indices)
    #         subset_indices = all_indices[:num_scenarios]
    #         dataset = Subset(dataset, subset_indices)

    #         # Random seed set before every split, same as above
    #         np.random.seed(self.args.seed)
    #         train_dataset, val_dataset, test_dataset = split_dataset(
    #             dataset,
    #             self.data_dir,
    #             self.args.data.val_ratio,
    #             self.args.data.test_ratio,
    #         )

    #         self.train_datasets.append(train_dataset)
    #         self.val_datasets.append(val_dataset)
    #         self.test_datasets.append(test_dataset)

    #     self.train_dataset_multi = ConcatDataset(self.train_datasets)
    #     self.val_dataset_multi = ConcatDataset(self.val_datasets)
    #     self._is_setup_done = True

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
