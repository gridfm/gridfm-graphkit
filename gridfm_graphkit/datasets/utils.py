import numpy as np
from torch.utils.data import Subset
from typing import Tuple
from torch import Tensor
import torch
from pathlib import Path


def split_dataset(
    dataset,
    log_dir: str,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Tuple[Subset, Subset, Subset]:
    """
    Splits a dataset into training, validation, and test sets, and logs the indices for each split to CSV files for further analysis

    Args:
        dataset (torch_geometric.dataDataset): The dataset to split.
        log_dir (str): Directory where CSV files containing the indices for each split will be saved.
        val_ratio (float, optional): Proportion of the dataset to include in the validation set.
        test_ratio (float, optional): Proportion of the dataset to include in the test set.

    Raises:
        ValueError: If `val_ratio + test_ratio >= 1`, which would leave no data for the training set.

    Returns:
        tuple: A tuple containing:
            - train_dataset (torch.utils.data.Subset): The training subset of the dataset.
            - val_dataset (torch.utils.data.Subset): The validation subset of the dataset.
            - test_dataset (torch.utils.data.Subset): The test subset of the dataset.
    """

    if val_ratio + test_ratio >= 1:
        raise ValueError("The sum of val_ratio and test_ratio must be less than 1.")

    val_size = int(val_ratio * len(dataset))
    test_size = int(test_ratio * len(dataset))
    train_size = len(dataset) - val_size - test_size

    # Generate shuffled indices and split manually
    indices = np.random.permutation(len(dataset))
    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    # Create subsets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, val_dataset, test_dataset


def split_dataset_by_load_scenario_idx(
    dataset,
    log_dir: str,
    load_scenarios: Tensor,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Tuple[Subset, Subset, Subset]:
    """Split dataset by unique load-scenario IDs to avoid scenario leakage."""
    if val_ratio + test_ratio >= 1:
        raise ValueError("The sum of val_ratio and test_ratio must be less than 1.")

    unique_load_scenarios = torch.unique(load_scenarios)
    val_size = int(val_ratio * len(unique_load_scenarios))
    test_size = int(test_ratio * len(unique_load_scenarios))
    train_size = len(unique_load_scenarios) - val_size - test_size

    unique_load_scenarios = torch.tensor(np.random.permutation(unique_load_scenarios))
    train_load_scenarios = unique_load_scenarios[:train_size]
    val_load_scenarios = unique_load_scenarios[train_size : train_size + val_size]
    test_load_scenarios = unique_load_scenarios[train_size + val_size :]

    train_indices = (
        torch.nonzero(torch.isin(load_scenarios, train_load_scenarios))
        .flatten()
        .tolist()
    )
    val_indices = (
        torch.nonzero(torch.isin(load_scenarios, val_load_scenarios)).flatten().tolist()
    )
    test_indices = (
        torch.nonzero(torch.isin(load_scenarios, test_load_scenarios))
        .flatten()
        .tolist()
    )

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, val_dataset, test_dataset


def split_from_existing_files(
    dataset,
    splits_folder: Path,
) -> Tuple[Subset, Subset, Subset]:
    """Build train/val/test subsets from split index files.

    Expects `train.pt`, `val.pt`, and `test.pt` inside `splits_folder`.
    Returns both the dataset subsets and the raw scenario ids per split.
    """
    output = []

    indices = {}

    for split in ["train", "val", "test"]:
        split_file = splits_folder / f"{split}.pt"
        assert split_file.is_file(), f"{str(split_file)} does not exist"
        split_indices = torch.load(str(split_file), weights_only=True)
        split_dataset = Subset(dataset, split_indices)
        output.append(split_dataset)
        split_indices = list(split_indices)
        print(f'{split=} {len(split_indices)=}')
        indices[split]=[int(t.item()) for t in split_indices]

    output = tuple(output)
    return output, indices