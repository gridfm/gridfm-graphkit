import numpy as np
from torch.utils.data import Subset
from typing import Tuple


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
