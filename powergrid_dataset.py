from gridfm_graphkit.datasets.normalizers import Normalizer, BaseMVANormalizer
from gridfm_graphkit.datasets.transforms import (
    AddEdgeWeights,
    AddNormalizedRandomWalkPE,
)

import os.path as osp
import os
import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset
import pandas as pd
from tqdm import tqdm
from typing import Optional, Callable
import glob
import re
import numpy as np


class GridDatasetDisk(Dataset):
    """
    A PyTorch Geometric `Dataset` for power grid data stored on disk.
    This dataset reads node and edge CSV files, applies normalization,
    and saves each graph separately on disk as a processed file.
    Data is loaded from disk lazily on demand.

    Args:
        root (str): Root directory where the dataset is stored.
        norm_method (str): Identifier for normalization method (e.g., "minmax", "standard").
        node_normalizer (Normalizer): Normalizer used for node features.
        edge_normalizer (Normalizer): Normalizer used for edge features.
        pe_dim (int): Length of the random walk used for positional encoding.
        mask_dim (int, optional): Number of features per-node that could be masked.
        transform (callable, optional): Transformation applied at runtime.
        pre_transform (callable, optional): Transformation applied before saving to disk.
        pre_filter (callable, optional): Filter to determine which graphs to keep.
    """

    def __init__(
        self,
        root: str,
        norm_method: str,
        node_normalizer: Normalizer,
        edge_normalizer: Normalizer,
        pe_dim: int,
        mask_dim: int = 6,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        self.norm_method = norm_method
        self.node_normalizer = node_normalizer
        self.edge_normalizer = edge_normalizer
        self.pe_dim = pe_dim
        self.mask_dim = mask_dim
        self.length = None
        self.files = None

        super().__init__(root, transform, pre_transform, pre_filter)

        # Load normalization stats if available
        node_stats_path = osp.join(
            self.processed_dir,
            f"node_stats_{self.norm_method}.pt",
        )
        edge_stats_path = osp.join(
            self.processed_dir,
            f"edge_stats_{self.norm_method}.pt",
        )
        if osp.exists(node_stats_path) and osp.exists(edge_stats_path):
            self.node_stats = torch.load(node_stats_path, weights_only=False)
            self.edge_stats = torch.load(edge_stats_path, weights_only=False)
            self.node_normalizer.fit_from_dict(self.node_stats)
            self.edge_normalizer.fit_from_dict(self.edge_stats)

    def scan_batch_files(self) -> tuple[list[str], list[str]]:
        """
        Scan directory for batch CSV files

        Returns:
            tuple: (bus_files, branch_files) sorted lists of file paths
        """
        # Pattern to match batch files
        raw_dir = "./data/scenario_33meshed/raw"
        bus_pattern = osp.join(osp.abspath(raw_dir), "bus_batch_*.csv")
        branch_pattern = osp.join(osp.abspath(raw_dir), "branch_batch_*.csv")

        # Find all matching files
        bus_files = glob.glob(bus_pattern)
        branch_files = glob.glob(branch_pattern)

        # Sort files numerically by batch number
        def sort_files_numerically(file_list):
            def extract_batch_number(filename):
                match = re.search(r'batch_(\d+)', filename)
                return int(match.group(1)) if match else 0

            return sorted(file_list, key=extract_batch_number)

        bus_files_sorted = sort_files_numerically(bus_files)
        branch_files_sorted = sort_files_numerically(branch_files)

        print(f"Found {len(bus_files_sorted)} bus batch files")
        print(f"Found {len(branch_files_sorted)} branch batch files")

        return bus_files_sorted, branch_files_sorted

    def load_all_batch_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load all bus and branch batch files into DataFrames
        Replace empty cells with zeros
        """
        bus_files, branch_files = self.scan_batch_files()

        if not bus_files or not branch_files:
            # List what files were actually found
            all_files = os.listdir(self.raw_dir)
            csv_files = [f for f in all_files if f.endswith('.csv')]

            error_msg = (
                f"No batch files found in {self.raw_dir}.\n"
                f"Expected files like: bus_batch_0001.csv, branch_batch_0001.csv\n"
                f"Files found in directory: {all_files}\n"
                f"CSV files found: {csv_files}"
            )
            raise FileNotFoundError(error_msg)

        # Load bus batches
        bus_dfs = []
        print("Loading bus batch files...")
        for i, bus_file in enumerate(tqdm(bus_files, desc="Bus batches")):
            try:
                # Read CSV and replace empty cells with 0
                df = pd.read_csv(bus_file)

                # Replace empty strings, NaN, and None with 0 for all numeric columns
                df = self._fill_empty_cells_with_zero(df)

                bus_dfs.append(df)

            except Exception as e:
                print(f"Error loading {bus_file}: {e}")
                continue

        if not bus_dfs:
            raise ValueError("No bus data loaded from batch files")

        combined_bus_df = pd.concat(bus_dfs, ignore_index=True)
        print(f"✓ Combined {len(bus_dfs)} bus batch files: {len(combined_bus_df):,} rows")

        # Load branch batches
        branch_dfs = []
        for i, branch_file in enumerate(tqdm(branch_files, desc="Branch batches")):
            try:
                # Read CSV and replace empty cells with 0
                df = pd.read_csv(branch_file)

                # Replace empty strings, NaN, and None with 0 for all numeric columns
                df = self._fill_empty_cells_with_zero(df)

                branch_dfs.append(df)

            except Exception as e:
                print(f"Error loading {branch_file}: {e}")
                continue

        if not branch_dfs:
            raise ValueError("No branch data loaded from batch files")

        combined_branch_df = pd.concat(branch_dfs, ignore_index=True)
        print(f"✓ Combined {len(branch_dfs)} branch batch files: {len(combined_branch_df):,} rows")

        return combined_bus_df, combined_branch_df

    def _fill_empty_cells_with_zero(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Replace empty cells with zeros in a DataFrame

        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame

        Returns:
        --------
        pd.DataFrame
            DataFrame with empty cells replaced by 0
        """
        # Make a copy to avoid modifying the original
        df_filled = df.copy()

        # Identify numeric columns (including integer and float)
        numeric_columns = df_filled.select_dtypes(include=[np.number]).columns

        # Identify non-numeric columns that should be handled differently
        non_numeric_columns = df_filled.select_dtypes(exclude=[np.number]).columns

        print(f"  Processing {len(df_filled)} rows, {len(numeric_columns)} numeric columns")

        # For numeric columns: replace NaN with 0
        if len(numeric_columns) > 0:
            # Count NaN values before replacement
            nan_count_before = df_filled[numeric_columns].isna().sum().sum()
            if nan_count_before > 0:
            # Replace NaN with 0 for numeric columns
                df_filled[numeric_columns] = df_filled[numeric_columns].fillna(0)

        # For non-numeric columns that might contain numeric data as strings
        for col in non_numeric_columns:
            # Try to convert to numeric, replacing non-convertible values with NaN first
            # then fill with 0
            converted = pd.to_numeric(df_filled[col], errors='coerce')
            if not converted.isna().all():  # If at least some values could be converted to numeric
                df_filled[col] = converted.fillna(0)

        # Also handle empty strings in numeric columns that might have been read as objects
        for col in df_filled.columns:
            if df_filled[col].dtype == 'object':
                # Replace empty strings with 0
                empty_string_mask = df_filled[col] == ''
                empty_count = empty_string_mask.sum()
                if empty_count > 0:
                    df_filled.loc[empty_string_mask, col] = 0

                # Also try to convert the entire column to numeric if possible
                try:
                    converted = pd.to_numeric(df_filled[col])
                    df_filled[col] = converted
                except (ValueError, TypeError):
                    # Column contains non-numeric values, leave as is
                    pass

        return df_filled

    @property
    def raw_file_names(self):
        #return ["pf_node.csv", "pf_edge.csv"]
        return []

    @property
    def processed_done_file(self):
        return f"processed_{self.norm_method}_{self.mask_dim}_{self.pe_dim}.done"

    @property
    def processed_file_names(self):
        return [self.processed_done_file]

    def download(self):
        pass

    def process(self):
        """
        node_df = pd.read_csv(osp.join(self.raw_dir, "pf_node.csv"))
        edge_df = pd.read_csv(osp.join(self.raw_dir, "pf_edge.csv"))
        """
        node_df, edge_df = self.load_all_batch_data() #load all batch data

        # Check the unique scenarios available
        scenarios = node_df["scenario"].unique()

        # Ensure node and edge data match
        """
        if not (scenarios == edge_df["scenario"].unique()).all():
            raise ValueError("Mismatch between node and edge scenario values.")
        """
        edge_scenarios = edge_df["scenario"].unique()
        if not set(scenarios) == set(edge_scenarios):
            print(f"Warning: Mismatch between node and edge scenarios.")
            print(f"Node scenarios: {len(scenarios)}, Edge scenarios: {len(edge_scenarios)}")
            # Use intersection of scenarios
            common_scenarios = set(scenarios) & set(edge_scenarios)
            node_df = node_df[node_df["scenario"].isin(common_scenarios)]
            edge_df = edge_df[edge_df["scenario"].isin(common_scenarios)]
            scenarios = node_df["scenario"].unique()
            print(f"Using {len(common_scenarios)} common scenarios")

        print(f"Processing {len(scenarios)} scenarios...")

        # normalize node attributes
        cols_to_normalize = ["Pd", "Qd", "Pg", "Qg", "Vm", "Va"]
        to_normalize = torch.tensor(
            node_df[cols_to_normalize].values,
            dtype=torch.float,
        )
        self.node_stats = self.node_normalizer.fit(to_normalize)
        node_df[cols_to_normalize] = self.node_normalizer.transform(
            to_normalize,
        ).numpy()

        # normalize edge attributes
        cols_to_normalize = ["G", "B"]
        to_normalize = torch.tensor(
            edge_df[cols_to_normalize].values,
            dtype=torch.float,
        )
        if isinstance(self.node_normalizer, BaseMVANormalizer):
            self.edge_stats = self.edge_normalizer.fit(
                to_normalize,
                self.node_normalizer.baseMVA,
            )
        else:
            self.edge_stats = self.edge_normalizer.fit(to_normalize)
        edge_df[cols_to_normalize] = self.edge_normalizer.transform(
            to_normalize,
        ).numpy()

        # save stats
        node_stats_path = osp.join(
            self.processed_dir,
            f"node_stats_{self.norm_method}.pt",
        )
        edge_stats_path = osp.join(
            self.processed_dir,
            f"edge_stats_{self.norm_method}.pt",
        )
        torch.save(self.node_stats, node_stats_path)
        torch.save(self.edge_stats, edge_stats_path)

        # Create groupby objects for scenarios
        node_groups = node_df.groupby("scenario")
        edge_groups = edge_df.groupby("scenario")

        for scenario_idx in tqdm(scenarios):
            # NODE DATA
            node_data = node_groups.get_group(scenario_idx)
            x = torch.tensor(
                node_data[
                    ["Pd", "Qd", "Pg", "Qg", "Vm", "Va", "PQ", "PV", "REF"]
                ].values,
                dtype=torch.float,
            )
            y = x[:, : self.mask_dim]

            # EDGE DATA
            edge_data = edge_groups.get_group(scenario_idx)
            edge_attr = torch.tensor(edge_data[["G", "B"]].values, dtype=torch.float)
            edge_index = torch.tensor(
                edge_data[["index1", "index2"]].values.T,
                dtype=torch.long,
            )

            # Create the Data object
            graph_data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y,
                scenario_id=scenario_idx,
            )
            pe_pre_transform = AddEdgeWeights()
            graph_data = pe_pre_transform(graph_data)
            pe_transform = AddNormalizedRandomWalkPE(
                walk_length=self.pe_dim,
                attr_name="pe",
            )
            graph_data = pe_transform(graph_data)
            torch.save(
                graph_data,
                osp.join(
                    self.processed_dir,
                    f"data_{self.norm_method}_{self.mask_dim}_{self.pe_dim}_index_{scenario_idx}.pt",
                ),
            )
        with open(osp.join(self.processed_dir, self.processed_done_file), "w") as f:
            f.write("done")

    def len(self):
        if self.files is None:
            self.files = sorted([
                f for f in os.listdir(self.processed_dir)
                if f.startswith(f"data_{self.norm_method}_{self.mask_dim}_{self.pe_dim}_index_")
                   and f.endswith(".pt")
            ])
        return len(self.files)

    def get(self, idx):
        if self.files is None:
            self.len()  # populate self.files

        if idx >= len(self.files):
            raise IndexError(f"Requested index {idx}, but dataset has only {len(self.files)} files.")

        file_path = osp.join(self.processed_dir, self.files[idx])
        data = torch.load(file_path, weights_only=False)
        if self.transform:
            data = self.transform(data)
        return data

    def change_transform(self, new_transform):
        """
        Temporarily switch to a new transform function, used when evaluating different tasks.

        Args:
            new_transform (Callable): The new transform to use.
        """
        self.original_transform = self.transform
        self.transform = new_transform

    def reset_transform(self):
        """
        Reverts the transform to the original one set during initialization, usually called after the evaluation step.
        """
        if self.original_transform is None:
            raise ValueError(
                "The original transform is None or the function change_transform needs to be called before",
            )
        self.transform = self.original_transform
