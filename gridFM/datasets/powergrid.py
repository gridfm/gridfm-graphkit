from gridFM.datasets.data_normalization import *
import os.path as osp
import torch
from torch_geometric.data import Dataset, Data, InMemoryDataset
import pandas as pd
from tqdm import tqdm
from gridFM.datasets.data_normalization import Normalizer
from typing import Optional, Callable


class GridDatasetMem(InMemoryDataset):
    def __init__(
        self,
        root: str,
        scenarios: int,
        norm_method: str,
        node_normalizer: Normalizer,
        edge_normalizer: Normalizer,
        mask_ratio: float = 0.5,
        mask_dim: int = 6,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        self.max_scenarios = scenarios
        self.norm_method = norm_method
        self.node_normalizer = node_normalizer
        self.edge_normalizer = edge_normalizer
        self.mask_ratio = mask_ratio
        self.mask_dim = mask_dim

        super().__init__(root, transform, pre_transform, pre_filter)

        node_stats_path = osp.join(
            self.processed_dir, f"node_stats_{self.norm_method}.pt"
        )
        edge_stats_path = osp.join(
            self.processed_dir, f"edge_stats_{self.norm_method}.pt"
        )
        if osp.exists(node_stats_path) and osp.exists(edge_stats_path):
            self.node_stats = torch.load(node_stats_path, weights_only=False)
            self.edge_stats = torch.load(edge_stats_path, weights_only=False)
            self.node_normalizer.fit_from_dict(self.node_stats)
            self.edge_normalizer.fit_from_dict(self.edge_stats)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["pf_node.csv", "pf_edge.csv"]  # No raw files needed for random graphs

    @property
    def processed_file_names(self):
        return [f"data_full_{self.norm_method}_{self.mask_ratio}.pt"]

    def download(self):
        pass

    def process(self):
        node_df = pd.read_csv(osp.join(self.raw_dir, "pf_node.csv"))
        edge_df = pd.read_csv(osp.join(self.raw_dir, "pf_edge.csv"))

        # Check the unique scenarios available
        scenarios = node_df["scenario"].unique()

        if self.max_scenarios > len(scenarios):
            raise ValueError(
                f"max_scenarios ({self.max_scenarios}) is greater than the number of "
                f"available scenarios ({len(scenarios)})"
            )

        edge_index = torch.tensor(
            edge_df[["index1", "index2"]].values.T, dtype=torch.long
        )

        edge_attr = edge_df[["G", "B"]].values
        node_attr = node_df[["Pd", "Qd", "Pg", "Qg", "Vm", "Va"]].values

        self.node_stats = self.node_normalizer.fit(node_attr)
        self.edge_stats = self.edge_normalizer.fit(edge_attr)

        # Save calculated statistics for future use
        node_stats_path = osp.join(
            self.processed_dir, f"node_stats_{self.norm_method}.pt"
        )
        edge_stats_path = osp.join(
            self.processed_dir, f"edge_stats_{self.norm_method}.pt"
        )

        torch.save(self.node_stats, node_stats_path)
        torch.save(self.edge_stats, edge_stats_path)

        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        edge_attr = self.edge_normalizer.transform(edge_attr)

        data_list = []
        # Iterate over each scenario to process and save each graph
        for idx in tqdm(range(self.max_scenarios)):

            # Filter node and edge data for the current scenario
            node_data = node_df[node_df["scenario"] == scenarios[idx]]

            # Create node features tensor (N, node_features)
            x = torch.tensor(
                node_data[
                    ["Pd", "Qd", "Pg", "Qg", "Vm", "Va", "PQ", "PV", "REF"]
                ].values,
                dtype=torch.float,
            )
            y = torch.tensor(
                node_data[["Pd", "Qd", "Pg", "Qg", "Vm", "Va"]].values,
                dtype=torch.float,
            )

            # Do not normalize the encoded type
            x[:, : self.mask_dim] = self.node_normalizer.transform(
                x[:, : self.mask_dim]
            )
            y = self.node_normalizer.transform(y)

            mask = torch.rand(x.size(0), self.mask_dim) < self.mask_ratio

            # Create the Data object
            graph_data = Data(
                x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, mask=mask
            )
            data_list.append(graph_data)
        self.save(data_list, self.processed_paths[0])
