from gridFM.datasets.data_normalization import Normalizer, BaseMVANormalizer
from gridFM.datasets.transforms import AddEdgeWeights, AddNormalizedRandomWalkPE

import os.path as osp
import torch
from torch_geometric.data import Data, InMemoryDataset
import pandas as pd
from tqdm import tqdm
from typing import Optional, Callable
from torch import Tensor


class GridDatasetMem(InMemoryDataset):
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
        self.original_transform = None

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
        return [f"data_full_{self.norm_method}.pt"]

    def download(self):
        pass

    def process(self):
        node_df = pd.read_csv(osp.join(self.raw_dir, "pf_node.csv"))
        edge_df = pd.read_csv(osp.join(self.raw_dir, "pf_edge.csv"))

        # Check the unique scenarios available
        scenarios = node_df["scenario"].unique()
        # Ensure node and edge data match
        assert (scenarios == edge_df["scenario"].unique()).all()

        ## normalize node attributes
        cols_to_normalize = ["Pd", "Qd", "Pg", "Qg", "Vm", "Va"]
        to_normalize = torch.tensor(
            node_df[cols_to_normalize].values, dtype=torch.float
        )
        self.node_stats = self.node_normalizer.fit(to_normalize)
        node_df[cols_to_normalize] = self.node_normalizer.transform(
            to_normalize
        ).numpy()

        ## normalize edge attributes
        cols_to_normalize = ["G", "B"]
        to_normalize = torch.tensor(
            edge_df[cols_to_normalize].values, dtype=torch.float
        )
        if isinstance(self.node_normalizer, BaseMVANormalizer):
            self.edge_stats = self.edge_normalizer.fit(
                to_normalize, self.node_normalizer.baseMVA
            )
        else:
            self.edge_stats = self.edge_normalizer.fit(to_normalize)
        edge_df[cols_to_normalize] = self.edge_normalizer.transform(
            to_normalize
        ).numpy()

        ## save stats
        node_stats_path = osp.join(
            self.processed_dir, f"node_stats_{self.norm_method}.pt"
        )
        edge_stats_path = osp.join(
            self.processed_dir, f"edge_stats_{self.norm_method}.pt"
        )
        torch.save(self.node_stats, node_stats_path)
        torch.save(self.edge_stats, edge_stats_path)

        # Create groupby objects for scenarios
        node_groups = node_df.groupby("scenario")
        edge_groups = edge_df.groupby("scenario")

        data_list = []
        for scenario_idx in tqdm(scenarios):
            ## NODE DATA
            node_data = node_groups.get_group(scenario_idx)
            x = torch.tensor(
                node_data[
                    ["Pd", "Qd", "Pg", "Qg", "Vm", "Va", "PQ", "PV", "REF"]
                ].values,
                dtype=torch.float,
            )
            y = x[:, : self.mask_dim]

            ## EDGE DATA
            edge_data = edge_groups.get_group(scenario_idx)
            edge_attr = torch.tensor(edge_data[["G", "B"]].values, dtype=torch.float)
            edge_index = torch.tensor(
                edge_data[["index1", "index2"]].values.T, dtype=torch.long
            )

            # Create the Data object
            graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            pe_pre_transform = AddEdgeWeights()
            graph_data = pe_pre_transform(graph_data)
            pe_transform = AddNormalizedRandomWalkPE(walk_length=self.pe_dim, attr_name="pe")
            graph_data = pe_transform(graph_data)
            data_list.append(graph_data)

        self.save(data_list, self.processed_paths[0])

    def change_transform(self, new_transform):
        self.original_transform = self.transform
        self.transform = new_transform

    def reset_transform(self):
        if self.original_transform is None:
            raise ValueError(
                "The original transform is None or the function change_transform needs to be called before"
            )
        self.transform = self.original_transform
