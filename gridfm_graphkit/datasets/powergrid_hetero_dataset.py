from gridfm_graphkit.datasets.normalizers import Normalizer

import os.path as osp
import os
import torch
from torch_geometric.data import Dataset
import pandas as pd
from tqdm import tqdm
from typing import Optional, Callable
from torch_geometric.data import HeteroData
from gridfm_graphkit.datasets.globals import VA_H, PG_H


class HeteroGridDatasetDisk(Dataset):
    """
    A PyTorch Geometric `Dataset` for power grid data stored on disk.
    This dataset reads node and edge CSV files and saves each graph
    separately on disk as a processed file. Data is loaded from disk
    lazily on demand. Normalization is applied at access time via
    the data_normalizer (which must be fitted externally before iteration).

    Args:
        root (str): Root directory where the dataset is stored.
        data_normalizer (Normalizer): Normalizer used for features (fitted externally by the datamodule).
        transform (callable, optional): Transformation applied at runtime.
        pre_transform (callable, optional): Transformation applied before saving to disk.
        pre_filter (callable, optional): Filter to determine which graphs to keep.
    """

    def __init__(
        self,
        root: str,
        data_normalizer: Normalizer,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        self.data_normalizer = data_normalizer
        self.length = None

        super().__init__(root, transform, pre_transform, pre_filter)

        load_scenarios_path = osp.join(self.processed_dir, "load_scenarios.pt")
        if osp.exists(load_scenarios_path):
            self.load_scenarios = torch.load(load_scenarios_path, weights_only=True)

    @property
    def raw_file_names(self):
        return ["bus_data.parquet", "gen_data.parquet", "branch_data.parquet"]

    @property
    def processed_done_file(self):
        return "processed_raw_files.done"

    @property
    def processed_file_names(self):
        return [
            self.processed_done_file,
        ]

    def download(self):
        pass

    def process(self):
        print("LOADING DATA")
        bus_data = pd.read_parquet(osp.join(self.raw_dir, "bus_data.parquet"))
        gen_data = pd.read_parquet(osp.join(self.raw_dir, "gen_data.parquet"))
        branch_data = pd.read_parquet(osp.join(self.raw_dir, "branch_data.parquet"))

        assert (
            bus_data["scenario"].min() == 0
            and bus_data["scenario"].max() == len(bus_data["scenario"].unique()) - 1
        )
        if "load_scenario_idx" in bus_data.columns:
            load_scenarios = torch.tensor(
                bus_data.groupby("scenario", sort=True)["load_scenario_idx"].first().values,
            )
            torch.save(load_scenarios, osp.join(self.processed_dir, "load_scenarios.pt"))

        agg_gen = (
            gen_data.groupby(["scenario", "bus"])[["min_q_mvar", "max_q_mvar"]]
            .sum()
            .reset_index()
        )
        bus_data = bus_data.merge(agg_gen, on=["scenario", "bus"], how="left").fillna(0)

        done_path = osp.join(self.processed_dir, self.processed_done_file)
        if osp.exists(done_path):
            print("Processed files already exist. Skipping processing.")
            return

        bus_features = [
            "Pd",
            "Qd",
            "Qg",
            "Vm",
            "Va",
            "PQ",
            "PV",
            "REF",
            "min_vm_pu",
            "max_vm_pu",
            "min_q_mvar",
            "max_q_mvar",
            "GS",
            "BS",
            "vn_kv",
        ]

        gen_features = [
            "p_mw",
            "min_p_mw",
            "max_p_mw",
            "cp0_eur",
            "cp1_eur_per_mw",
            "cp2_eur_per_mw2",
            "in_service",
        ]

        common_branch_features = ["tap", "ang_min", "ang_max", "rate_a", "br_status"]
        forward_branch_features = [
            "pf",
            "qf",
            "Yff_r",
            "Yff_i",
            "Yft_r",
            "Yft_i",
        ] + common_branch_features
        reverse_branch_features = [
            "pt",
            "qt",
            "Ytt_r",
            "Ytt_i",
            "Ytf_r",
            "Ytf_i",
        ] + common_branch_features

        # Group by scenario
        bus_groups = bus_data.groupby("scenario") # Groupby preserves the order of rows within each group. 
        # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html
        gen_groups = gen_data.groupby("scenario")
        branch_groups = branch_data.groupby("scenario")

        # Process each scenario
        for scenario in tqdm(
            bus_data["scenario"].unique(),
            desc="Processing scenarios",
        ):
            if osp.exists(osp.join(self.processed_dir, f"data_index_{scenario}.pt")):
                continue
            if (
                scenario not in gen_groups.groups
                or scenario not in branch_groups.groups
            ):
                raise ValueError

            data = HeteroData()

            # Bus nodes
            bus_df = bus_groups.get_group(scenario)
            # assert that the buses are in increasing order
            assert (bus_df["bus"].values == torch.arange(len(bus_df))).all(), "Buses are not in increasing order"
            #todo: we should remove this assert and store the bus idx in the tensors
            # right now we need the increasing order for e.g. the predict step that uses torch.arange(n_nodes) to index the buses.
            data["bus"].x = torch.tensor(bus_df[bus_features].values, dtype=torch.float)

            # Generator nodes
            gen_df = gen_groups.get_group(scenario).reset_index()
            data["gen"].x = torch.tensor(gen_df[gen_features].values, dtype=torch.float)
            gen_df["gen_index"] = gen_df.index  # Use actual index as generator ID
            # todo: change this to instead use the generator id as the index

            data["bus"].y = data["bus"].x[:, : (VA_H + 1)].clone()
            data["gen"].y = data["gen"].x[:, : (PG_H + 1)].clone()

            # Bus-Bus edges
            branch_df = branch_groups.get_group(scenario)

            forward_edges = torch.tensor(
                branch_df[["from_bus", "to_bus"]].values.T,
                dtype=torch.long,
            )
            forward_edge_attr = torch.tensor(
                branch_df[forward_branch_features].values,
                dtype=torch.float,
            )

            reverse_edges = torch.tensor(
                branch_df[["to_bus", "from_bus"]].values.T,
                dtype=torch.long,
            )
            reverse_edge_attr = torch.tensor(
                branch_df[reverse_branch_features].values,
                dtype=torch.float,
            )

            edge_index = torch.cat([forward_edges, reverse_edges], dim=1)
            edge_attr = torch.cat([forward_edge_attr, reverse_edge_attr], dim=0)

            forward_targets = torch.tensor(
                branch_df[["pf", "qf"]].values,
                dtype=torch.float,
            )
            reverse_targets = torch.tensor(
                branch_df[["pt", "qt"]].values,
                dtype=torch.float,
            )
            edge_y = torch.cat([forward_targets, reverse_targets], dim=0)

            data["bus", "connects", "bus"].edge_index = edge_index
            data["bus", "connects", "bus"].edge_attr = edge_attr
            data["bus", "connects", "bus"].y = edge_y

            # Gen-Bus and Bus-Gen edges
            data["gen", "connected_to", "bus"].edge_index = torch.tensor(
                gen_df[["gen_index", "bus"]].values.T,
                dtype=torch.long,
            )
            data["bus", "connected_to", "gen"].edge_index = torch.tensor(
                gen_df[["bus", "gen_index"]].values.T,
                dtype=torch.long,
            )

            data["scenario_id"] = torch.tensor([scenario], dtype=torch.long)

            # Save graph
            torch.save(
                data.to_dict(),
                osp.join(self.processed_dir, f"data_index_{scenario}.pt"),
            )

        with open(osp.join(self.processed_dir, self.processed_done_file), "w") as f:
            f.write("done")

    def len(self):
        if self.length is None:
            files = [
                f
                for f in os.listdir(self.processed_dir)
                if f.startswith(
                    "data_index_",
                )
                and f.endswith(".pt")
            ]
            self.length = len(files)
        return self.length

    def get(self, idx):
        file_name = osp.join(
            self.processed_dir,
            f"data_index_{idx}.pt",
        )
        if not osp.exists(file_name):
            raise IndexError(f"Data file {file_name} does not exist.")
        data_dict = torch.load(file_name, weights_only=True)
        data = HeteroData.from_dict(data_dict)
        self.data_normalizer.transform(data=data)
        return data
