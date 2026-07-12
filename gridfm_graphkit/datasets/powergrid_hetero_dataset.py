from gridfm_graphkit.datasets.normalizers import Normalizer

import os.path as osp
import os
import torch
from torch_geometric.data import Dataset
from typing import Optional, Callable
from torch_geometric.data import HeteroData
from gridfm_graphkit.datasets.hetero_preprocess import (
    build_load_scenarios,
    merge_agg_gen_into_bus,
    process_scenarios,
    read_raw_tables,
    assert_scenario_index,
)


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
        bus_data, gen_data, branch_data = read_raw_tables(self.raw_dir)
        assert_scenario_index(bus_data)

        load_scenarios = build_load_scenarios(bus_data)
        if load_scenarios is not None:
            torch.save(load_scenarios, osp.join(self.processed_dir, "load_scenarios.pt"))

        bus_data = merge_agg_gen_into_bus(bus_data, gen_data)

        done_path = osp.join(self.processed_dir, self.processed_done_file)
        if osp.exists(done_path):
            print("Processed files already exist. Skipping processing.")
            return

        process_scenarios(
            bus_data,
            gen_data,
            branch_data,
            self.processed_dir,
            skip_existing=True,
            show_progress=True,
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
