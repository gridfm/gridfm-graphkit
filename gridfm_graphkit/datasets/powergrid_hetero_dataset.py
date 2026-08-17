from gridfm_graphkit.datasets.normalizers import Normalizer

import os.path as osp
import os
import torch
from torch_geometric.data import Dataset
import pandas as pd
from tqdm import tqdm
from typing import Optional, Callable
from torch_geometric.data import HeteroData
from gridfm_graphkit.datasets.globals import (
    VA_H,
    PG_H,
    MIN_VM_H,
    MAX_VM_H,
    MIN_QG_H,
    MAX_QG_H,
    VN_KV,
    ANG_MIN,
    ANG_MAX,
    RATE_A,
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
        stream_partitions: str = "auto",
    ):
        _VALID = {"auto", "on", "off"}
        if stream_partitions not in _VALID:
            raise ValueError(
                f"stream_partitions={stream_partitions!r} is not valid; "
                f"choose one of {sorted(_VALID)}",
            )
        self.stream_partitions = stream_partitions
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
        partitions = self._detect_partitions()

        if self.stream_partitions == "on":
            if partitions is None:
                raise RuntimeError(
                    f"stream_partitions='on' requires Hive-partitioned parquet, "
                    f"but detected flat files in {self.raw_dir!r}. Use 'auto' or 'off'.",
                )
            return self._process_streaming(partitions)

        if self.stream_partitions == "auto" and partitions is not None:
            print(f"Detected {len(partitions)} Hive partitions — using streaming mode.")
            return self._process_streaming(partitions)

        # stream_partitions == "off", or "auto" with no partitions detected → legacy path
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
                bus_data.groupby("scenario", sort=True)["load_scenario_idx"]
                .first()
                .values,
            )
            torch.save(
                load_scenarios,
                osp.join(self.processed_dir, "load_scenarios.pt"),
            )

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

        # Group by scenario
        bus_groups = bus_data.groupby(
            "scenario",
        )  # Groupby preserves the order of rows within each group.
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
                raise ValueError(
                    f"Scenario {scenario} is missing generator or branch data.",
                )

            self._build_and_save_scenario(
                scenario,
                bus_groups.get_group(scenario),
                gen_groups.get_group(scenario),
                branch_groups.get_group(scenario),
            )

        with open(osp.join(self.processed_dir, self.processed_done_file), "w") as f:
            f.write("done")

    _PARTITION_PREFIX = "scenario_partition="

    def _partition_dir(self, table: str, partition_val: int) -> str:
        """Path to a single Hive partition directory for a raw table."""
        return osp.join(self.raw_dir, table, f"{self._PARTITION_PREFIX}{partition_val}")

    def _detect_partitions(self) -> list[int] | None:
        """Return sorted scenario_partition values when all three tables are Hive-partitioned, else None.

        Returns:
            Sorted list of partition integers when all three tables expose the same
            scenario_partition Hive layout, or None when at least one table is flat.

        Raises:
            ValueError: If all tables are partitioned but their partition sets differ,
                indicating a data-preparation inconsistency that would cause a silent
                read error later in streaming.
        """
        partition_sets: dict[str, set[int]] = {}
        for table in self.raw_file_names:
            table_path = osp.join(self.raw_dir, table)
            if not osp.isdir(table_path):
                return None
            parts = {
                int(d[len(self._PARTITION_PREFIX) :])
                for d in os.listdir(table_path)
                if d.startswith(self._PARTITION_PREFIX)
            }
            if not parts:
                return None
            partition_sets[table] = parts

        reference_table = self.raw_file_names[0]
        reference = partition_sets[reference_table]
        mismatches = {t: p for t, p in partition_sets.items() if p != reference}
        if mismatches:
            details = ", ".join(f"{t}={sorted(p)}" for t, p in mismatches.items())
            raise ValueError(
                f"Hive partition sets are inconsistent across tables. "
                f"{reference_table} has partitions {sorted(reference)}, "
                f"but the following tables differ: {details}. "
                f"Re-generate your partitioned data so all tables share the same "
                f"scenario_partition values.",
            )

        return sorted(reference)

    def _process_streaming(self, partitions: list[int]) -> None:
        """Build the scenario cache by reading one Hive partition at a time.

        Args:
            partitions: Sorted list of scenario_partition values to process.

        Raises:
            ValueError: If a scenario id appears in more than one partition,
                violating the datakit invariant that each scenario belongs to
                exactly one scenario_partition.
            ValueError: If the accumulated scenario ids across all partitions
                are not a contiguous 0..N-1 range.
        """
        done_path = osp.join(self.processed_dir, self.processed_done_file)
        if osp.exists(done_path):
            print("Processed files already exist. Skipping processing.")
            return

        print(f"Streaming {len(partitions)} partitions...")
        load_scenarios: dict[int, int] = {}
        # Maps scenario id → partition it was first seen in, to enforce the
        # datakit invariant: every scenario lives in exactly one partition.
        seen_scenarios: dict[int, int] = {}
        for partition_val in tqdm(partitions, desc="Processing partitions"):
            bus_data = pd.read_parquet(
                self._partition_dir("bus_data.parquet", partition_val),
            )
            gen_data = pd.read_parquet(
                self._partition_dir("gen_data.parquet", partition_val),
            )
            branch_data = pd.read_parquet(
                self._partition_dir("branch_data.parquet", partition_val),
            )

            partition_scenarios = set(int(s) for s in bus_data["scenario"].unique())
            duplicates = partition_scenarios & seen_scenarios.keys()
            if duplicates:
                offender = min(duplicates)
                raise ValueError(
                    f"Scenario {offender} appears in both partition "
                    f"{seen_scenarios[offender]} and partition {partition_val}. "
                    f"Each scenario must be fully contained within a single "
                    f"scenario_partition so that per-partition Q-limit aggregation "
                    f"(agg_gen) is equivalent to the legacy whole-dataset merge. "
                    f"Re-partition your data so no scenario spans two partitions.",
                )
            for s in partition_scenarios:
                seen_scenarios[s] = partition_val

            if "load_scenario_idx" in bus_data.columns:
                first_idx = bus_data.groupby("scenario")["load_scenario_idx"].first()
                load_scenarios.update(first_idx.to_dict())

            # Invariant: every scenario's rows are fully contained in this
            # partition, so summing Q-limits here equals the legacy whole-dataset
            # groupby sum.
            agg_gen = (
                gen_data.groupby(["scenario", "bus"])[["min_q_mvar", "max_q_mvar"]]
                .sum()
                .reset_index()
            )
            bus_data = bus_data.merge(
                agg_gen,
                on=["scenario", "bus"],
                how="left",
            ).fillna(0)

            bus_groups = bus_data.groupby("scenario")
            gen_groups = gen_data.groupby("scenario")
            branch_groups = branch_data.groupby("scenario")

            for scenario in bus_data["scenario"].unique():
                if osp.exists(
                    osp.join(self.processed_dir, f"data_index_{scenario}.pt"),
                ):
                    continue
                if (
                    scenario not in gen_groups.groups
                    or scenario not in branch_groups.groups
                ):
                    raise ValueError(
                        f"Scenario {scenario} is missing generator or branch data.",
                    )
                self._build_and_save_scenario(
                    scenario,
                    bus_groups.get_group(scenario),
                    gen_groups.get_group(scenario),
                    branch_groups.get_group(scenario),
                )

        # Mirror the legacy contiguity assertion: predict-step indexing
        # relies on scenario ids forming a complete 0..N-1 range.
        n = len(seen_scenarios)
        actual_min = min(seen_scenarios)
        actual_max = max(seen_scenarios)
        if actual_min != 0 or actual_max != n - 1:
            raise ValueError(
                f"Scenario ids are not contiguous integers from 0 to {n - 1}. "
                f"Found min={actual_min}, max={actual_max}, count={n}. "
                f"Ensure no scenarios are missing from your partitioned data.",
            )

        if load_scenarios:
            ordered = [load_scenarios[s] for s in sorted(load_scenarios)]
            torch.save(
                torch.tensor(ordered),
                osp.join(self.processed_dir, "load_scenarios.pt"),
            )

        with open(done_path, "w") as f:
            f.write("done")

    def _build_and_save_scenario(
        self,
        scenario: int,
        bus_df: pd.DataFrame,
        gen_df: pd.DataFrame,
        branch_df: pd.DataFrame,
    ) -> None:
        """Build and persist one scenario's heterogeneous graph to disk."""
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

        assert (bus_df["bus"].values == torch.arange(len(bus_df))).all(), (
            "Buses are not in increasing order"
        )

        data = HeteroData()

            # Bus nodes
            bus_df = bus_groups.get_group(scenario)
            # assert that the buses are in increasing order
            assert (bus_df["bus"].values == torch.arange(len(bus_df))).all(), (
                "Buses are not in increasing order"
            )
            # todo: we should remove this assert and store the bus idx in the tensors
            # right now we need the increasing order for e.g. the predict step that uses torch.arange(n_nodes) to index the buses.
            data["bus"].x = torch.tensor(bus_df[bus_features].values, dtype=torch.float)
            data["bus"].static = (
                data["bus"]
                .x[:, [MIN_VM_H, MAX_VM_H, MIN_QG_H, MAX_QG_H, VN_KV]]
                .clone()
            )

        # Generator nodes
        gen_df = gen_df.reset_index()
        data["gen"].x = torch.tensor(gen_df[gen_features].values, dtype=torch.float)
        gen_df["gen_index"] = gen_df.index

        data["bus"].y = data["bus"].x[:, : (VA_H + 1)].clone()
        data["gen"].y = data["gen"].x[:, : (PG_H + 1)].clone()

        # Bus-Bus edges
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
            data["bus", "connects", "bus"].static = edge_attr[
                :,
                [ANG_MIN, ANG_MAX, RATE_A],
            ].clone()
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

        torch.save(
            data.to_dict(),
            osp.join(self.processed_dir, f"data_index_{scenario}.pt"),
        )

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
