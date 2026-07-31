#!/usr/bin/env python3
"""
Parallel preprocessing for HeteroGridDatasetDisk raw parquet partitions.

Loads each parquet partition sequentially, then processes its scenarios in
parallel. Output files match the sequential ``process()`` path.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp

import numpy as np
import torch
from tqdm import tqdm

from gridfm_graphkit.datasets.hetero_preprocess import (
    REACTIVE_CORRECTION_MODES,
    assert_scenario_index,
    build_load_scenarios,
    get_partition_ids,
    merge_agg_gen_into_bus,
    process_partition,
    process_scenarios,
    read_raw_tables,
    validate_raw_partition_layout,
)

PROCESSED_DONE_FILE = "processed_raw_files.done"
REACTIVE_CORRECTION_MARKER = "reactive_correction.json"


def process_dataset_parallel(
    root: str,
    *,
    workers: int = 1,
    skip_existing: bool = True,
    force: bool = False,
    reactive_correction: str | None = None,
    base_mva: float = 100.0,
) -> None:
    raw_dir = osp.join(root, "raw")
    processed_dir = osp.join(root, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    print(f"Dataset root: {root}")
    print(f"Raw dir:      {raw_dir}")
    print(f"Processed dir:{processed_dir}")
    print(f"Reactive correction: {reactive_correction or 'none'}"
          f"{f' (baseMVA={base_mva})' if reactive_correction else ''}")

    done_path = osp.join(processed_dir, PROCESSED_DONE_FILE)
    if osp.exists(done_path) and not force:
        print("Processed files already exist. Skipping processing.")
        return

    partition_ids = get_partition_ids(raw_dir)
    if partition_ids is not None:
        print("Detected partitioned parquet layout.")
        partition_ids = validate_raw_partition_layout(raw_dir)
        print(f"Found {len(partition_ids)} partitions across bus/gen/branch tables.")
    else:
        print("No parquet partitions found; using sequential full-file processing.")

    if partition_ids is None:
        print("Loading full raw tables...")
        bus_data, gen_data, branch_data = read_raw_tables(raw_dir)
        assert_scenario_index(bus_data)
        n_scenarios = int(bus_data["scenario"].nunique())
        print(f"Validated {n_scenarios} scenarios (0 .. {n_scenarios - 1}).")

        load_scenarios = build_load_scenarios(bus_data)
        if load_scenarios is not None:
            load_path = osp.join(processed_dir, "load_scenarios.pt")
            torch.save(load_scenarios, load_path)
            print(f"Saved load_scenarios.pt ({load_scenarios.shape[0]} entries) -> {load_path}")

        bus_data = merge_agg_gen_into_bus(bus_data, gen_data)
        process_scenarios(
            bus_data,
            gen_data,
            branch_data,
            processed_dir,
            skip_existing=skip_existing,
            show_progress=True,
            workers=workers,
            reactive_correction=reactive_correction,
            base_mva=base_mva,
        )
    else:
        load_chunks = []
        previous_scenario_max = None
        print(
            f"Processing {len(partition_ids)} partitions sequentially "
            f"with {workers} scenario worker(s) per partition "
            f"(skip_existing={skip_existing})",
        )

        for partition_id in tqdm(partition_ids, desc="Partitions", unit="partition"):
            previous_scenario_max, load_chunk = process_partition(
                raw_dir,
                processed_dir,
                partition_id,
                skip_existing=skip_existing,
                show_progress=True,
                workers=workers,
                previous_scenario_max=previous_scenario_max,
                reactive_correction=reactive_correction,
                base_mva=base_mva,
            )
            if load_chunk is not None:
                load_chunks.append(load_chunk)

        if load_chunks:
            load_scenarios = torch.tensor(np.concatenate(load_chunks))
            load_path = osp.join(processed_dir, "load_scenarios.pt")
            torch.save(load_scenarios, load_path)
            print(
                f"Saved load_scenarios.pt ({load_scenarios.shape[0]} entries) -> {load_path}",
            )

    with open(done_path, "w", encoding="utf-8") as done_file:
        done_file.write("done")
    print(f"Done. Wrote {PROCESSED_DONE_FILE} -> {done_path}")

    # Provenance marker: record that (and how) the ground-truth reactive balance was
    # corrected, so a corrected dataset is distinguishable from a raw one on disk.
    if reactive_correction is not None:
        marker_path = osp.join(processed_dir, REACTIVE_CORRECTION_MARKER)
        with open(marker_path, "w", encoding="utf-8") as marker_file:
            json.dump({"mode": reactive_correction}, marker_file)
        print(f"Wrote {REACTIVE_CORRECTION_MARKER} (mode={reactive_correction}) -> {marker_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parallel HeteroGridDatasetDisk preprocessing by scenario.",
    )
    parser.add_argument(
        "root",
        help="Dataset root containing raw/ and processed/ (e.g. .../case10000_goc).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of parallel scenario workers per partition (default: CPU count).",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Reprocess scenarios even if data_index_<scenario>.pt already exists.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run even if processed_raw_files.done exists.",
    )
    parser.add_argument(
        "--reactive-correction",
        choices=("none", *REACTIVE_CORRECTION_MODES),
        default="none",
        help=(
            "Absorb the ground-truth reactive-power residual per bus at creation time. "
            "'none' (default): no correction, dataset built as-is. "
            "'qd_all': add residual to Qd on every bus. "
            "'qd_pq_qg_pvref': Qd on PQ buses, Qg on PV/REF buses."
        ),
    )
    parser.add_argument(
        "--base-mva",
        type=float,
        default=100.0,
        help="System base power (MVA) used to scale the per-unit branch/shunt flows "
        "when correcting (should match data.baseMVA; default 100).",
    )
    args = parser.parse_args()

    reactive_correction = None if args.reactive_correction == "none" else args.reactive_correction

    process_dataset_parallel(
        args.root,
        workers=max(1, args.workers),
        skip_existing=not args.no_skip_existing,
        force=args.force,
        reactive_correction=reactive_correction,
        base_mva=args.base_mva,
    )


if __name__ == "__main__":
    main()
