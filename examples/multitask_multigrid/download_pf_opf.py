"""Download the PF and OPF datakit slices used by the multi-task runs.

Layout under ``--dest``::

    pf/case14_ieee/raw/{bus,gen,branch}_data.parquet/scenario_partition=N/
    opf/case57_ieee/raw/...

Case14 and case30 get partitions 0-4 (1,000 scenarios). Case57 gets
partitions 0-29 (6,000 scenarios) so the 1,000-scenario and 6,000-scenario
case57 runs share one processed cache.
"""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

TABLES = ("bus_data.parquet", "gen_data.parquet", "branch_data.parquet")
JOBS = (
    ("pf", "gridfm/pf_small_case14_ieee", "case14_ieee", 5),
    ("pf", "gridfm/pf_small_case30_ieee", "case30_ieee", 5),
    ("pf", "gridfm/pf_small_case57_ieee", "case57_ieee", 30),
    ("opf", "gridfm/opf_small_case14_ieee", "case14_ieee", 5),
    ("opf", "gridfm/opf_small_case30_ieee", "case30_ieee", 5),
    ("opf", "gridfm/opf_small_case57_ieee", "case57_ieee", 30),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", required=True, help="Root passed to train as --data_path")
    args = parser.parse_args()
    dest = Path(args.dest)
    for kind, repo, name, n_parts in JOBS:
        patterns = [
            f"{table}/scenario_partition={i}/*"
            for i in range(n_parts)
            for table in TABLES
        ]
        out = dest / kind / name / "raw"
        out.mkdir(parents=True, exist_ok=True)
        print(f"downloading {repo} -> {out} ({n_parts} partitions)")
        snapshot_download(
            repo,
            repo_type="dataset",
            allow_patterns=patterns,
            local_dir=str(out),
        )
        for table in TABLES:
            found = list((out / table).glob("scenario_partition=*"))
            if len(found) < n_parts:
                raise SystemExit(
                    f"{out / table} has {len(found)} partitions, expected {n_parts}",
                )
    print("download ok")


if __name__ == "__main__":
    main()
