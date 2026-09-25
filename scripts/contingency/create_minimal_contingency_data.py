"""
Build minimal, already-intersected contingency analysis inputs.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

_parser = argparse.ArgumentParser()
_parser.add_argument("--predictions", type=Path, required=True)
_parser.add_argument("--bus-data", type=Path, required=True)
_parser.add_argument("--branch-data", type=Path, required=True)
_parser.add_argument("--output-dir", type=Path, required=True)
_args = _parser.parse_args()
PREDICTIONS_PATH = _args.predictions
BUS_DATA_PATH = _args.bus_data
BRANCH_DATA_PATH = _args.branch_data
_args.output_dir.mkdir(parents=True, exist_ok=True)
FILTERED_PREDS_LIGHT_PATH = _args.output_dir / "filtered_preds_light.parquet"
FILTERED_BUS_LIGHT_PATH = _args.output_dir / "filtered_bus_data_light.parquet"
FILTERED_BRANCH_LIGHT_PATH = _args.output_dir / "filtered_branch_data_light.parquet"


def main() -> None:
    preds = pd.read_parquet(PREDICTIONS_PATH)
    bus_data = pd.read_parquet(BUS_DATA_PATH)
    branch_data = pd.read_parquet(BRANCH_DATA_PATH)

    shared_scenarios = np.intersect1d(
        preds["scenario"].to_numpy(),
        bus_data["scenario"].to_numpy(),
    )
    # Keep only scenarios that exist in both predictions and AC bus truth.
    filtered_preds = preds[preds["scenario"].isin(shared_scenarios)].copy()
    filtered_bus_data = bus_data[bus_data["scenario"].isin(shared_scenarios)].copy()
    filtered_branch_data = branch_data[branch_data["scenario"].isin(shared_scenarios)].copy()

    # Keep only fields required by contingency_analysis.py.
    preds_needed_cols = ["scenario", "bus", "vm_pu", "va"]
    bus_needed_cols = ["scenario", "bus", "Vm", "Va", "Va_dc", "PV", "REF"]
    branch_needed_cols = [
        "scenario",
        "from_bus",
        "to_bus",
        "rate_a",
        "Yff_r",
        "Yff_i",
        "Yft_r",
        "Yft_i",
        "Ytf_r",
        "Ytf_i",
        "Ytt_r",
        "Ytt_i",
    ]

    # Use explicit prediction names so merged columns are unambiguous.
    filtered_preds_light = filtered_preds[preds_needed_cols].rename(
        columns={
            "vm_pu": "Vm_pred",
            "va": "Va_pred",
        }
    )
    filtered_bus_data_light = filtered_bus_data[bus_needed_cols].copy()
    filtered_branch_data_light = filtered_branch_data[branch_needed_cols].copy()

    filtered_preds_light.to_parquet(FILTERED_PREDS_LIGHT_PATH, index=False)
    filtered_bus_data_light.to_parquet(FILTERED_BUS_LIGHT_PATH, index=False)
    filtered_branch_data_light.to_parquet(FILTERED_BRANCH_LIGHT_PATH, index=False)

    print(f"Saved {FILTERED_PREDS_LIGHT_PATH}")
    print(f"Saved {FILTERED_BUS_LIGHT_PATH}")
    print(f"Saved {FILTERED_BRANCH_LIGHT_PATH}")


if __name__ == "__main__":
    main()
