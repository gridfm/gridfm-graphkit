"""Compute AC/DC power balance residuals and runtime statistics on the test split.


e.g.


python scripts/get_dc_ac_metrics.py \
  --experiment-id 666779732630359422 \
  --data-dir /dccstor/gridfm/powermodels_data/v4/contingency/ \
  --grid-name Texas2k_case1_2016summerpeak \
  --log-dir /dccstor/gridfm/mlflow_alban_contingency
  
  """

import argparse
import json
import os
import numpy as np
import pandas as pd
from gridfm_datakit.utils.power_balance import (
    compute_branch_powers_vectorized,
    compute_bus_balance,
)

SN_MVA = 100.0
N_SCENARIO_PER_PARTITION = 200
NUM_PROCESSES = 64


def load_test_data(data_dir: str, test_scenario_ids: list[int]):
    """Load bus, branch, and runtime parquet data filtered to test scenarios."""
    partitions = sorted(set(s // N_SCENARIO_PER_PARTITION for s in test_scenario_ids))
    test_set = set(test_scenario_ids)

    partition_filter = [("scenario_partition", "in", partitions)]

    bus_df = pd.read_parquet(
        os.path.join(data_dir, "bus_data.parquet"), filters=partition_filter
    )
    branch_df = pd.read_parquet(
        os.path.join(data_dir, "branch_data.parquet"), filters=partition_filter
    )
    runtime_df = pd.read_parquet(
        os.path.join(data_dir, "runtime_data.parquet"), filters=partition_filter
    )

    bus_df = bus_df[bus_df["scenario"].isin(test_set)].reset_index(drop=True)
    branch_df = branch_df[branch_df["scenario"].isin(test_set)].reset_index(drop=True)
    runtime_df = runtime_df[runtime_df["scenario"].isin(test_set)].reset_index(drop=True)

    print(f"Loaded {len(bus_df)} bus rows, {len(branch_df)} branch rows, "
          f"{len(runtime_df)} runtime rows for {len(test_set)} test scenarios")
    return bus_df, branch_df, runtime_df


def compute_residual_stats(balance_df: pd.DataFrame, dc: bool) -> dict:
    """Compute the 4 pf_task metrics: avg active/reactive residuals, PBE mean/max."""
    grouped = balance_df.groupby("scenario")

    if dc:
        P_mis = balance_df["P_mis_dc"].to_numpy()
        nan_scenarios = int(grouped["P_mis_dc"].apply(lambda x: x.isna().all()).sum())
        avg_active_res = float(np.nanmean(np.abs(P_mis)))
        return {
            "Avg. active res. (MW)": avg_active_res,
            "DC NaN scenarios": nan_scenarios,
        }
    else:
        P_mis = balance_df["P_mis_ac"].to_numpy()
        Q_mis = balance_df["Q_mis_ac"].to_numpy()
        pbe = np.sqrt(P_mis**2 + Q_mis**2)
        pbe_per_scenario_mean = grouped.apply(
            lambda g: np.nanmean(np.sqrt(g["P_mis_ac"].to_numpy()**2 + g["Q_mis_ac"].to_numpy()**2)),
            include_groups=False,
        )
        return {
            "Avg. active res. (MW)": float(np.nanmean(np.abs(P_mis))),
            "Avg. reactive res. (MVar)": float(np.nanmean(np.abs(Q_mis))),
            "PBE Mean": float(np.nanmean(pbe_per_scenario_mean)),
            "PBE Max": float(np.nanmax(pbe)),
        }


def compute_runtime_stats(runtime_df: pd.DataFrame) -> dict:
    """Compute runtime statistics (in ms) for AC and DC solvers."""
    results = {}
    for mode in ["ac", "dc"]:
        if mode not in runtime_df.columns:
            continue
        rt_ms = runtime_df[mode].to_numpy(dtype=float) * 1000.0 / NUM_PROCESSES
        valid = rt_ms[~np.isnan(rt_ms)]
        results[f"runtime_{mode}_mean_ms_with_64_cores"] = float(np.mean(valid))
        results[f"runtime_{mode}_std_ms_with_64_cores"] = float(np.std(valid))
        results[f"runtime_{mode}_max_ms_with_64_cores"] = float(np.max(valid))
    return results


def process_run(experiment_id: str, run_id: str, data_dir: str, log_dir: str, grid_name: str):
    """Compute and save AC/DC metrics for a single run."""
    run_dir = os.path.join(log_dir, experiment_id, run_id)
    artifacts_dir = os.path.join(run_dir, "artifacts")
    splits_json = os.path.join(
        artifacts_dir, "stats", f"{grid_name}_scenario_splits.json"
    )

    if not os.path.exists(splits_json):
        print(f"  Skipping {run_id}: no splits JSON found")
        return

    with open(splits_json) as f:
        splits = json.load(f)
    test_ids = splits["test"]
    print(f"  Test split: {len(test_ids)} scenarios")

    bus_df, branch_df, runtime_df = load_test_data(data_dir+f"/{grid_name}/raw/", test_ids)

    # AC residuals
    print("  Computing AC power balance...")
    balance_ac = compute_bus_balance(
        bus_df,
        branch_df,
        branch_df[["pf", "qf", "pt", "qt"]],
        dc=False,
        sn_mva=SN_MVA,
    )
    ac_stats = compute_residual_stats(balance_ac, dc=False)

    # DC residuals
    print("  Computing DC power balance...")
    pf_dc, _, pt_dc, _ = compute_branch_powers_vectorized(
        branch_df, bus_df, dc=True, sn_mva=SN_MVA
    )
    flows_dc = pd.DataFrame(
        {"pf_dc": pf_dc, "pt_dc": pt_dc}, index=branch_df.index
    )
    balance_dc = compute_bus_balance(
        bus_df, branch_df, flows_dc, dc=True, sn_mva=SN_MVA
    )
    dc_stats = compute_residual_stats(balance_dc, dc=True)

    # Runtime stats
    runtime_stats = compute_runtime_stats(runtime_df)

    # Build CSV
    rows = []
    for key, val in ac_stats.items():
        rows.append({"Metric": f"AC {key}", "Value": val})
    for key, val in dc_stats.items():
        rows.append({"Metric": f"DC {key}", "Value": val})
    for key, val in runtime_stats.items():
        rows.append({"Metric": key, "Value": val})

    df = pd.DataFrame(rows)
    out_dir = os.path.join(artifacts_dir, "test")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ac_dc_metrics.csv")
    df.to_csv(out_path, index=False)
    print(f"  Results saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--log-dir", type=str, default="mlruns")
    parser.add_argument("--grid-name", type=str, required=True)
    args = parser.parse_args()

    if args.run_id:
        run_ids = [args.run_id]
    else:
        exp_dir = os.path.join(args.log_dir, args.experiment_id)
        run_ids = [
            d for d in os.listdir(exp_dir)
            if os.path.isdir(os.path.join(exp_dir, d))
        ]
        print(f"Found {len(run_ids)} runs in experiment {args.experiment_id}")

    for run_id in sorted(run_ids):
        print(f"\nProcessing run {run_id}...")
        process_run(args.experiment_id, run_id, args.data_dir, args.log_dir, args.grid_name)


if __name__ == "__main__":
    main()
