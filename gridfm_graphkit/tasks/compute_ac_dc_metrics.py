"""Compute AC/DC power balance residuals and runtime statistics on test splits."""

import json
import os
import numpy as np
import pandas as pd
from gridfm_datakit.utils.power_balance import (
    compute_branch_powers_vectorized,
    compute_bus_balance,
)

N_SCENARIO_PER_PARTITION = 200
NUM_PROCESSES = 64


def _load_test_data(data_dir: str, test_scenario_ids: list[int]):
    partitions = sorted(set(s // N_SCENARIO_PER_PARTITION for s in test_scenario_ids))
    test_set = set(test_scenario_ids)
    partition_filter = [("scenario_partition", "in", partitions)]

    bus_df = pd.read_parquet(
        os.path.join(data_dir, "bus_data.parquet"),
        filters=partition_filter,
    )
    branch_df = pd.read_parquet(
        os.path.join(data_dir, "branch_data.parquet"),
        filters=partition_filter,
    )
    runtime_df = pd.read_parquet(
        os.path.join(data_dir, "runtime_data.parquet"),
        filters=partition_filter,
    )

    bus_df = bus_df[bus_df["scenario"].isin(test_set)].reset_index(drop=True)
    branch_df = branch_df[branch_df["scenario"].isin(test_set)].reset_index(drop=True)
    runtime_df = runtime_df[runtime_df["scenario"].isin(test_set)].reset_index(
        drop=True,
    )

    print(
        f"  Loaded {len(bus_df)} bus rows, {len(branch_df)} branch rows, "
        f"{len(runtime_df)} runtime rows for {len(test_set)} test scenarios",
    )
    return bus_df, branch_df, runtime_df


def _compute_residual_stats(balance_df: pd.DataFrame, dc: bool) -> dict:
    grouped = balance_df.groupby("scenario")

    if dc:
        P_mis = balance_df["P_mis_dc"].to_numpy()
        nan_scenarios = int(grouped["P_mis_dc"].apply(lambda x: x.isna().all()).sum())
        return {
            "Avg. active res. (MW)": float(np.nanmean(np.abs(P_mis))),
            "DC NaN scenarios": nan_scenarios,
        }

    P_mis = balance_df["P_mis_ac"].to_numpy()
    Q_mis = balance_df["Q_mis_ac"].to_numpy()
    pbe = np.sqrt(P_mis**2 + Q_mis**2)

    pbe_per_scenario_mean = grouped.apply(
        lambda g: np.nanmean(
            np.sqrt(g["P_mis_ac"].to_numpy() ** 2 + g["Q_mis_ac"].to_numpy() ** 2),
        ),
        include_groups=False,
    )

    return {
        "Avg. active res. (MW)": float(np.nanmean(np.abs(P_mis))),
        "Avg. reactive res. (MVar)": float(np.nanmean(np.abs(Q_mis))),
        "PBE Mean": float(np.nanmean(pbe_per_scenario_mean)),
        "PBE Max": float(np.nanmax(pbe)),
    }


def _compute_runtime_stats(runtime_df: pd.DataFrame) -> dict:
    results = {}
    for mode in ["ac", "dc"]:
        if mode not in runtime_df.columns:
            continue

        rt_ms = runtime_df[mode].to_numpy(dtype=float) * 1000.0 / NUM_PROCESSES
        valid = rt_ms[~np.isnan(rt_ms)]

        results[f"runtime_{mode}_mean_ms_with_{NUM_PROCESSES}_cores"] = float(
            np.mean(valid),
        )
        results[f"runtime_{mode}_median_ms_with_{NUM_PROCESSES}_cores"] = float(
            np.median(valid),
        )
        results[f"runtime_{mode}_std_ms_with_{NUM_PROCESSES}_cores"] = float(
            np.std(valid),
        )
        results[f"runtime_{mode}_max_ms_with_{NUM_PROCESSES}_cores"] = float(
            np.max(valid),
        )

    return results


def compute_ac_dc_metrics(
    artifacts_dir: str,
    data_dir: str,
    grid_name: str,
    sn_mva: float,
) -> bool:
    """Compute AC/DC ground-truth power balance and runtime metrics, save results.

    Saves:
        - Aggregated metrics (CSV)
        - AC per-bus residuals (Parquet)
        - DC per-bus residuals (Parquet)

    Returns:
        True if metrics were computed, False if splits JSON was not found.
    """

    splits_json = os.path.join(
        artifacts_dir,
        "stats",
        f"{grid_name}_scenario_splits.json",
    )
    if not os.path.exists(splits_json):
        print(f"  Skipping: no splits JSON found at {splits_json}")
        return False

    with open(splits_json) as f:
        test_ids = json.load(f)["test"]

    print(f"  Test split: {len(test_ids)} scenarios")

    bus_df, branch_df, runtime_df = _load_test_data(data_dir, test_ids)

    # =========================
    # AC residuals
    # =========================
    print("  Computing AC power balance...")
    balance_ac = compute_bus_balance(
        bus_df,
        branch_df,
        branch_df[["pf", "qf", "pt", "qt"]],
        dc=False,
        sn_mva=sn_mva,
    )

    ac_stats = _compute_residual_stats(balance_ac, dc=False)

    # =========================
    # DC residuals
    # =========================
    print("  Computing DC power balance...")
    pf_dc, _, pt_dc, _ = compute_branch_powers_vectorized(
        branch_df,
        bus_df,
        dc=True,
        sn_mva=sn_mva,
    )

    balance_dc = compute_bus_balance(
        bus_df,
        branch_df,
        pd.DataFrame(
            {"pf_dc": pf_dc, "pt_dc": pt_dc},
            index=branch_df.index,
        ),
        dc=True,
        sn_mva=sn_mva,
    )

    dc_stats = _compute_residual_stats(balance_dc, dc=True)

    # =========================
    # Save per-bus residuals (PARQUET)
    # =========================
    out_dir = os.path.join(artifacts_dir, "test")
    os.makedirs(out_dir, exist_ok=True)

    # AC: active + reactive
    ac_bus_residuals = (
        balance_ac[["scenario", "bus", "P_mis_ac", "Q_mis_ac"]]
        .copy()
        .rename(
            columns={
                "P_mis_ac": "active res. (MW)",
                "Q_mis_ac": "reactive res. (MVar)",
            },
        )
    )
    ac_residuals_path = os.path.join(out_dir, f"{grid_name}_ac_bus_residuals.parquet")
    ac_bus_residuals.to_parquet(ac_residuals_path, index=False)
    print(f"  AC per-bus residuals saved to {ac_residuals_path}")

    # DC: active only
    dc_bus_residuals = (
        balance_dc[["scenario", "bus", "P_mis_dc"]]
        .copy()
        .rename(
            columns={
                "P_mis_dc": "DC active res. (MW)",
            },
        )
    )

    dc_residuals_path = os.path.join(out_dir, f"{grid_name}_dc_bus_residuals.parquet")
    dc_bus_residuals.to_parquet(dc_residuals_path, index=False)
    print(f"  DC per-bus residuals saved to {dc_residuals_path}")

    # =========================
    # Save aggregated metrics (CSV)
    # =========================
    runtime_stats = _compute_runtime_stats(runtime_df)

    rows = []
    for key, val in ac_stats.items():
        rows.append({"Metric": f"AC {key}", "Value": val})
    for key, val in dc_stats.items():
        rows.append({"Metric": f"DC {key}", "Value": val})
    for key, val in runtime_stats.items():
        rows.append({"Metric": key, "Value": val})

    metrics_path = os.path.join(out_dir, f"{grid_name}_ac_dc_metrics.csv")
    pd.DataFrame(rows).to_csv(metrics_path, index=False)

    print(f"  Aggregated metrics saved to {metrics_path}")

    return True
