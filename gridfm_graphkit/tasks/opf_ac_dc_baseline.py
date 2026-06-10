"""Compute AC/DC OPF baseline metrics on test splits.

Uses the same AC/DC power-balance and residual aggregation as
:mod:`gridfm_graphkit.tasks.pf_ac_dc_baseline` (via shared helpers).

Adds OPF-style inequality metrics. Compared to
:mod:`gridfm_graphkit.tasks.opf_task` ``test_step``:

- **Residuals / runtime**: Same formulas as the PF baseline (ground-truth
  ``compute_bus_balance`` on parquet), not the neural ``ComputeNodeResiduals``
  in the task.
- **Optimality**: The task logs ``Opt gap`` = mean(|cost_pred − cost_gt| / cost_gt)
  per scenario (model vs label). Here **DC Mean optimality gap (%)** is the mean
  over scenarios of |cost_dc − cost_ac| / cost_ac × 100 with scenario totals from
  ``p_mw`` vs ``p_mw_dc`` (DC solution vs AC reference)
- **Branch thermal / angle / Qg**: Same relu-style violations and flat means as
  the task; **Pg bound** violations are baseline-only (not logged in ``opf_task``).
"""

import json
import os

import numpy as np
import pandas as pd
from gridfm_datakit.utils.power_balance import (
    compute_branch_powers_vectorized,
    compute_bus_balance,
)
from gridfm_graphkit.tasks.pf_ac_dc_baseline import (
    N_SCENARIO_PER_PARTITION,
    NUM_PROCESSES,
    _compute_residual_stats,
    _compute_runtime_stats,
)


def _load_test_data(data_dir: str, test_scenario_ids: list[int]):
    """Load OPF test-split bus/gen/branch/runtime tables from partitioned parquet."""
    partitions = sorted(set(s // N_SCENARIO_PER_PARTITION for s in test_scenario_ids))
    test_set = set(test_scenario_ids)
    partition_filter = [("scenario_partition", "in", partitions)]

    bus_df = pd.read_parquet(
        os.path.join(data_dir, "bus_data.parquet"),
        filters=partition_filter,
    )
    gen_df = pd.read_parquet(
        os.path.join(data_dir, "gen_data.parquet"),
        filters=partition_filter,
    )
    # drop where in_service is 0 so the means are computed over the same number of gens as for the model in opf_task
    gen_df = gen_df[gen_df["in_service"] == 1].reset_index(drop=True)
    branch_df = pd.read_parquet(
        os.path.join(data_dir, "branch_data.parquet"),
        filters=partition_filter,
    )
    branch_df = branch_df.drop(columns=["pf_dc", "pt_dc"], axis=1)
    # drop where br_status is 0 so the means are computed over the same number of branches as for the model in opf_task
    branch_df = branch_df[branch_df["br_status"] == 1].reset_index(drop=True)
    runtime_df = pd.read_parquet(
        os.path.join(data_dir, "runtime_data.parquet"),
        filters=partition_filter,
    )

    bus_df = bus_df[bus_df["scenario"].isin(test_set)].reset_index(drop=True)
    gen_df = gen_df[gen_df["scenario"].isin(test_set)].reset_index(drop=True)
    branch_df = branch_df[branch_df["scenario"].isin(test_set)].reset_index(drop=True)
    runtime_df = runtime_df[runtime_df["scenario"].isin(test_set)].reset_index(drop=True)

    print(
        f"  Loaded {len(bus_df)} bus rows, {len(gen_df)} gen rows, "
        f"{len(branch_df)} branch rows, {len(runtime_df)} runtime rows "
        f"for {len(test_set)} test scenarios",
    )
    return bus_df, gen_df, branch_df, runtime_df


def _compute_optimality_gap(gen_df: pd.DataFrame) -> dict:
    """Compute mean AC/DC scenario-level optimality gap from generator costs."""
    # Same aggregation as opf_task scatter_add + mean over graphs, but compares
    # scenario DC cost vs AC cost (not model pred vs GT).
    c0 = gen_df["cp0_eur"].to_numpy(dtype=float)
    c1 = gen_df["cp1_eur_per_mw"].to_numpy(dtype=float)
    c2 = gen_df["cp2_eur_per_mw2"].to_numpy(dtype=float)
    pg_ac = gen_df["p_mw"].to_numpy(dtype=float)
    pg_dc = gen_df["p_mw_dc"].to_numpy(dtype=float)
    g = gen_df.copy()
    g["cost_ac"] = (c0 + c1 * pg_ac + c2 * pg_ac * pg_ac) * g["in_service"] # all is already in MW
    g["cost_dc"] = (c0 + c1 * pg_dc + c2 * pg_dc * pg_dc) * g["in_service"] # all is already in MW
    per_scenario = g.groupby("scenario")[["cost_ac", "cost_dc"]].sum()
    cost_ac = per_scenario["cost_ac"].to_numpy(dtype=float)
    cost_dc = per_scenario["cost_dc"].to_numpy(dtype=float)
    gap_pct = np.abs((cost_dc - cost_ac) / cost_ac * 100.0)
    return {
        "AC Mean optimality gap (%)": 0.0,
        "DC Mean optimality gap (%)": float(np.nanmean(gap_pct)),
    }


def _compute_pg_violations(gen_df: pd.DataFrame) -> dict:
    """Compute mean AC/DC generator active-power bound violations."""
    min_p = gen_df["min_p_mw"].to_numpy(dtype=float)
    max_p = gen_df["max_p_mw"].to_numpy(dtype=float)
    pg_ac = gen_df["p_mw"].to_numpy(dtype=float)
    pg_dc = gen_df["p_mw_dc"].to_numpy(dtype=float)
    viol_ac = np.maximum(pg_ac - max_p, 0.0) + np.maximum(min_p - pg_ac, 0.0)
    viol_dc = np.maximum(pg_dc - max_p, 0.0) + np.maximum(min_p - pg_dc, 0.0)
    return {
        "AC Mean Pg bound violation (MW)": float(np.nanmean(viol_ac)),
        "DC Mean Pg bound violation (MW)": float(np.nanmean(viol_dc)),
    }


def _compute_qg_violations_ac(bus_df: pd.DataFrame, gen_df: pd.DataFrame) -> dict:
    """Compute AC reactive-power limit violations for PV/REF buses."""
    # opf_task style on bus Qg; AC only 
    bus = bus_df.copy()
    qg = bus["Qg"].to_numpy(dtype=float)
    agg_gen = (
    gen_df.groupby(["scenario", "bus"])[["min_q_mvar", "max_q_mvar"]]
    .sum()
    .reset_index())
    bus = bus.merge(agg_gen, on=["scenario", "bus"], how="left")
    assert bus[bus["PV"]==1]["min_q_mvar"].isna().sum() == 0, "PV buses have no min_q_mvar"
    assert bus[bus["PV"]==1]["max_q_mvar"].isna().sum() == 0, "PV buses have no max_q_mvar"
    assert bus[bus["REF"]==1]["min_q_mvar"].isna().sum() == 0, "REF buses have no min_q_mvar"
    assert bus[bus["REF"]==1]["max_q_mvar"].isna().sum() == 0, "REF buses have no max_q_mvar"
    bus["qg_violation_amount"] = np.maximum(qg - bus["max_q_mvar"], 0.0) + np.maximum(bus["min_q_mvar"] - qg, 0.0)
    pv = bus[bus["PV"] == 1]
    ref = bus[bus["REF"] == 1]
    pv_ref = bus[(bus["PV"] == 1) | (bus["REF"] == 1)]
    return {
        "AC Mean Qg violation PV buses": float(np.nanmean(pv["qg_violation_amount"].to_numpy(dtype=float))),
        "AC Mean Qg violation REF buses": float(np.nanmean(ref["qg_violation_amount"].to_numpy(dtype=float))),
        "AC Mean Qg violation": float(np.nanmean(pv_ref["qg_violation_amount"].to_numpy(dtype=float))),
    }


def _compute_branch_violations(branch_df: pd.DataFrame, bus_df: pd.DataFrame) -> dict:
    """Compute AC/DC branch thermal and angle-limit violation statistics."""
    rate = branch_df["rate_a"].to_numpy(dtype=float)
    ac_from = np.sqrt(
        branch_df["pf"].to_numpy(dtype=float) ** 2 + branch_df["qf"].to_numpy(dtype=float) ** 2,
    )
    ac_to = np.sqrt(
        branch_df["pt"].to_numpy(dtype=float) ** 2 + branch_df["qt"].to_numpy(dtype=float) ** 2,
    )
    dc_from = np.sqrt(branch_df["pf_dc_computed"].to_numpy(dtype=float) ** 2 + branch_df["qf_dc_computed"].to_numpy(dtype=float) ** 2) # reactive part is needed here
    dc_to = np.sqrt(branch_df["pt_dc_computed"].to_numpy(dtype=float) ** 2 + branch_df["qt_dc_computed"].to_numpy(dtype=float) ** 2)

    ac_thermal_from = np.maximum(ac_from - rate, 0.0)
    ac_thermal_to = np.maximum(ac_to - rate, 0.0)
    dc_thermal_from = np.maximum(dc_from - rate, 0.0)
    dc_thermal_to = np.maximum(dc_to - rate, 0.0)

    bus_angles = bus_df[["scenario", "bus", "Va", "Va_dc"]]
    # convert to radians
    bus_angles.loc[:, "Va"] = bus_angles["Va"] * np.pi / 180.0 
    bus_angles.loc[:, "Va_dc"] = bus_angles["Va_dc"] * np.pi / 180.0
    from_angles = bus_angles.rename(
        columns={"bus": "from_bus", "Va": "Va_from", "Va_dc": "Va_dc_from"},
    )
    to_angles = bus_angles.rename(
        columns={"bus": "to_bus", "Va": "Va_to", "Va_dc": "Va_dc_to"},
    )
    br = branch_df.merge(from_angles, on=["scenario", "from_bus"], how="left")
    br = br.merge(to_angles, on=["scenario", "to_bus"], how="left")
    
    # AC angle
    ac_angle_diff = br["Va_from"] - br["Va_to"]
    ac_angle_diff = (ac_angle_diff + np.pi) % (2 * np.pi) - np.pi # wrap    to [-pi, pi]
    ac_angle_excess_low = np.maximum(br["ang_min"] - ac_angle_diff, 0.0)
    ac_angle_excess_high = np.maximum(ac_angle_diff - br["ang_max"], 0.0)
    mean_ac_angle_violation = np.mean(ac_angle_excess_low + ac_angle_excess_high)
    # DC angle
    dc_angle_diff = br["Va_dc_from"] - br["Va_dc_to"]
    dc_angle_diff = (dc_angle_diff + np.pi) % (2 * np.pi) - np.pi
    dc_angle_excess_low = np.maximum(br["ang_min"] - dc_angle_diff, 0.0)
    dc_angle_excess_high = np.maximum(dc_angle_diff - br["ang_max"], 0.0)
    mean_dc_angle_violation = np.mean(dc_angle_excess_low + dc_angle_excess_high)

    return {
        "AC Mean branch thermal violation from (MVA)": float(np.nanmean(ac_thermal_from)),
        "AC Mean branch thermal violation to (MVA)": float(np.nanmean(ac_thermal_to)),
        "AC Mean branch angle difference violation (radians)": float(mean_ac_angle_violation),
        "DC Mean branch thermal violation from (MVA)": float(np.nanmean(dc_thermal_from)),
        "DC Mean branch thermal violation to (MVA)": float(np.nanmean(dc_thermal_to)),
        "DC Mean branch angle difference violation (radians)": float(mean_dc_angle_violation),
    }


def compute_opf_ac_dc_metrics(
    artifacts_dir: str,
    data_dir: str,
    grid_name: str,
    sn_mva: float,
) -> bool:
    """Compute AC/DC OPF baseline metrics (PF metrics + OPF inequalities), save results.

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

    bus_df, gen_df, branch_df, runtime_df = _load_test_data(data_dir, test_ids)

    print("  Computing AC power balance...")
    balance_ac = compute_bus_balance(
        bus_df,
        branch_df,
        branch_df[["pf", "qf", "pt", "qt"]],
        dc=False,
        sn_mva=sn_mva,
    )
    ac_stats = _compute_residual_stats(balance_ac, dc=False)

    print("  Computing DC power balance...")
    pf_dc, qf_dc, pt_dc, qt_dc = compute_branch_powers_vectorized(
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

    branch_df = branch_df.copy()
    branch_df["pf_dc_computed"] = pf_dc
    branch_df["pt_dc_computed"] = pt_dc
    branch_df["qf_dc_computed"] = qf_dc
    branch_df["qt_dc_computed"] = qt_dc
    

    opf_extra = {}
    opf_extra.update(_compute_optimality_gap(gen_df))
    opf_extra.update(_compute_branch_violations(branch_df, bus_df))
    opf_extra.update(_compute_pg_violations(gen_df))
    opf_extra.update(_compute_qg_violations_ac(bus_df, gen_df))

    out_dir = os.path.join(artifacts_dir, "test")
    os.makedirs(out_dir, exist_ok=True)

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

    runtime_stats = _compute_runtime_stats(runtime_df)

    rows = []
    for key, val in ac_stats.items():
        rows.append({"Metric": f"AC {key}", "Value": val})
    for key, val in dc_stats.items():
        rows.append({"Metric": f"DC {key}", "Value": val})
    for key, val in opf_extra.items():
        rows.append({"Metric": key, "Value": val})
    for key, val in runtime_stats.items():
        rows.append({"Metric": key, "Value": val})

    metrics_path = os.path.join(out_dir, f"{grid_name}_opf_ac_dc_metrics.csv")
    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    print(f"  Aggregated OPF AC/DC metrics saved to {metrics_path}")

    return True
