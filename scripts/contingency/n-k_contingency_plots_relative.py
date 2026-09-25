# %%
"""N-k contingency plots using relative active power balance residuals.

Buses with |pd_mw - pg_mw_target| > 0:
    relative residual [%] = 100 * active power residual / |pd_mw - pg_mw_target|

Buses with |pd_mw - pg_mw_target| == 0 (zero-injection):
    excluded from relative metrics; reported using absolute residuals [MW] only.

GENCO residuals come from predictions; DC residuals are merged with predictions
on (scenario, bus) to obtain pd_mw and pg_mw_target for the same denominator.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.lines import Line2D

# %%
_parser = argparse.ArgumentParser()
_parser.add_argument("--mlflow-dir", type=Path, required=True)
_parser.add_argument("--output-dir", type=Path, required=True)
_args = _parser.parse_args()
MLFLOW_RUN_DIRS = _args.mlflow_dir
FIG_DIR = _args.output_dir
CASE_PREFIX = "Texas2k_case1_2016summerpeak"

GENCO_COL = "GENCO_relative_active_residuals [-]"
DC_COL = "DC_relative_active_residuals [-]"
GENCO_ABS_COL = "GENCO_active_residuals [MW]"
DC_ABS_COL = "DC_active_residuals [MW]"
GENCO_ZERO_INJ_ABS_COL = "GENCO_zero_inj_active_residuals [MW]"
DC_ZERO_INJ_ABS_COL = "DC_zero_inj_active_residuals [MW]"
ZERO_INJ_ABS_YLABEL = "Absolute power balance residuals (MW)\n(zero net. inj. buses)"

# Match loading_error_boxplot_by_true_loading_Texas colors.
GENCO_COLOR = "tab:blue"
DC_COLOR = "tab:orange"
BAR_ALPHA = 0.6
FIG_DIR.mkdir(parents=True, exist_ok=True)

FONTSIZE = 24
THRESHOLD_FONTSIZE = 20


def _apply_font(size: int) -> None:
    plt.rcParams.update(
        {
            "font.size": size,
            "axes.titlesize": size,
            "axes.labelsize": size,
            "xtick.labelsize": size,
            "ytick.labelsize": size,
            "legend.fontsize": size,
            "figure.autolayout": False,
        }
    )


_apply_font(FONTSIZE)

# %%
def split_relative_and_zero_inj_abs(
    residual: np.ndarray,
    pd_mw: np.ndarray,
    pg_mw_target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    abs_net = np.abs(pd_mw - pg_mw_target)
    zero_inj_mask = abs_net == 0
    relative_pct = 100.0 * residual[~zero_inj_mask] / abs_net[~zero_inj_mask]
    zero_inj_abs = np.abs(residual[zero_inj_mask])
    return relative_pct, zero_inj_abs


def load_run_relative(run: Path) -> dict | None:
    try:
        with open(run / "meta.yaml", "r") as f:
            k_val = int(yaml.safe_load(f)["run_name"].split("_")[1])

        preds_path = run / "predictions.parquet"
        dc_path = run / "dc_bus_residuals.parquet"
        if not preds_path.is_file():
            base = run / "artifacts/test" / CASE_PREFIX
            preds_path = base.with_name(base.name + "_predictions.parquet")
            dc_path = base.with_name(base.name + "_dc_bus_residuals.parquet")

        preds = pd.read_parquet(preds_path)
        dc_bus = pd.read_parquet(dc_path)

        genco_rel, genco_zero_inj_abs = split_relative_and_zero_inj_abs(
            preds["active res. (MW)"].to_numpy(),
            preds["pd_mw"].to_numpy(),
            preds["pg_mw_target"].to_numpy(),
        )
        genco_abs = np.abs(preds["active res. (MW)"].to_numpy())

        merged = dc_bus.merge(
            preds[["scenario", "bus", "pd_mw", "pg_mw_target"]],
            on=["scenario", "bus"],
        )
        dc_rel, dc_zero_inj_abs = split_relative_and_zero_inj_abs(
            merged["DC active res. (MW)"].to_numpy(),
            merged["pd_mw"].to_numpy(),
            merged["pg_mw_target"].to_numpy(),
        )
        dc_abs = np.abs(merged["DC active res. (MW)"].to_numpy())

        return {
            "k": k_val,
            GENCO_COL: genco_rel,
            DC_COL: dc_rel,
            GENCO_ABS_COL: genco_abs,
            DC_ABS_COL: dc_abs,
            GENCO_ZERO_INJ_ABS_COL: genco_zero_inj_abs,
            DC_ZERO_INJ_ABS_COL: dc_zero_inj_abs,
        }
    except Exception as exc:
        print(f"Skipping {run.name}: {exc}")
        return None


# %%
k_vals = []
genco_relative = []
dc_relative = []
genco_abs = []
dc_abs = []
genco_zero_inj_abs = []
dc_zero_inj_abs = []

for run in MLFLOW_RUN_DIRS.iterdir():
    if not run.is_dir():
        continue

    result = load_run_relative(run)
    if result is None:
        continue

    k_vals.append(result["k"])
    genco_relative.append(result[GENCO_COL])
    dc_relative.append(result[DC_COL])
    genco_abs.append(result[GENCO_ABS_COL])
    dc_abs.append(result[DC_ABS_COL])
    genco_zero_inj_abs.append(result[GENCO_ZERO_INJ_ABS_COL])
    dc_zero_inj_abs.append(result[DC_ZERO_INJ_ABS_COL])

df_plotting = pd.DataFrame(
    {
        "k": k_vals,
        GENCO_COL: genco_relative,
        DC_COL: dc_relative,
        GENCO_ABS_COL: genco_abs,
        DC_ABS_COL: dc_abs,
        GENCO_ZERO_INJ_ABS_COL: genco_zero_inj_abs,
        DC_ZERO_INJ_ABS_COL: dc_zero_inj_abs,
    }
)

# %%

df_plotting["k"] = pd.to_numeric(df_plotting["k"], errors="coerce")
df_plotting = df_plotting.dropna(subset=["k"])
df3 = df_plotting.sort_values("k").set_index("k")

k_values = df3.index.to_numpy()


def flatten(series):
    values = []
    offsets = [0]

    for arr in series:
        arr = np.asarray(arr, dtype=np.float32)
        values.append(arr)
        offsets.append(offsets[-1] + len(arr))

    return np.concatenate(values), np.array(offsets, dtype=np.int64)


MAX_FLIERS_PER_BOX = 100  # subsample outliers for faster plotting / smaller files


def build_bxp(vals, offsets, max_fliers: int = MAX_FLIERS_PER_BOX):
    stats = []
    rng = np.random.default_rng(0)

    for i in range(len(offsets) - 1):
        arr = vals[offsets[i] : offsets[i + 1]]

        q1, med, q3 = np.percentile(arr, [25, 50, 75])
        iqr = q3 - q1

        low_fence = q1 - 1.5 * iqr
        high_fence = q3 + 1.5 * iqr

        whislo = arr[arr >= low_fence].min() if np.any(arr >= low_fence) else arr.min()
        whishi = arr[arr <= high_fence].max() if np.any(arr <= high_fence) else arr.max()
        fliers = arr[(arr < low_fence) | (arr > high_fence)]
        if fliers.size > max_fliers:
            fliers = rng.choice(fliers, size=max_fliers, replace=False)

        stats.append(
            {
                "med": med,
                "q1": q1,
                "q3": q3,
                "whislo": whislo,
                "whishi": whishi,
                "fliers": fliers,
            }
        )

    return stats


def compute_p999(vals, offsets):
    p = []
    for i in range(len(offsets) - 1):
        arr = vals[offsets[i] : offsets[i + 1]]
        p.append(np.percentile(arr, 99.9))
    return np.array(p)


def upper_outlier_pct(vals, offsets):
    pcts = []
    for i in range(len(offsets) - 1):
        arr = vals[offsets[i] : offsets[i + 1]]
        q1, _, q3 = np.percentile(arr, [25, 50, 75])
        high_fence = q3 + 1.5 * (q3 - q1)
        pcts.append(100.0 * np.mean(arr > high_fence))
    return pcts


def group_mean(vals, offsets):
    means = []
    for i in range(len(offsets) - 1):
        arr = vals[offsets[i] : offsets[i + 1]]
        means.append(np.mean(arr))
    return means


def group_median(vals, offsets):
    medians = []
    for i in range(len(offsets) - 1):
        arr = vals[offsets[i] : offsets[i + 1]]
        medians.append(np.median(arr))
    return medians


vals1, off1 = flatten(df3[GENCO_COL])
vals2, off2 = flatten(df3[DC_COL])
vals1_zero_inj, off1_zero_inj = flatten(df3[GENCO_ZERO_INJ_ABS_COL])
vals2_zero_inj, off2_zero_inj = flatten(df3[DC_ZERO_INJ_ABS_COL])
stats1 = build_bxp(vals1, off1)
stats2 = build_bxp(vals2, off2)
stats1_zero_inj = build_bxp(vals1_zero_inj, off1_zero_inj)
stats2_zero_inj = build_bxp(vals2_zero_inj, off2_zero_inj)
p1 = compute_p999(vals1, off1)
p2 = compute_p999(vals2, off2)

# %%
# Boxplot across k
fig, ax = plt.subplots(figsize=(14, 8))
x = np.arange(len(k_values), dtype=np.float32)

bp1 = ax.bxp(
    stats1,
    positions=x - 0.175,
    widths=0.35,
    patch_artist=True,
    showfliers=True,
)
bp2 = ax.bxp(
    stats2,
    positions=x + 0.175,
    widths=0.35,
    patch_artist=True,
    showfliers=True,
)

for b in bp1["boxes"]:
    b.set(facecolor=GENCO_COLOR, alpha=BAR_ALPHA, linewidth=0.8)
for b in bp2["boxes"]:
    b.set(facecolor=DC_COLOR, alpha=BAR_ALPHA, linewidth=0.8)
for m in bp1["medians"]:
    m.set(color=GENCO_COLOR, linewidth=2)
for m in bp2["medians"]:
    m.set(color=DC_COLOR, linewidth=2)

ax.set_xticks(x)
ax.set_xticklabels(k_values)
ax.set_yscale("log")
ax.set_xlabel("Number of components dropped $k$ [-]")
ax.set_ylabel("Relative power balance residuals (%)\n(zero net. inj. buses excluded)")
ax.set_yticks([0.01, 0.1, 1, 10, 100, 1000])
ax.set_yticklabels(["0.01%", "0.1%", "1%", "10%", "100%", "1000%"])
ax.grid(axis="y", linestyle="--", alpha=0.6)
ax.legend(
    handles=[
        Line2D([0], [0], color=GENCO_COLOR, lw=4, label="GENCO"),
        Line2D([0], [0], color=DC_COLOR, lw=4, label="DC-PF"),
    ],
    loc="upper right",
)
ax.set_ylim(1e-2, 1e5)
fig.tight_layout()
# PNG only: vector PDF with fliers is hundreds of MB.
fig.savefig(FIG_DIR / "boxplot_relative_residuals.png", dpi=600, bbox_inches="tight")
plt.close(fig)

# %%
# Boxplot across k — zero-injection buses ($|P_d-P_g|=0$), absolute residuals
fig, ax = plt.subplots(figsize=(14, 8))
x = np.arange(len(k_values), dtype=np.float32)

bp1 = ax.bxp(
    stats1_zero_inj,
    positions=x - 0.175,
    widths=0.35,
    patch_artist=True,
    showfliers=True,
)
bp2 = ax.bxp(
    stats2_zero_inj,
    positions=x + 0.175,
    widths=0.35,
    patch_artist=True,
    showfliers=True,
)

for b in bp1["boxes"]:
    b.set(facecolor=GENCO_COLOR, alpha=BAR_ALPHA, linewidth=0.8)
for b in bp2["boxes"]:
    b.set(facecolor=DC_COLOR, alpha=BAR_ALPHA, linewidth=0.8)
for m in bp1["medians"]:
    m.set(color=GENCO_COLOR, linewidth=2)
for m in bp2["medians"]:
    m.set(color=DC_COLOR, linewidth=2)

ax.set_xticks(x)
ax.set_xticklabels(k_values)
ax.set_yscale("log")
ax.set_xlabel("Number of components dropped $k$ [-]")
ax.set_ylabel(ZERO_INJ_ABS_YLABEL)
ax.grid(axis="y", linestyle="--", alpha=0.6)
ax.legend(
    handles=[
        Line2D([0], [0], color=GENCO_COLOR, lw=4, label="GENCO"),
        Line2D([0], [0], color=DC_COLOR, lw=4, label="DC-PF"),
    ],
    loc="upper right",
)
fig.tight_layout()
fig.savefig(FIG_DIR / "boxplot_zero_inj_absolute_residuals.png", dpi=600, bbox_inches="tight")
plt.close(fig)

# %%
# Share below relative thresholds at k = 10 (zero-injection buses excluded)
_apply_font(THRESHOLD_FONTSIZE)
x = np.asarray(df3.loc[10, GENCO_COL])
x2 = np.asarray(df3.loc[10, DC_COL])

thresholds = np.array([0.1, 1.0, 10])
p_x = np.array([(x <= t).mean() * 100 for t in thresholds])
p_x2 = np.array([(x2 <= t).mean() * 100 for t in thresholds])

x_pos = np.arange(len(thresholds))
width = 0.35
fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(
    x_pos - width / 2,
    p_x,
    width,
    label="GENCO",
    color=GENCO_COLOR,
    alpha=BAR_ALPHA,
)
ax.bar(
    x_pos + width / 2,
    p_x2,
    width,
    label="DC-PF",
    color=DC_COLOR,
    alpha=BAR_ALPHA,
)
ax.set_xticks(x_pos)
ax.set_xticklabels([f"{t:g}" for t in thresholds])
ax.set_xlabel("Relative residual threshold (%)")
ax.set_ylabel("Samples below threshold (%)\n(zero net. inj. buses excluded)")
ax.set_ylim(0, 100)
ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "threshold_share_relative_k10.pdf", dpi=400, bbox_inches="tight")
plt.close("all")

print(f"1%: {p_x[1]:.2f}% for GENCO and {p_x2[1]:.2f}% for DC-PF")

# %%
# Share below absolute thresholds at k = 10 (zero-injection buses only)
x_zero_abs = np.asarray(df3.loc[10, GENCO_ZERO_INJ_ABS_COL])
x2_zero_abs = np.asarray(df3.loc[10, DC_ZERO_INJ_ABS_COL])

thresholds_abs = np.array([0.1, 1, 10])
p_zero_abs = np.array([(x_zero_abs <= t).mean() * 100 for t in thresholds_abs])
p2_zero_abs = np.array([(x2_zero_abs <= t).mean() * 100 for t in thresholds_abs])

x_pos = np.arange(len(thresholds_abs))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(
    x_pos - width / 2,
    p_zero_abs,
    width,
    label="GENCO",
    color=GENCO_COLOR,
    alpha=BAR_ALPHA,
)
ax.bar(
    x_pos + width / 2,
    p2_zero_abs,
    width,
    label="DC-PF",
    color=DC_COLOR,
    alpha=BAR_ALPHA,
)

ax.set_xticks(x_pos)
ax.set_xticklabels([f"{t}" for t in thresholds_abs])
ax.set_xlabel("Residual threshold (MW)")
ax.set_ylabel("Samples below threshold (%)\n(zero net. inj. buses only)")

ax.set_ylim(0, 100)
ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "threshold_share_absolute_k10_zero_inj.pdf", dpi=400, bbox_inches="tight")
plt.close("all")

print(
    "Zero-injection buses below 1 MW: "
    f"{np.mean(x_zero_abs <= 1) * 100:.2f}% for GENCO and "
    f"{np.mean(x2_zero_abs <= 1) * 100:.2f}% for DC-PF"
)

# %%
# Share below absolute thresholds at k = 10 (all buses)
x_abs = np.asarray(df3.loc[10, GENCO_ABS_COL])
x2_abs = np.asarray(df3.loc[10, DC_ABS_COL])

thresholds_abs = np.array([0.1, 1, 10])
p_abs = np.array([(x_abs <= t).mean() * 100 for t in thresholds_abs])
p2_abs = np.array([(x2_abs <= t).mean() * 100 for t in thresholds_abs])

x_pos = np.arange(len(thresholds_abs))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(
    x_pos - width / 2,
    p_abs,
    width,
    label="GENCO",
    color=GENCO_COLOR,
    alpha=BAR_ALPHA,
)
ax.bar(
    x_pos + width / 2,
    p2_abs,
    width,
    label="DC-PF",
    color=DC_COLOR,
    alpha=BAR_ALPHA,
)

ax.set_xticks(x_pos)
ax.set_xticklabels([f"{t}" for t in thresholds_abs])
ax.set_xlabel("Residual threshold (MW)")
ax.set_ylabel("Samples below threshold (%)\n(all buses)")

ax.set_ylim(0, 100)
ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "threshold_share_absolute_k10.pdf", dpi=400, bbox_inches="tight")
plt.close("all")

print(
    "All buses below 1 MW: "
    f"{np.mean(x_abs <= 1) * 100:.2f}% for GENCO and "
    f"{np.mean(x2_abs <= 1) * 100:.2f}% for DC-PF"
)

# %%
# Share below 1% threshold vs k
_apply_font(THRESHOLD_FONTSIZE)
THRESHOLD_1PCT = 1.0  # residuals are stored in percent


def share_below_threshold_per_k(vals, offsets, threshold):
    shares = []
    for i in range(len(offsets) - 1):
        arr = vals[offsets[i] : offsets[i + 1]]
        shares.append(100.0 * np.mean(arr <= threshold))
    return np.array(shares)


genco_below_1pct = share_below_threshold_per_k(vals1, off1, THRESHOLD_1PCT)
dc_below_1pct = share_below_threshold_per_k(vals2, off2, THRESHOLD_1PCT)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(
    k_values,
    genco_below_1pct,
    "o-",
    color=GENCO_COLOR,
    label="GENCO",
    linewidth=2,
    markersize=8,
)
ax.plot(
    k_values,
    dc_below_1pct,
    "o-",
    color=DC_COLOR,
    label="DC-PF",
    linewidth=2,
    markersize=8,
)
ax.set_xlabel("Number of components dropped $k$ [-]")
ax.set_ylabel("Samples below 1% threshold (%)\n(zero net. inj. buses excluded)")
ax.set_xticks(k_values)
ax.set_ylim(0, 100)
ax.grid(axis="y", linestyle="--", alpha=0.6)
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "below_1pct_threshold_vs_k.pdf", dpi=400, bbox_inches="tight")
plt.close("all")

# %%
reduced_stats_df = pd.DataFrame(
    {
        "k": k_values,
        "GENCO_mean_relative_residual [-]": group_mean(vals1, off1),
        "DC-PF_mean_relative_residual [-]": group_mean(vals2, off2),
        "GENCO_median_relative_residual [-]": group_median(vals1, off1),
        "DC-PF_median_relative_residual [-]": group_median(vals2, off2),
        "GENCO_upper_outliers [%]": upper_outlier_pct(vals1, off1),
        "DC-PF_upper_outliers [%]": upper_outlier_pct(vals2, off2),
        "GENCO_p99.9 [-]": p1,
        "DC-PF_p99.9 [-]": p2,
    }
).set_index("k")

print(reduced_stats_df)
reduced_stats_df.to_csv(FIG_DIR / "relative_residual_summary_by_k.csv")

for k in [1, 10, 20]:
    ratio = (
        reduced_stats_df.loc[k, "DC-PF_median_relative_residual [-]"]
        / reduced_stats_df.loc[k, "GENCO_median_relative_residual [-]"]
    )
    print(f"Median residual ratio at k={k} (DC-PF / GENCO): {ratio:.6f}")

# %%
