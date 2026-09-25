# %%
# Import analysis dependencies and shared utility functions.
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from sklearn.metrics import r2_score


from contingency_utils import (
    compute_branch_powers_vectorized,
    plot_mass_correlation_density,
    plot_mass_correlation_density_voltage,
)

# %% [markdown]
# ## Configuration

# %%
# Define input artifacts and evaluation hyperparameters used throughout the run.
_parser = argparse.ArgumentParser()
_parser.add_argument("--light-dir", type=Path, required=True)
_parser.add_argument("--output-dir", type=Path, required=True)
_args = _parser.parse_args()
FILTERED_PREDS_LIGHT_PATH = _args.light_dir / "filtered_preds_light.parquet"
FILTERED_BUS_LIGHT_PATH = _args.light_dir / "filtered_bus_data_light.parquet"
FILTERED_BRANCH_LIGHT_PATH = _args.light_dir / "filtered_branch_data_light.parquet"
FIG_DIR = _args.output_dir
FIG_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_TAG = "Texas"
MODEL_LABEL = "GENCO base"
BETA = 10.0
SN_MVA = 100.0
FONTSIZE = 24

plt.rcParams.update(
    {
        "font.size": FONTSIZE,
        "axes.titlesize": FONTSIZE,
        "axes.labelsize": FONTSIZE,
        "xtick.labelsize": FONTSIZE,
        "ytick.labelsize": FONTSIZE,
        "legend.fontsize": FONTSIZE,
        "figure.autolayout": False,
    }
)

# Keep this enabled for the publication pipeline. A separate comparison already
# showed this correction has negligible impact on GENCO metrics for this dataset.
APPLY_PV_REF_CORRECTION = False


# %% [markdown]
# ## Load Prebuilt Light Data

# %%
# Load minimal aligned inputs and build a single bus-level working table.
missing_light_inputs = [
    str(p)
    for p in (
        FILTERED_PREDS_LIGHT_PATH,
        FILTERED_BUS_LIGHT_PATH,
        FILTERED_BRANCH_LIGHT_PATH,
    )
    if not p.exists()
]
# Fail fast so results are always generated from the intended light inputs.
if missing_light_inputs:
    missing_list = ", ".join(missing_light_inputs)
    raise FileNotFoundError(
        f"Missing light input files: {missing_list}. "
        "Run create_minimal_contingency_data.py first."
    )

filtered_preds = pd.read_parquet(FILTERED_PREDS_LIGHT_PATH)
filtered_bus_data = pd.read_parquet(FILTERED_BUS_LIGHT_PATH)
filtered_branch_data = pd.read_parquet(FILTERED_BRANCH_LIGHT_PATH)

# Merge once on (scenario, bus) so all downstream computations share aligned rows.
pf_node = filtered_preds.merge(filtered_bus_data, on=["scenario", "bus"], how="left")

# %% [markdown]
# ## Build Corrected GENCO and DC Voltage/Angle States

# %%
# Build GENCO/DC state variants used to compute comparable branch flows.
pf_node["Vm_pred_corrected"] = pf_node["Vm_pred"].astype(np.float64)
pf_node["Va_pred_corrected"] = pf_node["Va_pred"]
pf_node["Vm_dc_corrected"] = np.ones(len(pf_node), dtype=np.float64)
pf_node["Va_dc_corrected"] = pf_node["Va_dc"]

if APPLY_PV_REF_CORRECTION:
    # Enforce generator voltage setpoints at PV buses and reference angles at REF buses.
    pv_mask = pf_node["PV"] == 1
    ref_mask = pf_node["REF"] == 1
    pf_node.loc[pv_mask, "Vm_pred_corrected"] = pf_node.loc[pv_mask, "Vm"]
    pf_node.loc[ref_mask, "Va_pred_corrected"] = pf_node.loc[ref_mask, "Va"]
    pf_node.loc[pv_mask, "Vm_dc_corrected"] = pf_node.loc[pv_mask, "Vm"]
    pf_node.loc[ref_mask, "Va_dc_corrected"] = pf_node.loc[ref_mask, "Va"]

# Branch-power function expects degree inputs.
pf_node["Va_pred_corrected"] = np.rad2deg(pf_node["Va_pred_corrected"])
# `Va` comes from bus data and is used as-is for ground-truth branch flows.

# %% [markdown]
# ## Compute Line Loadings (AC Ground Truth, GENCO, DC)

# %%
# Compute per-branch loading for AC ground truth, GENCO prediction, and DC baseline.
rated_branch_data = filtered_branch_data[filtered_branch_data["rate_a"] > 0].copy()
# Avoid divide-by-zero and non-rated branches in loading normalization.
rate_a = rated_branch_data["rate_a"].to_numpy(dtype=np.float64)

loadings_map = {}
for flag in ("gt", "pred", "dc"):
    pf, qf, pt, qt = compute_branch_powers_vectorized(
        rated_branch_data,
        pf_node,
        sn_mva=SN_MVA,
        flag=flag,
    )
    s_from = np.hypot(pf, qf)
    s_to = np.hypot(pt, qt)
    loadings_map[flag] = (np.maximum(s_from, s_to) / rate_a).flatten()

loadings = loadings_map["gt"]
loadings_genco = loadings_map["pred"]
loadings_dc = loadings_map["dc"]

print("number of scenarios: ", len(pf_node["scenario"].unique()))
print("number of branches: ", len(filtered_branch_data))
print("number of buses: ", len(filtered_bus_data))


# %%
def plot_mass_correlation_density_voltage(
    pf_node,
    prediction_dir,
    label_plot,
    x_min=0.85,
    y_min=0.85,
    x_max=1.15,
    y_max=1.15,
    vm_nominal=1.0,
    vm_dev_threshold=0.10,
):
    """
    TODO docstrings
    TODO refactor if we pass by parameters a few more plot deets we can use plot_mass_correlation_density for both

    """
    # Get the global min and max for color scaling (avoid log(0) by setting min to at least 1)
    vmin = 1
    bin_width = 0.001  # consistent bin width for both plots

    # Generate consistent bins
    x_bins = np.arange(x_min, x_max + bin_width, bin_width)
    y_bins = np.arange(y_min, y_max + bin_width, bin_width)

    # estimate vmax on mean count of elements across bins
    counts, _, _ = np.histogram2d(
        pf_node["Vm"],
        pf_node["Vm_pred_corrected"],
        bins=[x_bins, y_bins],
    )

    counts[counts == 0] = np.nan
    means = np.nanmean(counts)
    std = np.nanstd(counts)
    vmax = means + 3 * std

    # r2
    r2 = r2_score(pf_node["Vm"], pf_node["Vm_pred_corrected"])

    # Create figure with shared x-axis
    fig, ax1 = plt.subplots(figsize=(9, 7))

    # --- GENCO Mass Correlation ---
    h1 = ax1.hist2d(
        pf_node["Vm"],
        pf_node["Vm_pred_corrected"],
        bins=[x_bins, y_bins],
        norm=LogNorm(vmin=vmin, vmax=vmax),
        cmap="inferno",
    )
    vm_lower_limit = vm_nominal - vm_dev_threshold
    vm_upper_limit = vm_nominal + vm_dev_threshold
    ax1.axvline(vm_lower_limit, color="black", linestyle=":", linewidth=2.0)
    ax1.axhline(vm_lower_limit, color="black", linestyle=":", linewidth=2.0)
    ax1.axvline(vm_upper_limit, color="black", linestyle=":", linewidth=2.0)
    ax1.axhline(vm_upper_limit, color="black", linestyle=":", linewidth=2.0)

    ax1.plot([0, 5], [0, 5], "k--", linewidth=0.5)
    ax1.set_xlabel("True voltage magnitude [p.u.]")
    ax1.set_ylabel("Predicted voltage magnitude [p.u.]")
    ax1.set_title(label_plot)
    ax1.text(
        0.5,
        0.95,
        rf"$R^2 = {r2:.5f}$",
        transform=ax1.transAxes,
        fontsize=FONTSIZE,
        weight="bold",
        ha="center",
        va="top",
    )

    # Colorbar
    cbar = fig.colorbar(h1[3], ax=ax1, pad=0.02)
    cbar.set_label(r"$\mathrm{Number\ of\ buses}$ [-]")
    cbar.ax.tick_params(labelsize=FONTSIZE)

    # Style adjustments
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.grid(True, linewidth=0.3)

    plt.tight_layout()
    plt.savefig(
        FIG_DIR / f"mass_correlation_density_voltage_{prediction_dir}.pdf",
        bbox_inches="tight",
    )
    plt.close()


def estimate_density_vmax(
    true_vals,
    predicted_vals,
    x_bins,
    y_bins,
):
    counts, _, _ = np.histogram2d(true_vals, predicted_vals, bins=[x_bins, y_bins])
    counts[counts == 0] = np.nan
    means = np.nanmean(counts)
    std = np.nanstd(counts)
    return means + 3 * std


def plot_mass_correlation_density(
    true_vals,
    predicted_vals,
    model_name,
    label_plot,
    x_max=2,
    y_max=3,
    vmax=None,
):
    """
    TODO docstring

    """
    # TODO check if these parameters need to be passed by func or default behavior
    vmin = 1
    x_min = 0
    y_min = 0
    bin_width = 0.01  # consistent bin width for both plots

    # Generate consistent bins
    x_bins = np.arange(x_min, x_max + bin_width, bin_width)
    y_bins = np.arange(y_min, y_max + bin_width, bin_width)

    if vmax is None:
        vmax = estimate_density_vmax(true_vals, predicted_vals, x_bins, y_bins)

    # r2
    r2 = r2_score(true_vals, predicted_vals)

    # Create figure with shared x-axis
    fig, ax1 = plt.subplots(figsize=(9, 7))

    # --- GENCO Mass Correlation ---
    h1 = ax1.hist2d(
        true_vals,
        predicted_vals,
        bins=[x_bins, y_bins],
        norm=LogNorm(vmin=vmin, vmax=vmax),
        cmap="inferno",
    )
    ax1.axvline(1, color="black", linestyle="--", linewidth=2.0)
    ax1.axhline(1, color="black", linestyle="--", linewidth=2.0)
    ax1.plot([0, 5], [0, 5], "k--", linewidth=0.5)
    ax1.set_xlabel("True loading [-]")
    ax1.set_ylabel("Predicted loading [-]")
    ax1.set_title(label_plot)
    ax1.text(
        x_max - 1.5,
        0.93,
        rf"$R^2 = {r2:.5f}$",
        transform=ax1.transAxes,
        fontsize=FONTSIZE,
        weight="bold",
    )

    # Colorbar
    cbar = fig.colorbar(h1[3], ax=ax1, pad=0.02)
    cbar.set_label(r"$\mathrm{Number\ of\ lines}$ [-]")
    cbar.ax.tick_params(labelsize=FONTSIZE)

    # Style adjustments
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.grid(True, linewidth=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / f"mass_correlation_density_{model_name}.pdf", bbox_inches="tight")
    plt.close()


def plot_mass_correlation_density_comparison(
    true_vals,
    predicted_vals_left,
    predicted_vals_right,
    left_label,
    right_label,
    output_name,
    x_max=2,
    y_max=3,
    vmax=None,
):
    """Plot two density comparisons side by side with shared axes and color scale."""
    vmin = 1
    x_min = 0
    y_min = 0
    bin_width = 0.01

    x_bins = np.arange(x_min, x_max + bin_width, bin_width)
    y_bins = np.arange(y_min, y_max + bin_width, bin_width)

    if vmax is None:
        vmax = max(
            estimate_density_vmax(true_vals, predicted_vals_left, x_bins, y_bins),
            estimate_density_vmax(true_vals, predicted_vals_right, x_bins, y_bins),
        )

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    hist = None

    for ax, predicted_vals, label in zip(
        axes,
        [predicted_vals_left, predicted_vals_right],
        [left_label, right_label],
    ):
        r2 = r2_score(true_vals, predicted_vals)
        hist = ax.hist2d(
            true_vals,
            predicted_vals,
            bins=[x_bins, y_bins],
            norm=LogNorm(vmin=vmin, vmax=vmax),
            cmap="inferno",
        )
        ax.axvline(1, color="black", linestyle="--", linewidth=2.0)
        ax.axhline(1, color="black", linestyle="--", linewidth=2.0)
        ax.plot([0, 5], [0, 5], "k--", linewidth=0.5)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("True loading [-]")
        ax.set_title(label)
        ax.grid(True, linewidth=0.3)
        ax.text(
            0.05,
            0.93,
            rf"$R^2 = {r2:.5f}$",
            transform=ax.transAxes,
            fontsize=FONTSIZE,
            weight="bold",
        )

    axes[0].set_ylabel("Predicted loading [-]")

    fig.subplots_adjust(right=0.88, wspace=0.12)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.72])
    cbar = fig.colorbar(hist[3], cax=cbar_ax)
    cbar.set_label(r"$\mathrm{Number\ of\ lines}$ [-]")
    cbar.ax.tick_params(labelsize=FONTSIZE)

    plt.savefig(FIG_DIR / f"mass_correlation_density_{output_name}.pdf", bbox_inches="tight")
    plt.close()
    
# %%
loading_x_bins = np.arange(0, 2 + 0.01, 0.01)
loading_y_bins = np.arange(0, 3 + 0.01, 0.01)
shared_loading_vmax = max(
    estimate_density_vmax(loadings, loadings_genco, loading_x_bins, loading_y_bins),
    estimate_density_vmax(loadings, loadings_dc, loading_x_bins, loading_y_bins),
)
# Larger fonts for the side-by-side loading density panel.
FONTSIZE = 28
plt.rcParams.update(
    {
        "font.size": FONTSIZE,
        "axes.titlesize": FONTSIZE,
        "axes.labelsize": FONTSIZE,
        "xtick.labelsize": FONTSIZE,
        "ytick.labelsize": FONTSIZE,
        "legend.fontsize": FONTSIZE,
    }
)
plot_mass_correlation_density_comparison(
    true_vals=loadings,
    predicted_vals_left=loadings_genco,
    predicted_vals_right=loadings_dc,
    left_label="GENCO base",
    right_label="DC-PF",
    output_name=f"{OUTPUT_TAG}_vs_dc",
    vmax=shared_loading_vmax,
)
FONTSIZE = 24
plt.rcParams.update(
    {
        "font.size": FONTSIZE,
        "axes.titlesize": FONTSIZE,
        "axes.labelsize": FONTSIZE,
        "xtick.labelsize": FONTSIZE,
        "ytick.labelsize": FONTSIZE,
        "legend.fontsize": FONTSIZE,
    }
)

# %%
# Repeat threshold selection/evaluation for voltage-violation detection (GENCO only).
plot_mass_correlation_density_voltage(pf_node, OUTPUT_TAG, MODEL_LABEL)


# %% 
overload_mask = loadings > 1.0
overload_mask_genco = loadings_genco > 1.0
overload_mask_dc = loadings_dc > 1.0
print(f"Share of branches overloaded: {overload_mask.mean()*100:.4f}%")
print(f"Share of branches overloaded in GENCO: {overload_mask_genco.mean()*100:.4f}%")
print(f"Share of branches overloaded in DC-PF: {overload_mask_dc.mean()*100:.4f}%")

print("Number of branches overloaded: ", overload_mask.sum())


# %%
# mae on loadings of overloaded branches
mae_genco = np.mean(np.abs(loadings_genco[overload_mask] - loadings[overload_mask]))
mae_dc = np.mean(np.abs(loadings_dc[overload_mask] - loadings[overload_mask]))
print(f"MAE on loadings of overloaded branches in GENCO: {mae_genco:.4f}")
print(f"MAE on loadings of overloaded branches in DC-PF: {mae_dc:.4f}")

# mae on loadings of non-overloaded branches
mae_genco_non_overload = np.mean(np.abs(loadings_genco[~overload_mask] - loadings[~overload_mask]))
mae_dc_non_overload = np.mean(np.abs(loadings_dc[~overload_mask] - loadings[~overload_mask]))
print(f"MAE on loadings of non-overloaded branches in GENCO: {mae_genco_non_overload:.4f}")
print(f"MAE on loadings of non-overloaded branches in DC-PF: {mae_dc_non_overload:.4f}")
# %%
# ratio of mae of overloaded branches to mae of non-overloaded branches
ratio_mae_genco = mae_genco / mae_genco_non_overload
ratio_mae_dc = mae_dc / mae_dc_non_overload
print(f"Ratio of MAE on loadings of overloaded branches to MAE on loadings of non-overloaded branches in GENCO: {ratio_mae_genco:.4f}")
print(f"Ratio of MAE on loadings of overloaded branches to MAE on loadings of non-overloaded branches in DC-PF: {ratio_mae_dc:.4f}")
# %%

loading_error_by_bin = pd.DataFrame(
    {
        "true_loading": loadings,
        "GENCO": np.abs(loadings_genco - loadings),
        "DC-PF": np.abs(loadings_dc - loadings),
    }
)

loading_bins = [0.0, 0.25, 0.5, 0.75, 1.0, 1.1, 1.25, 1.5, 2.0, np.inf]
loading_bin_labels = [
    "0–0.25",
    "0.25–0.5",
    "0.5–0.75",
    "0.75–1",
    "1–1.1",
    "1.1–1.25",
    "1.25–1.5",
    "1.5–2",
    ">2",
]
loading_error_by_bin["True loading bin"] = pd.cut(
    loading_error_by_bin["true_loading"],
    bins=loading_bins,
    labels=loading_bin_labels,
    include_lowest=True,
    right=True,
)

present_loading_bin_labels = [
    label
    for label in loading_bin_labels
    if (loading_error_by_bin["True loading bin"] == label).any()
]
loading_bin_counts = (
    loading_error_by_bin["True loading bin"]
    .value_counts()
    .reindex(present_loading_bin_labels)
)


def _compact_n(n: int) -> str:
    if n < 1000:
        return str(n)
    return f"{n:.1e}".replace("e+0", "e").replace("e+", "e")


BOX_FS = 30
fig, ax = plt.subplots(figsize=(16, 8.5))
positions = np.arange(len(present_loading_bin_labels))
box_width = 0.32
model_box_specs = [
    ("GENCO", -box_width / 2, "tab:blue"),
    ("DC-PF", box_width / 2, "tab:orange"),
]

for model_name, offset, color in model_box_specs:
    model_errors_by_bin = [
        loading_error_by_bin.loc[
            loading_error_by_bin["True loading bin"] == bin_label,
            model_name,
        ].to_numpy()
        for bin_label in present_loading_bin_labels
    ]
    boxplot = ax.boxplot(
        model_errors_by_bin,
        positions=positions + offset,
        widths=box_width,
        patch_artist=True,
        showfliers=False,
    )
    for box in boxplot["boxes"]:
        box.set(facecolor=color, alpha=0.6)
    for median in boxplot["medians"]:
        median.set(color="black", linewidth=1.2)

ax.axvline(3.5, color="black", linestyle="--", linewidth=1.0, alpha=0.8)
ax.set_yscale("log")
ax.set_ylim(1e-5, 10)
ax.set_xticks(positions)
ax.set_xticklabels(
    [
        f"{lab} (n={_compact_n(int(loading_bin_counts.loc[lab]))})"
        for lab in present_loading_bin_labels
    ],
    fontsize=BOX_FS,
    rotation=70,
    ha="right",
)
ax.tick_params(axis="y", labelsize=BOX_FS)
ax.set_xlabel("True loading [-]", labelpad=6, fontsize=BOX_FS)
ax.set_ylabel("Absolute loading error [-]", labelpad=10, fontsize=BOX_FS)
ax.grid(axis="y", alpha=0.3)

ax.legend(
    handles=[
        plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.6)
        for _, _, color in model_box_specs
    ],
    labels=[model_name for model_name, _, _ in model_box_specs],
    frameon=True,
    loc="upper left",
    bbox_to_anchor=(1.01, 1.0),
    borderaxespad=0.0,
    fontsize=BOX_FS,
)
fig.subplots_adjust(left=0.09, right=0.88, top=0.97, bottom=0.32)
fig.savefig(
    FIG_DIR / f"loading_error_boxplot_by_true_loading_{OUTPUT_TAG}.pdf",
    bbox_inches="tight",
    pad_inches=0.3,
)
plt.close()
# %%

undervoltage_limit = 0.9
overvoltage_limit = 1.1

voltage_violation_mask = (pf_node["Vm"] < undervoltage_limit) | (pf_node["Vm"] > overvoltage_limit)
voltage_violation_mask_genco = (
    (pf_node["Vm_pred_corrected"] < undervoltage_limit)
    | (pf_node["Vm_pred_corrected"] > overvoltage_limit)
)
voltage_violation_mask_dc = (
    (pf_node["Vm_dc_corrected"] < undervoltage_limit)
    | (pf_node["Vm_dc_corrected"] > overvoltage_limit)
)

print(f"Share of buses with voltage violations: {voltage_violation_mask.mean()*100:.4f}%")
print(f"Share of buses with voltage violations in GENCO: {voltage_violation_mask_genco.mean()*100:.4f}%")
print(f"Share of buses with voltage violations in DC-PF: {voltage_violation_mask_dc.mean()*100:.4f}%")

print("Number of buses with voltage violations: ", voltage_violation_mask.sum())

# %%
# mae on voltage magnitudes of buses with voltage violations
mae_vm_genco = np.mean(
    np.abs(
        pf_node.loc[voltage_violation_mask, "Vm_pred_corrected"]
        - pf_node.loc[voltage_violation_mask, "Vm"]
    )
)
mae_vm_dc = np.mean(
    np.abs(
        pf_node.loc[voltage_violation_mask, "Vm_dc_corrected"]
        - pf_node.loc[voltage_violation_mask, "Vm"]
    )
)
print(f"MAE on voltage magnitudes of violating buses in GENCO: {mae_vm_genco:.4f}")
print(f"MAE on voltage magnitudes of violating buses in DC-PF: {mae_vm_dc:.4f}")

# mae on voltage magnitudes of buses without voltage violations
mae_vm_genco_non_violation = np.mean(
    np.abs(
        pf_node.loc[~voltage_violation_mask, "Vm_pred_corrected"]
        - pf_node.loc[~voltage_violation_mask, "Vm"]
    )
)
mae_vm_dc_non_violation = np.mean(
    np.abs(
        pf_node.loc[~voltage_violation_mask, "Vm_dc_corrected"]
        - pf_node.loc[~voltage_violation_mask, "Vm"]
    )
)
print(f"MAE on voltage magnitudes of non-violating buses in GENCO: {mae_vm_genco_non_violation:.4f}")
print(f"MAE on voltage magnitudes of non-violating buses in DC-PF: {mae_vm_dc_non_violation:.4f}")
# %%
# ratio of mae of voltage-violating buses to mae of non-violating buses
ratio_mae_vm_genco = mae_vm_genco / mae_vm_genco_non_violation
ratio_mae_vm_dc = mae_vm_dc / mae_vm_dc_non_violation
print(f"Ratio of MAE on voltage magnitudes of violating buses to MAE on voltage magnitudes of non-violating buses in GENCO: {ratio_mae_vm_genco:.4f}")
print(f"Ratio of MAE on voltage magnitudes of violating buses to MAE on voltage magnitudes of non-violating buses in DC-PF: {ratio_mae_vm_dc:.4f}")
# %%

voltage_error_by_bin = pd.DataFrame(
    {
        "true_voltage": pf_node["Vm"],
        "GENCO": np.abs(pf_node["Vm_pred_corrected"] - pf_node["Vm"]),
    }
)

voltage_bins = [0.0, 0.9, 0.95, 1.0, 1.05, 1.1, np.inf]
voltage_bin_labels = [
    "≤0.9",
    "0.9–0.95",
    "0.95–1",
    "1–1.05",
    "1.05–1.1",
    ">1.1",
]
voltage_error_by_bin["True voltage bin"] = pd.cut(
    voltage_error_by_bin["true_voltage"],
    bins=voltage_bins,
    labels=voltage_bin_labels,
    include_lowest=True,
    right=True,
)

present_voltage_bin_labels = [
    label
    for label in voltage_bin_labels
    if (voltage_error_by_bin["True voltage bin"] == label).any()
]
voltage_bin_counts = (
    voltage_error_by_bin["True voltage bin"]
    .value_counts()
    .reindex(present_voltage_bin_labels)
)

BOX_FS = 26
fig, ax = plt.subplots(figsize=(13, 8))
positions = np.arange(len(present_voltage_bin_labels))
voltage_errors_by_bin = [
    voltage_error_by_bin.loc[
        voltage_error_by_bin["True voltage bin"] == bin_label,
        "GENCO",
    ].to_numpy()
    for bin_label in present_voltage_bin_labels
]
boxplot = ax.boxplot(
    voltage_errors_by_bin,
    positions=positions,
    widths=0.45,
    patch_artist=True,
    showfliers=False,
)
for box in boxplot["boxes"]:
    box.set(facecolor="tab:blue", alpha=0.6)
for median in boxplot["medians"]:
    median.set(color="black", linewidth=1.2)

ax.axvline(0.5, color="black", linestyle="--", linewidth=1.0, alpha=0.8)
ax.axvline(4.5, color="black", linestyle="--", linewidth=1.0, alpha=0.8)
ax.set_xlabel("True voltage mag. [p.u.]", labelpad=14, fontsize=BOX_FS)
ax.set_ylabel("Absolute voltage mag. error [p.u.]", labelpad=10, fontsize=BOX_FS)
ax.set_xticks(positions)
ax.set_xticklabels(
    [
        f"{bin_label}\nn={_compact_n(int(voltage_bin_counts.loc[bin_label]))}"
        for bin_label in present_voltage_bin_labels
    ],
    fontsize=BOX_FS,
)
ax.tick_params(axis="both", labelsize=BOX_FS)
ax.set_yscale("log")
ax.set_ylim(1e-9, 10)
ax.grid(axis="y", alpha=0.3)
fig.subplots_adjust(left=0.14, right=0.995, top=0.97, bottom=0.22)
fig.savefig(
    FIG_DIR / f"voltage_error_boxplot_by_true_voltage_{OUTPUT_TAG}.pdf",
    bbox_inches="tight",
    pad_inches=0.3,
)
plt.close()
# %%