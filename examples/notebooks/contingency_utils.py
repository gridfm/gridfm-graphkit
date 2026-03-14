import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import pearsonr
import seaborn as sns
import numpy as np
import copy
import pandas as pd
from typing import Tuple, Union

# Power flow utils
def compute_branch_powers_vectorized(
    branch_df: pd.DataFrame,
    bus_df: pd.DataFrame,
    sn_mva: float,
    flag: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute branch power flows for all branches in a vectorized fashion.

    Args:
        branch_df: DataFrame with branch data including Yff, Yft, Ytf, Ytt admittances
        bus_df: DataFrame with bus data including Vm and Va (or Va_dc for DC mode)
        dc: If True, use DC power flow (Va_dc, Vm=1.0), else use AC (Va, Vm)
        sn_mva: System base power in MVA used to scale complex power results

    Returns:
        Tuple of (pf, qf, pt, qt) power flow arrays in MW/MVAR
    """
    scenarios = branch_df["scenario"].to_numpy(dtype=int)
    from_bus = branch_df["from_bus"].to_numpy(dtype=int)
    to_bus = branch_df["to_bus"].to_numpy(dtype=int)

    idx_from = pd.MultiIndex.from_arrays(
        [scenarios, from_bus],
        names=["scenario", "bus"],
    )
    idx_to = pd.MultiIndex.from_arrays([scenarios, to_bus], names=["scenario", "bus"])

    bus_df_indexed = bus_df.set_index(["scenario", "bus"]).copy()
    if flag=='gt':
        Va = np.radians(bus_df_indexed["Va"])
        Vm = bus_df_indexed["Vm"]

    elif flag=='pred':
        Va = np.radians(bus_df_indexed["Va_pred_corrected"])
        Vm = bus_df_indexed["Vm_pred_corrected"]

    else:
        Va = np.radians(bus_df_indexed["Va_dc_corrected"])
        Vm = bus_df_indexed["Vm_dc_corrected"]

    bus_df_indexed["V"] = Vm * (np.cos(Va) + 1j * np.sin(Va))
    Vf = bus_df_indexed["V"].loc[idx_from].to_numpy(dtype=np.complex128)
    Vt = bus_df_indexed["V"].loc[idx_to].to_numpy(dtype=np.complex128)

    Yff = branch_df["Yff_r"].to_numpy(dtype=np.float64) + 1j * branch_df[
        "Yff_i"
    ].to_numpy(dtype=np.float64)
    Yft = branch_df["Yft_r"].to_numpy(dtype=np.float64) + 1j * branch_df[
        "Yft_i"
    ].to_numpy(dtype=np.float64)
    Ytf = branch_df["Ytf_r"].to_numpy(dtype=np.float64) + 1j * branch_df[
        "Ytf_i"
    ].to_numpy(dtype=np.float64)
    Ytt = branch_df["Ytt_r"].to_numpy(dtype=np.float64) + 1j * branch_df[
        "Ytt_i"
    ].to_numpy(dtype=np.float64)

    If = Yff * Vf + Yft * Vt
    It = Ytt * Vt + Ytf * Vf

    Sf = Vf * np.conj(If) * sn_mva
    St = Vt * np.conj(It) * sn_mva

    pf = np.real(Sf)
    qf = np.imag(Sf)
    pt = np.real(St)
    qt = np.imag(St)

    return pf, qf, pt, qt

# Plotting utils
def compute_cm_metrics(y_test, y_pred, model_name, label_plot):
    """
    Compute confusion matrix (TP,FP,TN,FN) for predicted overleads along with their respective rates and accuracy metric.

    Parameters:
    - y_pred: predicted overloads
    - y_test: ground truth overloads
    - prediction_dir:
    - label_plot:
    """

    TP = (y_test & y_pred).sum()
    FP = ((~y_test) & y_pred).sum()
    TN = ((~y_test) & (~y_pred)).sum()
    FN = (y_test & (~y_pred)).sum()

    # accuracy
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    print(f"Accuracy: {accuracy:.3f}")

    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    TNR = TN / (TN + FP)
    FNR = FN / (FN + TP)
    # TODO change text to fit both overloadings and voltage violations
    print("Confusion Matrix:")
    print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
    print(
        f"GENCO\nTPR: {TPR:.3f} (percentage of overloadings correctly predicted)\nFPR: {FPR:.3f} (percentage of non-overloadings predicted as overloadings)\nTNR: {TNR:.2f}\nFNR: {FNR:.2f}",
    )
    with open(f"metrics_overloading_{model_name}.txt", "w") as f:
        f.write(f"Accuracy: {accuracy:.3f}\n")
        f.write("Confusion Matrix:\n")
        f.write(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}\n")
        f.write(f"{label_plot} Metrics:\n")
        f.write(f"TPR: {TPR:.5f} (percentage of overloadings correctly predicted)\n")
        f.write(
            f"FPR: {FPR:.5f} (percentage of non-overloadings predicted as overloadings)\n",
        )
        f.write(f"TNR: {TNR:.5f}\n")
        f.write(f"FNR: {FNR:.5f}\n")
    return TP, FP, TN, FN


def plot_mass_correlation_density(
    true_vals,
    gfm_vals,
    model_name,
    label_plot,
    x_max=2,
    y_max=3,
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

    # estimate vmax on mean count of elements across bins
    counts, _, _ = np.histogram2d(true_vals, gfm_vals, bins=[x_bins, y_bins])

    counts[counts == 0] = np.nan
    means = np.nanmean(counts)
    std = np.nanstd(counts)
    vmax = means + 3 * std

    # Pearson correlations
    corr_gfm, _ = pearsonr(true_vals, gfm_vals)

    # Create figure with shared x-axis
    fig, ax1 = plt.subplots(figsize=(9, 7))

    # --- GENCO Mass Correlation ---
    h1 = ax1.hist2d(
        true_vals,
        gfm_vals,
        bins=[x_bins, y_bins],
        norm=LogNorm(vmin=vmin, vmax=vmax),
        cmap="inferno",
    )
    ax1.axvline(1, color="black", linestyle="--", linewidth=2.0)
    ax1.axhline(1, color="black", linestyle="--", linewidth=2.0)
    ax1.plot([0, 5], [0, 5], "k--", linewidth=0.5)
    ax1.set_xlabel("True Loadings", fontsize=12)
    ax1.set_ylabel("Predicted Loadings", fontsize=12)
    ax1.set_title(label_plot, fontsize=14)
    ax1.text(
        x_max - 1.5,
        0.93,
        f"r = {corr_gfm:.5f}",
        transform=ax1.transAxes,
        fontsize=13,
        weight="bold",
    )

    # Colorbar
    cbar = fig.colorbar(h1[3], ax=ax1, pad=0.02)
    cbar.set_label("Number of samples", fontsize=10)

    # Style adjustments
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.grid(True, linewidth=0.3)
    ax1.tick_params(axis="both", labelsize=10)

    plt.tight_layout()
    plt.savefig(f"mass_correlation_density_{model_name}.png", bbox_inches="tight")
    plt.show()

def plot_cm(TN, FP, FN, TP, model_name, label_plot):
    """
    TODO docstring
    """
    cm = np.array([[TN, FP], [FN, TP]])

    cm_labels = ["Non-overload", "Overload"]

    fig_cm, ax_cm = plt.subplots(figsize=(6, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cbar=False,
        square=True,
        linewidths=0.5,
        cmap="Blues",
        xticklabels=cm_labels,
        yticklabels=cm_labels,
        ax=ax_cm,
        annot_kws={"size": 14},
    )

    ax_cm.set_xlabel("Predicted", fontsize=12)
    ax_cm.set_ylabel("True", fontsize=12)
    ax_cm.set_title(f"Confusion Matrix {label_plot}", fontsize=14)
    ax_cm.tick_params(axis="both", labelsize=12)

    plt.tight_layout()
    plt.savefig(f"confusion_matrix_overload_{model_name}.png", bbox_inches="tight")
    plt.show()

def plot_cm_vm(TN, FP, FN, TP, model_name, label_plot):
    """
    TODO docstring
    """
    cm = np.array([[TN, FP], [FN, TP]])

    cm_labels = ["No violation", "Violation"]

    fig_cm, ax_cm = plt.subplots(figsize=(6, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cbar=False,
        square=True,
        linewidths=0.5,
        cmap="Blues",
        xticklabels=cm_labels,
        yticklabels=cm_labels,
        ax=ax_cm,
        annot_kws={"size": 14},
    )

    ax_cm.set_xlabel("Predicted", fontsize=12)
    ax_cm.set_ylabel("True", fontsize=12)
    ax_cm.set_title(f"Confusion Matrix {label_plot}", fontsize=14)
    ax_cm.tick_params(axis="both", labelsize=12)

    plt.tight_layout()
    plt.savefig(f"confusion_matrix_overload_{model_name}.png", bbox_inches="tight")
    plt.show()

def plot_loading_predictions(
    loadings_pred,
    loadings_dc,
    loadings_gt,
    prediction_dir,
    label_plot,
):
    """
    TODO docstrings
    """
    plt.hist(
        loadings_pred,
        alpha=0.5,
        label=label_plot,
        density=True,
        bins=100,
    )
    plt.hist(loadings_dc, alpha=0.5, label="DC Solver", density=True, bins=100)
    plt.hist(loadings_gt, alpha=0.5, label="Ground truth", density=True, bins=100)

    plt.xlabel("Loading Values")
    plt.ylabel("Density")
    plt.yscale("log")
    plt.legend()

    plt.savefig(f"distribution_loading_predictions_{prediction_dir}.png")
    plt.show()

def plot_mass_correlation_density_voltage(
    pf_node,
    prediction_dir,
    label_plot,
    x_min=0.85,
    y_min=0.85,
    x_max=1.15,
    y_max=1.15,
    vm_nominal=1.0,
    vm_dev_threshold=0.05,
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

    # Pearson correlations
    corr_vm, _ = pearsonr(pf_node["Vm"], pf_node["Vm_pred_corrected"])

    # Create figure with shared x-axis
    fig, ax1 = plt.subplots(figsize=(8, 6), dpi=400)

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
    ax1.axvline(vm_lower_limit, color="black", linestyle="--", linewidth=2.0)
    ax1.axhline(vm_lower_limit, color="black", linestyle="--", linewidth=2.0)
    ax1.axvline(vm_upper_limit, color="black", linestyle="--", linewidth=2.0)
    ax1.axhline(vm_upper_limit, color="black", linestyle="--", linewidth=2.0)

    ax1.plot([0, 5], [0, 5], "k--", linewidth=0.5)
    ax1.set_xlabel("True Voltage Magnitude", fontsize=12)
    ax1.set_ylabel("Predicted Voltage magnitude", fontsize=12)
    ax1.set_title(label_plot, fontsize=14)
    ax1.text(
        0.5,
        0.95,
        f"r = {corr_vm:.5f}",
        transform=ax1.transAxes,
        fontsize=13,
        weight="bold",
        ha="center",
        va="top",
    )

    # Colorbar
    cbar = fig.colorbar(h1[3], ax=ax1, pad=0.02)
    cbar.set_label("Number of samples", fontsize=10)

    # Style adjustments
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.grid(True, linewidth=0.3)
    ax1.tick_params(axis="both", labelsize=10)

    plt.tight_layout()
    plt.savefig(
        f"mass_correlation_density_voltage_{prediction_dir}.png",
        bbox_inches="tight",
    )
    plt.show()

