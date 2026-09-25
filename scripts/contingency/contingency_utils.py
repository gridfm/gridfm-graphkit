import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import pearsonr
import seaborn as sns
import numpy as np
import copy
import pandas as pd
from typing import Tuple

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

    elif flag=='dc':
        Va = np.radians(bus_df_indexed["Va_dc_corrected"])
        Vm = bus_df_indexed["Vm_dc_corrected"]

    else:
        raise ValueError(f"Invalid flag: {flag}")
    bus_df_indexed["V"] = Vm * (np.cos(Va) + 1j * np.sin(Va))
    # Reindex by branch-aligned (scenario, bus) keys to preserve branch_df row order.
    Vf = bus_df_indexed["V"].reindex(idx_from).to_numpy(dtype=np.complex128)
    Vt = bus_df_indexed["V"].reindex(idx_to).to_numpy(dtype=np.complex128)

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
def compute_cm_metrics(
    y_test,
    y_pred,
    model_name,
    label_plot,
    task_label: str = "overloading classification",
):
    """
    Compute confusion matrix metrics and save a readable text report.

    Parameters:
    - y_pred: predicted overloads
    - y_test: ground truth overloads
    - prediction_dir:
    - label_plot:
    """

    TP = int((y_test & y_pred).sum())
    FP = int(((~y_test) & y_pred).sum())
    TN = int(((~y_test) & (~y_pred)).sum())
    FN = int((y_test & (~y_pred)).sum())

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
        f"{label_plot}\n"
        f"TPR/Recall: {TPR:.6f}\n"
        f"FPR: {FPR:.6f}\n"
        f"TNR/Specificity: {TNR:.6f}\n"
        f"FNR: {FNR:.6f}",
    )
    with open(f"metrics_overloading_{model_name}.txt", "w") as f:
        f.write(f"Task: {task_label}\n")
        f.write(f"Model: {label_plot}\n")
        f.write(f"Accuracy: {accuracy:.3f}\n")
        f.write("Confusion Matrix:\n")
        f.write(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}\n")
        f.write("Rates:\n")
        f.write(f"TPR/Recall: {TPR:.5f}\n")
        f.write(f"FPR: {FPR:.5f}\n")
        f.write(f"TNR/Specificity: {TNR:.5f}\n")
        f.write(f"FNR: {FNR:.5f}\n")
    return TP, FP, TN, FN


def sweep_threshold_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: np.ndarray,
    beta: float = 10.0,
) -> pd.DataFrame:
    """Compute confusion-matrix-derived metrics across thresholds."""
    y_true = np.asarray(y_true, dtype=bool)
    y_score = np.asarray(y_score, dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64)

    rows = []
    beta_sq = beta**2
    for threshold in thresholds:
        y_pred = y_score > threshold

        tp = int((y_true & y_pred).sum())
        fp = int(((~y_true) & y_pred).sum())
        tn = int(((~y_true) & (~y_pred)).sum())
        fn = int((y_true & (~y_pred)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        tpr = recall
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        denom = beta_sq * precision + recall
        fbeta = ((1 + beta_sq) * precision * recall / denom) if denom > 0 else 0.0

        rows.append(
            {
                "threshold": float(threshold),
                "TP": tp,
                "FP": fp,
                "TN": tn,
                "FN": fn,
                "precision": precision,
                "recall": recall,
                "TPR": tpr,
                "FPR": fpr,
                "Fbeta": fbeta,
            }
        )

    return pd.DataFrame(rows)


def make_threshold_metrics_table(
    metrics_df: pd.DataFrame,
    model_label: str,
    beta: float,
) -> pd.DataFrame:
    """Create a publication-ready threshold sweep table with clear columns."""
    table = metrics_df.copy()
    table["Model"] = model_label
    table["Threshold"] = table["threshold"]
    table["True Positives"] = table["TP"]
    table["False Positives"] = table["FP"]
    table["True Negatives"] = table["TN"]
    table["False Negatives"] = table["FN"]
    table["Precision"] = table["precision"]
    table["Recall"] = table["recall"]
    table["True Positive Rate (TPR)"] = table["TPR"]
    table["False Positive Rate (FPR)"] = table["FPR"]
    table[f"F-beta (beta={beta:g})"] = table["Fbeta"]

    ordered_cols = [
        "Model",
        "Threshold",
        "True Positives",
        "False Positives",
        "True Negatives",
        "False Negatives",
        "Precision",
        "Recall",
        "True Positive Rate (TPR)",
        "False Positive Rate (FPR)",
        f"F-beta (beta={beta:g})",
    ]
    return table[ordered_cols]


def make_best_threshold_summary(
    best_row: pd.Series,
    model_label: str,
    task_label: str,
    beta: float,
) -> pd.DataFrame:
    """Create a one-row, publication-ready best-threshold summary table."""
    return pd.DataFrame(
        [
            {
                "Task": task_label,
                "Model": model_label,
                "Best Threshold": float(best_row["threshold"]),
                "True Positives": int(best_row["TP"]),
                "False Positives": int(best_row["FP"]),
                "True Negatives": int(best_row["TN"]),
                "False Negatives": int(best_row["FN"]),
                "Precision": float(best_row["precision"]),
                "Recall": float(best_row["recall"]),
                "True Positive Rate (TPR)": float(best_row["TPR"]),
                "False Positive Rate (FPR)": float(best_row["FPR"]),
                f"F-beta (beta={beta:g})": float(best_row["Fbeta"]),
            }
        ]
    )


def make_confusion_summary_table(
    tp: int,
    fp: int,
    tn: int,
    fn: int,
    model_label: str,
    task_label: str,
) -> pd.DataFrame:
    """Create a publication-ready confusion matrix summary table."""
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return pd.DataFrame(
        [
            {
                "Task": task_label,
                "Model": model_label,
                "True Positives": int(tp),
                "False Positives": int(fp),
                "True Negatives": int(tn),
                "False Negatives": int(fn),
                "Accuracy": float(accuracy),
                "Precision": float(precision),
                "Recall": float(recall),
                "Specificity (TNR)": float(specificity),
                "False Positive Rate (FPR)": float(fpr),
                "False Negative Rate (FNR)": float(fnr),
            }
        ]
    )


def pick_best_threshold(metrics_df: pd.DataFrame) -> pd.Series:
    """Pick threshold maximizing Fbeta, then recall, then minimizing FPR."""
    ranked = metrics_df.sort_values(
        by=["Fbeta", "recall", "FPR"],
        ascending=[False, False, True],
    )
    return ranked.iloc[0]


def plot_tpr_fpr_threshold_curves(
    model_metrics_df: pd.DataFrame,
    dc_metrics_df: pd.DataFrame,
    best_model_row: pd.Series,
    best_dc_row: pd.Series,
    output_path: str,
    model_label: str = "GENCO",
    dc_label: str = "DC",
    xlim: Tuple[float, float] = (0.0, 0.02),
    ylim: Tuple[float, float] = (0.95, 1.0),
):
    """Plot TPR-FPR trajectories with threshold annotations."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=200)
    eps = 1e-6

    # Sort by FPR so trajectories are smooth and comparable.
    model_plot = model_metrics_df.sort_values("FPR")
    dc_plot = dc_metrics_df.sort_values("FPR")
    model_fpr_plot = np.clip(model_plot["FPR"].to_numpy(dtype=float), eps, None)
    dc_fpr_plot = np.clip(dc_plot["FPR"].to_numpy(dtype=float), eps, None)

    ax.plot(model_fpr_plot, model_plot["TPR"], label=model_label, alpha=0.7)
    ax.plot(dc_fpr_plot, dc_plot["TPR"], label=dc_label, alpha=0.7)

    # Sparse threshold annotations to keep readability.
    for _, row in model_metrics_df.iloc[:: max(1, len(model_metrics_df) // 8)].iterrows():
        row_fpr = max(float(row["FPR"]), eps)
        ax.scatter(row_fpr, row["TPR"], color="red", s=18, zorder=3)
        ax.annotate(
            f"{row['threshold']:.3f}",
            (row_fpr, row["TPR"]),
            textcoords="offset points",
            xytext=(4, -6),
            fontsize=9,
            color="black",
        )

    for _, row in dc_metrics_df.iloc[:: max(1, len(dc_metrics_df) // 8)].iterrows():
        row_fpr = max(float(row["FPR"]), eps)
        ax.scatter(row_fpr, row["TPR"], color="blue", s=18, zorder=3)
        ax.annotate(
            f"{row['threshold']:.3f}",
            (row_fpr, row["TPR"]),
            textcoords="offset points",
            xytext=(4, -6),
            fontsize=9,
            color="black",
        )

    # Highlight selected best thresholds.
    ax.scatter(max(float(best_model_row["FPR"]), eps), best_model_row["TPR"], color="red", s=50, zorder=4)
    ax.scatter(max(float(best_dc_row["FPR"]), eps), best_dc_row["TPR"], color="blue", s=50, zorder=4)

    x_min = max(float(xlim[0]), eps)
    x_max = float(xlim[1])
    if x_max <= x_min:
        x_max = max(np.max(np.r_[model_fpr_plot, dc_fpr_plot]) * 1.05, x_min * 10)
    ax.set_xscale("log")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(*ylim)
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_tpr_fpr_threshold_curve_single(
    metrics_df: pd.DataFrame,
    best_row: pd.Series,
    output_path: str,
    model_label: str = "GENCO",
    xlim: Tuple[float, float] = (0.0, 0.02),
    ylim: Tuple[float, float] = (0.95, 1.0),
):
    """Plot a single-model TPR-FPR trajectory with threshold annotations."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=200)
    eps = 1e-6
    metrics_plot = metrics_df.sort_values(["FPR", "TPR", "threshold"]).reset_index(drop=True)
    metrics_fpr_plot = np.clip(metrics_plot["FPR"].to_numpy(dtype=float), eps, None)
    metrics_tpr_plot = metrics_plot["TPR"].to_numpy(dtype=float)

    # Step trajectory makes plateau regions (same TPR, rising FPR) visually explicit.
    ax.step(metrics_fpr_plot, metrics_tpr_plot, where="post", label=model_label, alpha=0.8)
    ax.scatter(metrics_fpr_plot, metrics_tpr_plot, color="red", s=10, alpha=0.35, zorder=2)

    fpr_only_increase = np.r_[
        False,
        np.isclose(np.diff(metrics_tpr_plot), 0.0) & (np.diff(metrics_fpr_plot) > 0.0),
    ]
    if np.any(fpr_only_increase):
        ax.scatter(
            metrics_fpr_plot[fpr_only_increase],
            metrics_tpr_plot[fpr_only_increase],
            color="orange",
            edgecolor="black",
            linewidth=0.4,
            s=28,
            zorder=4,
            label="FPR increase only",
        )

    for _, row in metrics_plot.iloc[:: max(1, len(metrics_plot) // 8)].iterrows():
        row_fpr = max(float(row["FPR"]), eps)
        ax.scatter(row_fpr, row["TPR"], color="red", s=18, zorder=3)
        ax.annotate(
            f"{row['threshold']:.3f}",
            (row_fpr, row["TPR"]),
            textcoords="offset points",
            xytext=(4, -6),
            fontsize=9,
            color="black",
        )

    ax.scatter(max(float(best_row["FPR"]), eps), best_row["TPR"], color="red", s=50, zorder=4)

    x_min = max(float(xlim[0]), eps)
    x_max = float(xlim[1])
    if x_max <= x_min:
        x_max = max(np.max(metrics_fpr_plot) * 1.05, x_min * 10)
    ax.set_xscale("log")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(*ylim)
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_mass_correlation_density(
    true_vals,
    predicted_vals,
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
    counts, _, _ = np.histogram2d(true_vals, predicted_vals, bins=[x_bins, y_bins])

    counts[counts == 0] = np.nan
    means = np.nanmean(counts)
    std = np.nanstd(counts)
    vmax = means + 3 * std

    # Pearson correlations
    corr_pred, _ = pearsonr(true_vals, predicted_vals)

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
    ax1.set_xlabel("True Loadings", fontsize=12)
    ax1.set_ylabel("Predicted Loadings", fontsize=12)
    ax1.set_title(label_plot, fontsize=14)
    ax1.text(
        x_max - 1.5,
        0.93,
        f"r = {corr_pred:.5f}",
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
    plt.close()

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
    plt.close()

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
    plt.close()

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
    ax1.axvline(vm_lower_limit, color="black", linestyle=":", linewidth=2.0)
    ax1.axhline(vm_lower_limit, color="black", linestyle=":", linewidth=2.0)
    ax1.axvline(vm_upper_limit, color="black", linestyle=":", linewidth=2.0)
    ax1.axhline(vm_upper_limit, color="black", linestyle=":", linewidth=2.0)

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
    plt.close()

