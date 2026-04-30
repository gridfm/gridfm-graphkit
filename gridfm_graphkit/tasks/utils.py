import torch
from torch_scatter import scatter_mean, scatter_max
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


def residual_stats_by_type(residual, mask, bus_batch):
    residual_masked = residual[mask]
    batch_masked = bus_batch[mask]
    mean_res = scatter_mean(torch.abs(residual_masked), batch_masked, dim=0)
    max_res, _ = scatter_max(torch.abs(residual_masked), batch_masked, dim=0)
    return mean_res, max_res


def plot_residuals_histograms(outputs, dataset_name, plot_dir):
    """
    Plot mean/max residuals for P and Q, per bus type with consistent bins.
    """
    bus_types = ["REF", "PV", "PQ"]
    colors = ["#6baed6", "#fd8d3c", "#74c476"]  # PQ, PV, REF

    stats = [
        ("mean_residual_P", "Mean P Residual"),
        ("mean_residual_Q", "Mean Q Residual"),
        ("max_residual_P", "Max P Residual"),
        ("max_residual_Q", "Max Q Residual"),
    ]

    for stat_key, title in stats:
        # Gather all data first to compute common bin edges
        all_data = torch.cat(
            [
                torch.cat([d[f"{stat_key}_{bus_type}"] for d in outputs])
                for bus_type in bus_types
            ],
        ).numpy()

        # Define bins across the entire data range
        bins = np.linspace(all_data.min(), all_data.max(), 61)  # 30 bins of equal width

        plt.figure(figsize=(10, 6))
        for bus_type, color in zip(bus_types, colors):
            data = torch.cat([d[f"{stat_key}_{bus_type}"] for d in outputs]).numpy()
            plt.hist(data, bins=bins, alpha=0.6, label=bus_type, color=color)

        plt.title(f"{title} per Bus Type in {dataset_name}")
        plt.xlabel("Residual (MW or MVar)")
        plt.ylabel("Frequency")
        plt.legend(title="Bus Type")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        save_path = os.path.join(plot_dir, f"{stat_key}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()


def plot_correlation_by_node_type(
    preds: torch.Tensor,
    targets: torch.Tensor,
    masks: dict,
    feature_labels: list,
    plot_dir: str,
    prefix: str = "",
    xlabel: str = "Target",
    ylabel: str = "Pred",
    qg_violation_mask: torch.Tensor = None,
):
    """
    Create correlation scatter plots per node type (PQ, PV, REF),
    and highlight Qg violations in red if a violation mask is provided.

    Args:
        preds (torch.Tensor): Predictions [N, F]
        targets (torch.Tensor): Targets [N, F]
        masks (dict): { "PQ": mask, "PV": mask, "REF": mask }
        feature_labels (list): Feature labels
        plot_dir (str): Directory to save plots
        prefix (str): Optional filename prefix
        qg_violation_mask (torch.BoolTensor): Global mask of Qg limit violations
    """

    os.makedirs(plot_dir, exist_ok=True)

    for node_type, mask in masks.items():
        if len(mask.shape) == 1:
            preds_masked = preds[mask]
            targets_masked = targets[mask]
        else:
            preds_masked = torch.where(mask, preds, 0)
            targets_masked = torch.where(mask, targets, 0)

        if preds_masked.numel() == 0:
            continue

        # ALSO mask Qg violations for this node type
        if qg_violation_mask is not None:
            qg_violation_mask_local = qg_violation_mask[mask].cpu().numpy()
        else:
            qg_violation_mask_local = None

        fig, axes = plt.subplots(2, 2, figsize=(15, 8))
        axes = axes.flatten()

        for i, (ax, label) in enumerate(zip(axes, feature_labels)):
            x = targets_masked[:, i].cpu().numpy().flatten()
            y = preds_masked[:, i].cpu().numpy().flatten()

            # --- normal scatter for all except Qg ---
            if label != "Qg" or qg_violation_mask_local is None:
                sns.scatterplot(x=x, y=y, s=6, alpha=0.4, ax=ax, edgecolor=None)
            else:
                # --- For Qg: split normal vs violating points ---
                normal_mask = ~qg_violation_mask_local
                viol_mask = qg_violation_mask_local

                # Normal (blue)
                sns.scatterplot(
                    x=x[normal_mask],
                    y=y[normal_mask],
                    s=6,
                    alpha=0.4,
                    ax=ax,
                    edgecolor=None,
                    label="Valid Qg",
                )

                # Violating (RED)
                sns.scatterplot(
                    x=x[viol_mask],
                    y=y[viol_mask],
                    s=8,
                    alpha=0.8,
                    ax=ax,
                    edgecolor="red",
                    color="red",
                    label="Qg violation",
                )

                ax.legend()

            # --- reference y=x line ---
            min_val = min(x.min(), y.min())
            max_val = max(x.max(), y.max())
            ax.plot(
                [min_val, max_val],
                [min_val, max_val],
                "k--",
                linewidth=1.0,
                alpha=0.7,
            )

            # --- R² correlation ---
            corr = np.corrcoef(x, y)[0, 1]
            if label != "Qg" or qg_violation_mask_local is None:
                ax.set_title(f"{node_type} – {label}\nR² = {corr**2:.3f}")
            else:
                num_violations = qg_violation_mask_local.sum().item()
                total_points = qg_violation_mask_local.shape[0]
                ax.set_title(
                    f"{node_type} – {label}\nR² = {corr**2:.3f} - {num_violations} violations out of {total_points} predictions",
                )
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

        plt.tight_layout()
        filename = f"{prefix}_correlation_{node_type}.png"
        plt.savefig(os.path.join(plot_dir, filename), dpi=300)
        plt.close(fig)
