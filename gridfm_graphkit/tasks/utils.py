import torch
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_max
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from gridfm_graphkit.models.utils import ComputeBranchFlow
from gridfm_graphkit.datasets.globals import (
    VA_OUT,
    ANG_MIN,
    ANG_MAX,
    RATE_A,
    YFF_TT_R,
    YFF_TT_I,
    YFT_TF_R,
    YFT_TF_I,
)


def local_index_per_graph(batch_index: torch.Tensor) -> torch.Tensor:
    """Return 0..N-1 local indices for each graph in a batched entity axis."""

    return torch.cat(
        [
            torch.arange(int(count), device=batch_index.device)
            for count in torch.bincount(batch_index)
        ],
    )


def embedding_table_from_tensor(
    embedding: torch.Tensor,
    *,
    id_columns: dict[str, np.ndarray],
    prefix: str = "emb",
) -> dict[str, np.ndarray]:
    """Convert one embedding tensor into a parquet-ready table mapping."""

    values = embedding.detach().cpu().numpy()
    table = {key: value for key, value in id_columns.items()}
    for index in range(values.shape[1]):
        table[f"{prefix}_{index:03d}"] = values[:, index]
    return table


def residual_stats_by_type(residual, mask, bus_batch):
    """Return per-graph mean and max absolute residuals for a masked bus subset."""
    residual_masked = residual[mask]
    batch_masked = bus_batch[mask]
    abs_residual = torch.abs(residual_masked)

    # torch_scatter on MPS can dispatch into a CPU-only path for scatter_max.
    # Compute the grouped stats on CPU and move the results back so verbose
    # evaluation works without changing the torch/torch_scatter stack.
    if abs_residual.device.type == "mps":
        abs_residual_cpu = abs_residual.cpu()
        batch_masked_cpu = batch_masked.cpu()
        mean_res = scatter_mean(abs_residual_cpu, batch_masked_cpu, dim=0).to(
            abs_residual.device,
        )
        max_res, _ = scatter_max(abs_residual_cpu, batch_masked_cpu, dim=0)
        max_res = max_res.to(abs_residual.device)
    else:
        mean_res = scatter_mean(abs_residual, batch_masked, dim=0)
        max_res, _ = scatter_max(abs_residual, batch_masked, dim=0)
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
        all_data = (
            torch.cat(
                [
                    torch.cat([d[f"{stat_key}_{bus_type}"] for d in outputs])
                    for bus_type in bus_types
                ],
            )
            .float()
            .numpy()
        )

        # Define bins across the entire data range
        bins = np.linspace(all_data.min(), all_data.max(), 61)  # 30 bins of equal width

        plt.figure(figsize=(10, 6))
        for bus_type, color in zip(bus_types, colors):
            data = (
                torch.cat([d[f"{stat_key}_{bus_type}"] for d in outputs])
                .float()
                .numpy()
            )
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


def compute_branch_predictions(
    eval_bus,
    target,
    bus_edge_index,
    bus_edge_attr,
    scenario_ids,
    local_bus_idx,
):
    """Compute branch-level predictions and ground-truth constraint violations.

    Args:
        eval_bus:       Clamped model predictions [num_bus, 4]. Branch flows
                        and angle violations are computed from this.
        target:         Ground truth bus tensor [num_bus, 4]. Target branch
                        flows and angle violations are computed from this.
        bus_edge_index: Edge index [2, num_edges] (batch-global bus indices).
        bus_edge_attr:  Edge features [num_edges, num_edge_features].
        scenario_ids:   Scenario ID per bus [num_bus] (batch-global).
        local_bus_idx:  Per-graph local bus index [num_bus].

    Returns:
        dict of numpy arrays, one entry per directed edge.
    """
    branch_flow_layer = ComputeBranchFlow()

    from_bus_idx = bus_edge_index[0]
    to_bus_idx = bus_edge_index[1]

    # Branch limits — ANG_MIN/ANG_MAX restored to degrees by inverse_transform;
    # convert to radians to match VA_OUT which stays in radians.
    angle_min = bus_edge_attr[:, ANG_MIN] * torch.pi / 180.0
    angle_max = bus_edge_attr[:, ANG_MAX] * torch.pi / 180.0
    branch_thermal_limits = bus_edge_attr[:, RATE_A]

    def _branch_flows(bus_state):
        Pft, Qft = branch_flow_layer(bus_state, bus_edge_index, bus_edge_attr)
        Sft = torch.sqrt(Pft**2 + Qft**2)
        thermal_excess = F.relu(Sft - branch_thermal_limits)
        return Pft, Qft, thermal_excess

    def _angle_violations(bus_state):
        angles = bus_state[:, VA_OUT]
        diff = angles[from_bus_idx] - angles[to_bus_idx]
        diff = (diff + torch.pi) % (2 * torch.pi) - torch.pi  # wrap to [-pi, pi]
        return diff, F.relu(angle_min - diff), F.relu(diff - angle_max)

    # Predicted
    Pft, Qft, thermal_excess = _branch_flows(eval_bus)
    angle_diff, angle_excess_low, angle_excess_high = _angle_violations(eval_bus)

    # Ground truth
    Pft_target, Qft_target, thermal_excess_target = _branch_flows(target)
    angle_diff_target, angle_excess_low_target, angle_excess_high_target = (
        _angle_violations(target)
    )

    def _np(t):
        return t.detach().cpu().numpy()

    return {
        "scenario": scenario_ids[from_bus_idx].cpu().numpy(),
        "from_bus": local_bus_idx[from_bus_idx].cpu().numpy(),
        "to_bus": local_bus_idx[to_bus_idx].cpu().numpy(),
        "Pft": _np(Pft),
        "Qft": _np(Qft),
        "Pft_target": _np(Pft_target),
        "Qft_target": _np(Qft_target),
        "angle_diff": _np(angle_diff),
        "angle_excess_low": _np(angle_excess_low),
        "angle_excess_high": _np(angle_excess_high),
        "angle_diff_target": _np(angle_diff_target),
        "angle_excess_low_target": _np(angle_excess_low_target),
        "angle_excess_high_target": _np(angle_excess_high_target),
        "thermal_excess": _np(thermal_excess),
        "thermal_excess_target": _np(thermal_excess_target),
        # Fields needed for current-based loading computation
        "rate_a": _np(branch_thermal_limits),
        "Yff_r": _np(bus_edge_attr[:, YFF_TT_R]),
        "Yff_i": _np(bus_edge_attr[:, YFF_TT_I]),
        "Yft_r": _np(bus_edge_attr[:, YFT_TF_R]),
        "Yft_i": _np(bus_edge_attr[:, YFT_TF_I]),
    }
