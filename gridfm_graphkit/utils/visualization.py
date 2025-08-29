from gridfm_graphkit.training.loss import PBELoss
from gridfm_graphkit.datasets.globals import PQ, PV, REF

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import pearsonr
import seaborn as sns
import numpy as np
import copy


def visualize_error(data_point, output, node_normalizer):
    loss = PBELoss(visualization=True)

    loss_dict = loss(
        output,
        data_point.y,
        data_point.edge_index,
        data_point.edge_attr,
        data_point.mask,
    )
    active_loss = loss_dict["Nodal Active Power Loss in p.u."]
    active_loss = active_loss.cpu() * node_normalizer.baseMVA

    # Create a graph
    G = nx.Graph()
    edges = [
        (u, v)
        for u, v in zip(
            data_point.edge_index[0].tolist(),
            data_point.edge_index[1].tolist(),
        )
        if u != v
    ]
    G.add_edges_from(edges)

    # Assign labels based on node type
    node_shapes = {"REF": "s", "PV": "H", "PQ": "o"}
    num_nodes = data_point.x.shape[0]
    mask_PQ = data_point.x[:, PQ] == 1
    mask_PV = data_point.x[:, PV] == 1
    mask_REF = data_point.x[:, REF] == 1
    node_labels = {}
    for i in range(num_nodes):
        if mask_REF[i]:
            node_labels[i] = "REF"
        elif mask_PV[i]:
            node_labels[i] = "PV"
        elif mask_PQ[i]:
            node_labels[i] = "PQ"

    # Set node positions
    pos = nx.spring_layout(G, seed=42)

    # Define colormap
    cmap = plt.cm.viridis
    vmin = min(active_loss)
    vmax = max(active_loss)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(13, 7))

    # Draw nodes with heatmap coloring
    for node_type, shape in node_shapes.items():
        nodes = [i for i in node_labels if node_labels[i] == node_type]
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=nodes,
            node_color=[active_loss[i] for i in nodes],
            cmap=cmap,
            node_size=800,
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            node_shape=shape,
        )

    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=0.5, ax=ax)

    # Draw labels (node types)
    nx.draw_networkx_labels(
        G,
        pos,
        labels=node_labels,
        font_size=10,
        font_color="white",
        font_weight="bold",
        ax=ax,
    )

    # Add colorbar
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax)
    cbar.set_label("Active Power Residuals (MW)", fontsize=12)
    cbar.ax.tick_params(labelsize=12)

    for spine in ax.spines.values():
        spine.set_linewidth(2)  # Adjust thickness here (e.g., 2 or any value)

    # Show plot
    plt.title("Nodal Active Power Residuals", fontsize=14, fontweight="bold")
    plt.show()


def visualize_quantity_heatmap(
    data_point,
    output,
    quantity,
    quantity_name,
    unit,
    node_normalizer,
):
    """
    Visualizes a heatmap of a specified quantity (VM, PD, QD, PG, QG, VA) for a given dataset and model.

    Parameters:
        data_point: Power grid data.
        model: The trained model used for inference.
        quantity: The quantity to visualize (e.g., VM, PD, QD, PG, QG, VA).
    """
    data_point = copy.deepcopy(data_point)
    output = copy.deepcopy(output)
    mask_PQ = data_point.x[:, PQ] == 1
    mask_PV = data_point.x[:, PV] == 1
    mask_REF = data_point.x[:, REF] == 1

    output = node_normalizer.inverse_transform(output)
    denormalized_gt = node_normalizer.inverse_transform(data_point.y)

    gt_values = denormalized_gt[:, quantity]
    predicted_values = output[:, quantity]
    predicted_values[~data_point.mask[:, quantity]] = denormalized_gt[
        ~data_point.mask[:, quantity],
        quantity,
    ]

    num_nodes = data_point.x.shape[0]

    node_shapes = {"REF": "s", "PV": "H", "PQ": "o"}

    # Create graph
    G = nx.Graph()
    edges = [
        (u, v)
        for u, v in zip(
            data_point.edge_index[0].tolist(),
            data_point.edge_index[1].tolist(),
        )
        if u != v
    ]
    G.add_edges_from(edges)

    node_labels = {}
    for i in range(num_nodes):
        if mask_REF[i]:
            node_labels[i] = "REF"
        elif mask_PV[i]:
            node_labels[i] = "PV"
        elif mask_PQ[i]:
            node_labels[i] = "PQ"

    pos = nx.spring_layout(G, seed=42)
    cmap = plt.cm.viridis
    vmin = min(predicted_values)
    vmax = max(predicted_values)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    masked_node_indices = np.where(data_point.mask[:, quantity].cpu())[0]

    # Create subplots for side-by-side layout (3 plots)
    fig, axes = plt.subplots(1, 3, figsize=(22, 8))

    # First plot (ground truth values)
    ax = axes[0]
    for node_type, shape in node_shapes.items():
        nodes = [i for i in node_labels if node_labels[i] == node_type]
        node_size = 390 if node_type == "REF" else 600
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=nodes,
            node_color=[gt_values[i] for i in nodes],
            cmap=cmap,
            node_size=node_size,
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            node_shape=shape,
        )

    nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=0.5, ax=ax, width=2)
    nx.draw_networkx_labels(
        G,
        pos,
        labels=node_labels,
        font_size=10,
        font_color="white",
        font_weight="bold",
        ax=ax,
    )
    ax.set_title(f"Ground truth {quantity_name}", fontsize=14, fontweight="bold")

    for spine in ax.spines.values():
        spine.set_linewidth(2)  # Adjust thickness

    # Second plot (with masked nodes in gray)
    ax = axes[1]
    for node_type, shape in node_shapes.items():
        nodes = [i for i in node_labels if node_labels[i] == node_type]
        node_size = 390 if node_type == "REF" else 600
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=nodes,
            node_color=[gt_values[i] for i in nodes],
            cmap=cmap,
            node_size=node_size,
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            node_shape=shape,
        )

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=masked_node_indices,
        node_color="#D3D3D3",
        node_size=750,
        ax=ax,
    )
    nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=0.5, ax=ax, width=2)
    nx.draw_networkx_labels(
        G,
        pos,
        labels=node_labels,
        font_size=10,
        font_color="white",
        font_weight="bold",
        ax=ax,
    )
    ax.set_title(f"Masked {quantity_name}", fontsize=14, fontweight="bold")

    for spine in ax.spines.values():
        spine.set_linewidth(2)  # Adjust thickness

    # Third plot (predicted values without masking)
    ax = axes[2]
    for node_type, shape in node_shapes.items():
        nodes = [i for i in node_labels if node_labels[i] == node_type]
        node_size = 390 if node_type == "REF" else 600
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=nodes,
            node_color=[predicted_values[i] for i in nodes],
            cmap=cmap,
            node_size=node_size,
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            node_shape=shape,
        )

    nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=0.5, ax=ax, width=2)
    nx.draw_networkx_labels(
        G,
        pos,
        labels=node_labels,
        font_size=10,
        font_color="white",
        font_weight="bold",
        ax=ax,
    )
    ax.set_title(f"Reconstructed {quantity_name}", fontsize=14, fontweight="bold")

    for spine in ax.spines.values():
        spine.set_linewidth(2)  # Adjust thickness

    # Colorbar placement
    cbar_ax = fig.add_axes([0.93, 0.1, 0.02, 0.8])
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cbar_ax)
    cbar.set_label(f"{quantity_name} ({unit})", fontsize=12)
    cbar.ax.tick_params(labelsize=12)

    plt.subplots_adjust(right=0.9)
    plt.show()


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
    fig, ax1 = plt.subplots(figsize=(8, 6), dpi=400)

    # --- GridFM Mass Correlation ---
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

    # --- GridFM Mass Correlation ---
    h1 = ax1.hist2d(
        pf_node["Vm"],
        pf_node["Vm_pred_corrected"],
        bins=[x_bins, y_bins],
        norm=LogNorm(vmin=vmin, vmax=vmax),
        cmap="inferno",
    )
    ax1.axvline(x_min + 0.05, color="black", linestyle="--", linewidth=2.0)
    ax1.axhline(y_min + 0.05, color="black", linestyle="--", linewidth=2.0)
    ax1.axvline(x_max - 0.05, color="black", linestyle="--", linewidth=2.0)
    ax1.axhline(y_max - 0.05, color="black", linestyle="--", linewidth=2.0)

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
