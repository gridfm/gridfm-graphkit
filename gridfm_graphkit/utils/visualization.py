from gridfm_graphkit.training.loss import PBELoss

from types import SimpleNamespace

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import pearsonr
import seaborn as sns
import numpy as np


BUS_NODE_SHAPES = {"REF": "s", "PV": "H", "PQ": "o"}


def _bus_graph(data):
    """Build the bus-level topology and per-bus type labels from a HeteroData sample.

    Args:
        data: A single (unbatched) ``HeteroData`` sample carrying a
            ``("bus", "connects", "bus")`` edge type and a ``mask_dict`` with the
            boolean ``PQ`` / ``PV`` / ``REF`` bus-type vectors.

    Returns:
        A ``(networkx.Graph, dict)`` pair: the undirected bus graph (isolated
        buses included) and a mapping from bus index to ``"REF"``/``"PV"``/``"PQ"``.
    """
    num_bus = data["bus"].x.shape[0]
    edge_index = data.edge_index_dict[("bus", "connects", "bus")]

    graph = nx.Graph()
    graph.add_nodes_from(range(num_bus))
    graph.add_edges_from(
        (u, v) for u, v in zip(edge_index[0].tolist(), edge_index[1].tolist()) if u != v
    )

    mask_dict = data.mask_dict
    is_ref, is_pv, is_pq = mask_dict["REF"], mask_dict["PV"], mask_dict["PQ"]
    node_labels = {}
    for i in range(num_bus):
        if is_ref[i]:
            node_labels[i] = "REF"
        elif is_pv[i]:
            node_labels[i] = "PV"
        elif is_pq[i]:
            node_labels[i] = "PQ"
    return graph, node_labels


def _draw_bus_nodes(graph, pos, node_labels, values, ax, vmin, vmax, cmap, resize_ref):
    """Draw bus nodes coloured by ``values``, using a distinct marker per bus type."""
    for node_type, shape in BUS_NODE_SHAPES.items():
        nodes = [i for i in node_labels if node_labels[i] == node_type]
        if not nodes:
            continue
        node_size = (390 if node_type == "REF" else 600) if resize_ref else 800
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=nodes,
            node_color=[values[i] for i in nodes],
            cmap=cmap,
            node_size=node_size,
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            node_shape=shape,
        )


def _label_bus_nodes(graph, pos, node_labels, ax, width=2):
    """Draw the bus edges and overlay the bus-type label on each node."""
    nx.draw_networkx_edges(graph, pos, edge_color="gray", alpha=0.5, ax=ax, width=width)
    nx.draw_networkx_labels(
        graph,
        pos,
        labels=node_labels,
        font_size=10,
        font_color="white",
        font_weight="bold",
        ax=ax,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(2)


def visualize_error(data, output, baseMVA=1.0):
    """Plot per-bus active power residuals of a prediction on the grid topology.

    Residuals come from the power balance equations via
    [`PBELoss`][gridfm_graphkit.training.loss.PBELoss] in visualization mode, which
    returns the per-bus mismatch instead of only its mean.

    Args:
        data: A single (unbatched) ``HeteroData`` sample.
        output: Model prediction dict, ``{"bus": Tensor[N_bus, 4], "gen": ...}``.
        baseMVA: Power base used to convert the p.u. residuals to MW. Pass the
            sample's ``baseMVA`` to get MW; leave at ``1.0`` to plot p.u.

    Returns:
        The per-bus active power residuals as a detached CPU tensor.
    """
    loss = PBELoss(SimpleNamespace(visualization=True), None)
    loss_dict = loss(
        output,
        data.y_dict,
        data.edge_index_dict,
        data.edge_attr_dict,
        data.mask_dict,
        x_dict=data.x_dict,
    )
    active_loss = loss_dict["Nodal Active Power Loss in p.u."].detach().cpu() * baseMVA

    graph, node_labels = _bus_graph(data)
    pos = nx.spring_layout(graph, seed=42)

    cmap = plt.cm.viridis
    vmin, vmax = float(active_loss.min()), float(active_loss.max())
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(13, 7))
    _draw_bus_nodes(
        graph,
        pos,
        node_labels,
        active_loss,
        ax,
        vmin,
        vmax,
        cmap,
        resize_ref=False,
    )
    _label_bus_nodes(graph, pos, node_labels, ax, width=1)

    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax)
    unit = "MW" if baseMVA != 1.0 else "p.u."
    cbar.set_label(f"Active Power Residuals ({unit})", fontsize=12)
    cbar.ax.tick_params(labelsize=12)

    plt.title("Nodal Active Power Residuals", fontsize=14, fontweight="bold")
    plt.show()
    return active_loss


def visualize_quantity_heatmap(
    data,
    output,
    pred_col,
    target_col,
    quantity_name,
    unit,
    scale=1.0,
):
    """Compare ground truth, masked input and reconstruction for one bus quantity.

    Draws three panels on the bus topology: the ground truth, the same values with
    the masked (unknown) buses greyed out, and the model's reconstruction. Buses
    the model was *given* are clamped back to ground truth in the third panel, so
    only the masked buses show reconstruction error.

    Bus predictions and bus targets use different column layouts, hence the two
    separate indices: ``pred_col`` indexes the 4-column model output
    (``VM_OUT``/``VA_OUT``/``PG_OUT``/``QG_OUT``) while ``target_col`` indexes the
    bus feature layout (``VM_H``/``VA_H``/``QG_H``/…), which ``data["bus"].y`` and
    ``mask_dict["bus"]`` share.

    Args:
        data: A single (unbatched) ``HeteroData`` sample.
        output: Model prediction dict, ``{"bus": Tensor[N_bus, 4], "gen": ...}``.
        pred_col: Column of ``output["bus"]`` holding the quantity.
        target_col: Column of ``data["bus"].y`` / ``mask_dict["bus"]`` holding it.
        quantity_name: Human-readable name, used in the titles.
        unit: Unit shown on the colourbar.
        scale: Factor applied to both prediction and target before plotting, to
            convert from the normalized representation (e.g. ``baseMVA`` for
            powers, ``180 / pi`` for voltage angles).
    """
    gt_values = (data["bus"].y[:, target_col].detach().cpu() * scale).clone()
    predicted_values = (output["bus"][:, pred_col].detach().cpu() * scale).clone()

    # Only masked buses are actually reconstructed; the rest were model inputs.
    mask = data.mask_dict["bus"][:, target_col].detach().cpu()
    predicted_values[~mask] = gt_values[~mask]
    masked_node_indices = np.where(mask.numpy())[0]

    graph, node_labels = _bus_graph(data)
    pos = nx.spring_layout(graph, seed=42)

    cmap = plt.cm.viridis
    # Share one colour scale across all three panels so they are comparable.
    vmin = float(min(gt_values.min(), predicted_values.min()))
    vmax = float(max(gt_values.max(), predicted_values.max()))
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(1, 3, figsize=(22, 8))

    panels = (
        (axes[0], gt_values, f"Ground truth {quantity_name}"),
        (axes[1], gt_values, f"Masked {quantity_name}"),
        (axes[2], predicted_values, f"Reconstructed {quantity_name}"),
    )
    for ax, values, title in panels:
        _draw_bus_nodes(
            graph,
            pos,
            node_labels,
            values,
            ax,
            vmin,
            vmax,
            cmap,
            resize_ref=True,
        )
        if ax is axes[1]:
            # Grey out the buses whose value was hidden from the model.
            nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=masked_node_indices,
                node_color="#D3D3D3",
                node_size=750,
                ax=ax,
            )
        _label_bus_nodes(graph, pos, node_labels, ax)
        ax.set_title(title, fontsize=14, fontweight="bold")

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
    fig, ax1 = plt.subplots(figsize=(9, 7))

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
