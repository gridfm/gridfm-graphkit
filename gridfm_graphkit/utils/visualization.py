from types import SimpleNamespace

from gridfm_graphkit.training.loss import PBELoss
from gridfm_graphkit.datasets.globals import (
    PD_H,
    PD_OUT,
    PG_H,
    PG_OUT_GEN,
    PQ_H,
    PV_H,
    QD_H,
    QD_OUT,
    QG_H,
    QG_OUT,
    REF_H,
    VA_H,
    VA_OUT,
    VM_H,
    VM_OUT,
)

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import pearsonr
import seaborn as sns
import numpy as np
import copy


BUS_EDGE_TYPE = ("bus", "connects", "bus")
GEN_BUS_EDGE_TYPE = ("gen", "connected_to", "bus")
NODE_SHAPES = {"REF": "s", "PV": "H", "PQ": "o"}
BUS_QUANTITIES = {
    "pd": (PD_H, PD_OUT),
    "qd": (QD_H, QD_OUT),
    "qg": (QG_H, QG_OUT),
    "vm": (VM_H, VM_OUT),
    "va": (VA_H, VA_OUT),
}
BUS_QUANTITY_INDEXES = {
    target_index: name for name, (target_index, _) in BUS_QUANTITIES.items()
}


def _topology(data_point):
    bus_features = data_point.x_dict["bus"]
    num_nodes = bus_features.size(0)
    edge_index = data_point.edge_index_dict[BUS_EDGE_TYPE]
    edges = [
        (int(source), int(target))
        for source, target in zip(edge_index[0].tolist(), edge_index[1].tolist())
        if source != target
    ]
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(edges)

    node_labels = {}
    for node, bus in enumerate(bus_features):
        if bus[PQ_H] == 1:
            node_labels[node] = "PQ"
        elif bus[PV_H] == 1:
            node_labels[node] = "PV"
        elif bus[REF_H] == 1:
            node_labels[node] = "REF"
    return graph, node_labels


def _draw_node_values(ax, graph, pos, values, node_labels, cmap, vmin, vmax):
    for node_type, shape in NODE_SHAPES.items():
        nodes = [node for node, label in node_labels.items() if label == node_type]
        if nodes:
            nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=nodes,
                node_color=values[nodes],
                cmap=cmap,
                node_size=650,
                ax=ax,
                vmin=vmin,
                vmax=vmax,
                node_shape=shape,
            )
    nx.draw_networkx_edges(graph, pos, edge_color="gray", alpha=0.5, ax=ax)
    nx.draw_networkx_labels(
        graph,
        pos,
        labels=node_labels,
        font_size=10,
        font_color="white",
        font_weight="bold",
        ax=ax,
    )
    ax.set_axis_off()


def _draw_masked_nodes(ax, graph, pos, mask, node_labels):
    for node_type, shape in NODE_SHAPES.items():
        nodes = [
            node
            for node, label in node_labels.items()
            if label == node_type and mask[node]
        ]
        if nodes:
            nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=nodes,
                node_color="#D3D3D3",
                node_size=750,
                ax=ax,
                node_shape=shape,
            )


def _numpy(tensor):
    return tensor.detach().cpu().numpy()


def _resolve_quantity(quantity):
    if isinstance(quantity, str):
        name = quantity.strip().lower()
        if name == "pg":
            return "pg"
        if name in BUS_QUANTITIES:
            return name
    elif isinstance(quantity, (int, np.integer)):
        if int(quantity) in BUS_QUANTITY_INDEXES:
            return BUS_QUANTITY_INDEXES[int(quantity)]

    supported = "Pg, " + ", ".join(name.upper() for name in BUS_QUANTITIES)
    raise ValueError(
        f"Unsupported quantity {quantity!r}; supported quantities are {supported}",
    )


def _aggregate_generators_to_buses(values, edge_index, num_buses):
    aggregated = np.zeros(num_buses, dtype=values.dtype)
    np.add.at(aggregated, _numpy(edge_index[1]).astype(np.int64), values)
    return aggregated


def visualize_error(data_point, output, node_normalizer):
    """Plot nodal active-power residuals in MW on the heterogeneous grid.

    PBELoss residuals are in p.u.; they are scaled by ``node_normalizer.baseMVA``.
    """
    loss_args = SimpleNamespace(visualization=True)
    loss = PBELoss(loss_args, args=None)

    loss_dict = loss(
        output,
        data_point.y_dict,
        data_point.edge_index_dict,
        data_point.edge_attr_dict,
        data_point.mask_dict,
        x_dict=data_point.x_dict,
    )
    active_loss = _numpy(loss_dict["Nodal Active Power Loss in p.u."]).reshape(-1)
    active_loss = active_loss * node_normalizer.baseMVA
    graph, node_labels = _topology(data_point)
    if len(active_loss) != graph.number_of_nodes():
        raise ValueError("PBELoss residual count must match the number of bus nodes")

    pos = nx.spring_layout(graph, seed=42)
    cmap = plt.cm.viridis
    vmin = float(np.min(active_loss))
    vmax = float(np.max(active_loss))
    fig, ax = plt.subplots(figsize=(13, 7))
    _draw_node_values(ax, graph, pos, active_loss, node_labels, cmap, vmin, vmax)
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax)),
        ax=ax,
    )
    colorbar.set_label("Nodal Active Power Residual (MW)")
    ax.set_title("Nodal Active Power Residuals", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def visualize_quantity_heatmap(
    data_point,
    output,
    quantity,
    quantity_name,
    unit,
    node_normalizer,
):
    """Plot target, masked target, and reconstructed values on bus topology.

    ``quantity`` accepts ``Pd``, ``Qd``, ``Qg``, ``Vm``, ``Va``, or ``Pg``.
    Integer values are accepted for the corresponding bus target columns
    (0 through 4); ``Pg`` must be selected by name because it is generator-level
    in the targets and outputs. The normalizer is applied to deep copies, so
    plotting does not mutate the supplied sample or prediction dictionaries.
    """
    quantity_key = _resolve_quantity(quantity)
    plot_data = copy.deepcopy(data_point)
    plot_output = copy.deepcopy(output)
    node_normalizer.inverse_transform(plot_data)
    node_normalizer.inverse_output(plot_output, plot_data)

    num_nodes = plot_data.x_dict["bus"].size(0)
    if quantity_key == "pg":
        gen_bus_edge_index = plot_data.edge_index_dict[GEN_BUS_EDGE_TYPE]
        gt_values = _aggregate_generators_to_buses(
            _numpy(plot_data.y_dict["gen"][:, PG_H]),
            gen_bus_edge_index,
            num_nodes,
        )
        predicted_values = _aggregate_generators_to_buses(
            _numpy(plot_output["gen"][:, PG_OUT_GEN]),
            gen_bus_edge_index,
            num_nodes,
        )
        masked_generators = _numpy(plot_data.mask_dict["gen"][:, PG_H]).astype(bool)
        masked_values = np.zeros(num_nodes, dtype=bool)
        np.logical_or.at(
            masked_values,
            _numpy(gen_bus_edge_index[1]).astype(np.int64),
            masked_generators,
        )
    else:
        target_index, output_index = BUS_QUANTITIES[quantity_key]
        if plot_output["bus"].size(1) <= output_index:
            raise ValueError(
                f"Model bus output does not include {quantity_key.upper()} "
                f"(requires column {output_index})",
            )
        gt_values = _numpy(plot_data.y_dict["bus"][:, target_index])
        predicted_values = _numpy(plot_output["bus"][:, output_index])
        masked_values = _numpy(
            plot_data.mask_dict["bus"][:, target_index],
        ).astype(bool)

    reconstructed_values = np.where(masked_values, predicted_values, gt_values)
    graph, node_labels = _topology(plot_data)
    if len(gt_values) != graph.number_of_nodes():
        raise ValueError("Quantity values must match the number of bus nodes")

    pos = nx.spring_layout(graph, seed=42)
    cmap = plt.cm.viridis
    all_values = np.concatenate((gt_values, predicted_values))
    vmin = float(np.nanmin(all_values))
    vmax = float(np.nanmax(all_values))
    fig, axes = plt.subplots(1, 3, figsize=(22, 8))
    _draw_node_values(
        axes[0],
        graph,
        pos,
        gt_values,
        node_labels,
        cmap,
        vmin,
        vmax,
    )
    _draw_node_values(
        axes[1],
        graph,
        pos,
        gt_values,
        node_labels,
        cmap,
        vmin,
        vmax,
    )
    _draw_masked_nodes(axes[1], graph, pos, masked_values, node_labels)
    _draw_node_values(
        axes[2],
        graph,
        pos,
        reconstructed_values,
        node_labels,
        cmap,
        vmin,
        vmax,
    )
    axes[0].set_title(f"Ground truth {quantity_name}", fontsize=14, fontweight="bold")
    axes[1].set_title(f"Masked {quantity_name}", fontsize=14, fontweight="bold")
    axes[2].set_title(f"Reconstructed {quantity_name}", fontsize=14, fontweight="bold")

    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(
            cmap=cmap,
            norm=plt.Normalize(vmin=vmin, vmax=vmax),
        ),
        ax=axes,
        location="right",
        fraction=0.025,
        pad=0.02,
    )
    colorbar.set_label(f"{quantity_name} ({unit})", fontsize=12)
    colorbar.ax.tick_params(labelsize=12)
    fig.tight_layout()
    return fig


def plot_mass_correlation_density(
    true_vals,
    gfm_vals,
    model_name,
    label_plot,
    x_max=2,
    y_max=3,
):
    """Plot a log-scaled 2D histogram of true versus predicted loadings.

    The Pearson correlation of the two arrays is annotated on the plot. The figure
    is saved as ``mass_correlation_density_{model_name}.png`` and then shown.

    Args:
        true_vals: Ground-truth loading values.
        gfm_vals: Model-predicted loading values.
        model_name: Suffix used in the saved file name.
        label_plot: Plot title.
        x_max: Upper x-axis limit.
        y_max: Upper y-axis limit.
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
    """Plot the confusion matrix of line-overload predictions.

    The figure is saved as ``confusion_matrix_overload_{model_name}.png`` and then
    shown.

    Args:
        TN: True negatives (non-overload predicted as non-overload).
        FP: False positives.
        FN: False negatives.
        TP: True positives (overload predicted as overload).
        model_name: Suffix used in the saved file name.
        label_plot: Included in the plot title.
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
    """Plot overlaid normalized histograms of predicted, DC and ground-truth loadings.

    The y-axis is logarithmic. The figure is saved as
    ``distribution_loading_predictions_{prediction_dir}.png`` and then shown.

    Args:
        loadings_pred: Model-predicted loadings.
        loadings_dc: Loadings from the DC solver.
        loadings_gt: Ground-truth loadings.
        prediction_dir: Suffix used in the saved file name.
        label_plot: Legend label for the model predictions.
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
    """Plot a log-scaled 2D histogram of true versus predicted voltage magnitudes.

    Dashed lines mark 0.05 p.u. inside each axis limit. The Pearson correlation is
    annotated on the plot. The figure is saved as
    ``mass_correlation_density_voltage_{prediction_dir}.png`` and then shown.

    Args:
        pf_node: Mapping or DataFrame with ``Vm`` (true) and
            ``Vm_pred_corrected`` (predicted) columns, in p.u.
        prediction_dir: Suffix used in the saved file name.
        label_plot: Plot title.
        x_min: Lower x-axis limit.
        y_min: Lower y-axis limit.
        x_max: Upper x-axis limit.
        y_max: Upper y-axis limit.
    """
    # TODO refactor if we pass by parameters a few more plot deets we can use plot_mass_correlation_density for both
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
