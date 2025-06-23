from gridfm_graphkit.utils.loss import PBELoss
from gridfm_graphkit.datasets.globals import PQ, PV, REF

import torch
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import copy


def visualize_error(data_point, model, baseMVA, device):
    data_point = copy.deepcopy(data_point)
    active_loss = None
    loss = PBELoss(visualization=True)

    # Inference of one data point
    model.eval()
    with torch.no_grad():
        data_point = data_point.to(device)
        mask_value_expanded = model.mask_value.expand(data_point.x.shape[0], -1)
        data_point.x[:, : data_point.mask.shape[1]][data_point.mask] = (
            mask_value_expanded[data_point.mask]
        )
        output = model(
            data_point.x,
            data_point.pe,
            data_point.edge_index,
            data_point.edge_attr,
            torch.zeros(data_point.x.shape[0], dtype=int).to(device),
        )
        loss_dict = loss(
            output,
            data_point.y,
            data_point.edge_index,
            data_point.edge_attr,
            data_point.mask,
        )
        active_loss = loss_dict["Nodal Active Power Loss in p.u."]
        active_loss = active_loss.cpu() * baseMVA

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
    model,
    quantity,
    quantity_name,
    unit,
    node_normalizer,
    cmap,
    device,
):
    """
    Visualizes a heatmap of a specified quantity (VM, PD, QD, PG, QG, VA) for a given dataset and model.

    Parameters:
        data_point: Power grid data.
        model: The trained model used for inference.
        quantity: The quantity to visualize (e.g., VM, PD, QD, PG, QG, VA).
    """
    data_point = copy.deepcopy(data_point)
    mask_PQ = data_point.x[:, PQ] == 1
    mask_PV = data_point.x[:, PV] == 1
    mask_REF = data_point.x[:, REF] == 1
    gt_values = data_point.y[:, quantity]  # Extract ground truth values

    # Inference of one data point
    model.eval()
    with torch.no_grad():
        data_point = data_point.to(device)
        mask_value_expanded = model.mask_value.expand(data_point.x.shape[0], -1)
        data_point.x[:, : data_point.mask.shape[1]][data_point.mask] = (
            mask_value_expanded[data_point.mask]
        )
        output = model(
            data_point.x,
            data_point.pe,
            data_point.edge_index,
            data_point.edge_attr,
            torch.zeros(data_point.x.shape[0], dtype=int).to(device),
        )
        output = node_normalizer.inverse_transform(output)
        denormalized_gt = node_normalizer.inverse_transform(data_point.y)

        gt_values = denormalized_gt[:, quantity]
        predicted_values = output[:, quantity]
        predicted_values[~data_point.mask[:, quantity]] = denormalized_gt[
            ~data_point.mask[:, quantity],
            quantity,
        ]

    num_nodes = data_point.x.shape[0]
    predicted_values = predicted_values.cpu()
    gt_values = gt_values.cpu()

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
    cmap = cmap
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
    ax.set_title(f"Input grid {quantity_name}", fontsize=14, fontweight="bold")

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
    ax.set_title(f"Masked grid {quantity_name}", fontsize=14, fontweight="bold")

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
    ax.set_title(f"Reconstructed grid {quantity_name}", fontsize=14, fontweight="bold")

    for spine in ax.spines.values():
        spine.set_linewidth(2)  # Adjust thickness

    # Colorbar placement
    cbar_ax = fig.add_axes([0.93, 0.1, 0.02, 0.8])
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cbar_ax)
    cbar.set_label(f"{quantity_name} ({unit})", fontsize=12)
    cbar.ax.tick_params(labelsize=12)

    plt.subplots_adjust(right=0.9)
    plt.show()
