import networkx as nx
from gridfm_graphkit.training.loss import PBELoss
from gridfm_graphkit.datasets.globals import PQ, PV, REF
import matplotlib.pyplot as plt


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
