from gridfm_graphkit.datasets.globals import PD, QD, PG, QG, VM, VA, G, B, REF

import torch.nn.functional as F
import torch
from torch_geometric.utils import to_torch_coo_tensor
import torch.nn as nn

from torch_geometric.utils import to_dense_adj
from collections import deque


class MaskedMSELoss(nn.Module):
    """
    Mean Squared Error loss computed only on masked elements.
    """

    def __init__(self, reduction="mean"):
        super(MaskedMSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, target, edge_index=None, edge_attr=None, mask=None, x=None):
        loss = F.mse_loss(pred[mask], target[mask], reduction=self.reduction)
        return {"loss": loss, "Masked MSE loss": loss.detach()}


class MSELoss(nn.Module):
    """Standard Mean Squared Error loss."""

    def __init__(self, reduction="mean"):
        super(MSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, target, edge_index=None, edge_attr=None, mask=None, x=None):
        loss = F.mse_loss(pred, target, reduction=self.reduction)
        return {"loss": loss, "MSE loss": loss.detach()}


class SCELoss(nn.Module):
    """Scaled Cosine Error Loss with optional masking and normalization."""

    def __init__(self, alpha=3):
        super(SCELoss, self).__init__()
        self.alpha = alpha

    def forward(self, pred, target, edge_index=None, edge_attr=None, mask=None, x=None):
        if mask is not None:
            pred = F.normalize(pred[mask], p=2, dim=-1)
            target = F.normalize(target[mask], p=2, dim=-1)
        else:
            pred = F.normalize(pred, p=2, dim=-1)
            target = F.normalize(target, p=2, dim=-1)

        loss = ((1 - (pred * target).sum(dim=-1)).pow(self.alpha)).mean()

        return {
            "loss": loss,
            "SCE loss": loss.detach(),
        }


class PBELoss(nn.Module):
    """
    Loss based on the Power Balance Equations.
    """

    def __init__(self, visualization=False):
        super(PBELoss, self).__init__()

        self.visualization = visualization

    def forward(self, pred, target, edge_index, edge_attr, mask, x=None):
        # Create a temporary copy of pred to avoid modifying it
        temp_pred = pred.clone()

        # If a value is not masked, then use the original one
        unmasked = ~mask
        temp_pred[unmasked] = target[unmasked]

        # Voltage magnitudes and angles
        V_m = temp_pred[:, VM]  # Voltage magnitudes
        V_a = temp_pred[:, VA]  # Voltage angles

        # Compute the complex voltage vector V
        V = V_m * torch.exp(1j * V_a)

        # Compute the conjugate of V
        V_conj = torch.conj(V)

        # Extract edge attributes for Y_bus
        edge_complex = edge_attr[:, G] + 1j * edge_attr[:, B]

        # Construct sparse admittance matrix (real and imaginary parts separately)
        Y_bus_sparse = to_torch_coo_tensor(
            edge_index,
            edge_complex,
            size=(target.size(0), target.size(0)),
        )

        # Conjugate of the admittance matrix
        Y_bus_conj = torch.conj(Y_bus_sparse)

        # Compute the complex power injection S_injection
        S_injection = torch.diag(V) @ Y_bus_conj @ V_conj

        # Compute net power balance
        net_P = temp_pred[:, PG] - temp_pred[:, PD]
        net_Q = temp_pred[:, QG] - temp_pred[:, QD]
        S_net_power_balance = net_P + 1j * net_Q

        # Power balance loss
        loss = torch.mean(
            torch.abs(S_net_power_balance - S_injection),
        )  # Mean of absolute complex power value

        real_loss_power = torch.mean(
            torch.abs(torch.real(S_net_power_balance - S_injection)),
        )
        imag_loss_power = torch.mean(
            torch.abs(torch.imag(S_net_power_balance - S_injection)),
        )
        if self.visualization:
            return {
                "loss": loss,
                "Power loss in p.u.": loss.detach(),
                "Active Power Loss in p.u.": real_loss_power.detach(),
                "Reactive Power Loss in p.u.": imag_loss_power.detach(),
                "Nodal Active Power Loss in p.u.": torch.abs(
                    torch.real(S_net_power_balance - S_injection),
                ),
                "Nodal Reactive Power Loss in p.u.": torch.abs(
                    torch.imag(S_net_power_balance - S_injection),
                ),
            }
        else:
            return {
                "loss": loss,
                "Power loss in p.u.": loss.detach(),
                "Active Power Loss in p.u.": real_loss_power.detach(),
                "Reactive Power Loss in p.u.": imag_loss_power.detach(),
            }


class MixedLoss(nn.Module):
    """
    Combines multiple loss functions with weighted sum.

    Args:
        loss_functions (list[nn.Module]): List of loss functions.
        weights (list[float]): Corresponding weights for each loss function.
    """

    def __init__(self, loss_functions, weights):
        super(MixedLoss, self).__init__()

        if len(loss_functions) != len(weights):
            raise ValueError(
                "The number of loss functions must match the number of weights.",
            )

        self.loss_functions = nn.ModuleList(loss_functions)
        self.weights = weights

    def forward(self, pred, target, edge_index=None, edge_attr=None, mask=None, x=None):
        """
        Compute the weighted sum of all specified losses.

        Parameters:

        - pred: Predictions.
        - target: Ground truth.
        - edge_index: Optional edge index for graph-based losses.
        - edge_attr: Optional edge attributes for graph-based losses.
        - mask: Optional mask to filter the inputs for certain losses.

        Returns:
        - A dictionary with the total loss and individual losses.
        """
        total_loss = 0.0
        loss_details = {}

        for i, loss_fn in enumerate(self.loss_functions):
            loss_output = loss_fn(
                pred,
                target,
                edge_index=edge_index,
                edge_attr=edge_attr,
                mask=mask,
                x=x,
            )

            # Assume each loss function returns a dictionary with a "loss" key
            individual_loss = loss_output.pop("loss")
            weighted_loss = self.weights[i] * individual_loss

            total_loss += weighted_loss

            # Add other keys from the loss output to the details
            for key, val in loss_output.items():
                loss_details[key] = val

        loss_details["loss"] = total_loss
        return loss_details


class VLDLoss(nn.Module):
    """
    Global connectivity / energization constraint to the REF bus,
    considering both failed nodes (low Vm) and failed edges (|G|,|B| below threshold).

    Args:
    voltage_threshold: Below this, node is considered failed.
    edge_GB_threshold: Edge is considered failed if both |G| and |B| are below this threshold (in normalized units).
    beta: Sharpness of sigmoid in connectivity update.
    penalty_scale: Global scale for this loss.
    margin_factor: margin = margin_factor * voltage_threshold.
    safety_margin: Extra hops added to max BFS distance for K.
    undirected: If True, treat connectivity as undirected.
    max_K_cap: Upper cap on K to avoid extreme propagation depth.

    Training procedure
    - Builds a "healthy" connectivity graph using only edges with
      sufficiently large |G| and |B|.
    - Uses BFS from REF on this healthy-edge graph to choose K per graph.
    - Propagates a soft connectivity score h from REF through:
        * healthy edges, and
        * healthy nodes.
    - Penalizes:
        * reachable nodes: high V but h ~ 0 (disconnected but energized)
        * unreachable nodes: any non-zero V (must be ~0 if islanded).
    """

    def __init__(self, visualization=False):
        super(VLDLoss, self).__init__()

        self.visualization = visualization

        self.voltage_threshold = 1e-3
        self.edge_GB_threshold = 1e-6
        self.beta = 8.0
        self.penalty_scale = 1.0
        self.margin_factor = 0.5
        self.safety_margin = 1
        self.undirected = True
        self.max_K_cap = 64
        self.INF = 10**9  # sentinel for unreachable


        """
        self.voltage_threshold = args.voltage_loss_detector.voltage_threshold
        self.edge_GB_threshold = args.voltage_loss_detector.edge_GB_threshold
        self.beta = args.voltage_loss_detector.beta
        self.penalty_scale = args.voltage_loss_detector.penalty_scale
        self.margin_factor = args.voltage_loss_detector.margin_factor
        self.safety_margin = args.voltage_loss_detector.safety_margin
        self.undirected = args.voltage_loss_detector.undirected
        self.max_K_cap = args.voltage_loss_detector.max_K_caps
        """

    # ---------- build healthy-edge adjacency / edge_index ----------

    def healthy_edge_mask(self, edge_attr):
        """
        Decide which edges are "healthy" based on G,B magnitude.

        Args:
            edge_attr: (E, F_edge), with G,B at indices G,B.

        Returns:
            mask: (E,) bool tensor, True for healthy edges.
        """
        G_vals = edge_attr[:, G]
        B_vals = edge_attr[:, B]
        # Edge is failed if both |G| and |B| are below threshold
        healthy = (G_vals.abs() >= self.edge_GB_threshold) | (
            B_vals.abs() >= self.edge_GB_threshold
        )
        return healthy

    def pruned_edge_index(self, edge_index, edge_attr):
        """
        Keep only healthy edges for connectivity graph.
        """
        healthy = self.healthy_edge_mask(edge_attr)  # (E,)
        return edge_index[:, healthy]

    # ---------- BFS utilities ----------

    def bfs_distance_from_ref(self, edge_index, num_nodes, ref_idx: int):
        """
        Unweighted BFS distances from REF node on the connectivity graph.

        Returns:
            dist: (N,) tensor with hop distances (INF for unreachable nodes).
        """
        row, col = edge_index
        adj = [[] for _ in range(num_nodes)]
        # undirected BFS on connectivity
        for u, v in zip(row.tolist(), col.tolist()):
            adj[u].append(v)
            adj[v].append(u)

        dist = [self.INF] * num_nodes
        dist[ref_idx] = 0
        q = deque([ref_idx])

        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == self.INF:
                    dist[v] = dist[u] + 1
                    q.append(v)

        return torch.tensor(dist, dtype=torch.long, device=edge_index.device)

    def choose_K_for_graph(self, edge_index, num_nodes, ref_idx):
        """
        Per-graph K from BFS on healthy-edge graph.
        """
        if edge_index.numel() == 0:
            # no healthy edges: only REF is trivially connected
            dist = torch.full((num_nodes,), self.INF, dtype=torch.long, device=edge_index.device)
            dist[ref_idx] = 0
            reachable = dist < self.INF
            return 1, dist, reachable

        dist = self.bfs_distance_from_ref(edge_index, num_nodes, ref_idx)
        reachable = dist < self.INF

        if reachable.sum() <= 1:
            return 1, dist, reachable

        max_dist = dist[reachable].max().item()
        K = max_dist + self.safety_margin
        K = max(1, min(K, self.max_K_cap))
        return K, dist, reachable

    # ---------- adjacency for propagation (healthy edges only) ----------

    def build_normalized_adj(self, edge_index, num_nodes):
        A = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0].float()  # (N, N)

        if self.undirected:
            A = ((A + A.t()) > 0).float()

        deg = A.sum(dim=1, keepdim=True)  # (N, 1)
        deg = torch.clamp(deg, min=1.0)
        A_norm = A / deg
        return A_norm

    # ---------- main forward ----------

    def forward(self, pred, target, edge_index, edge_attr, mask, x):
        """
        Args:
            pred: (N, F) predicted node features.
            target: (N, F) ground truth node features.
            edge_index: (2, E) full graph edges (can include failed edges).
            edge_attr: (E, F_edge) edge features with G,B.
            mask: (N, M) boolean mask used to form hybrid predictions.

        Returns:
            dict with:
                - "loss": scalar Voltage Detector loss
                - "Voltage Detector Loss": detached scalar
        """
        device = pred.device

        # 1) Hybrid prediction: target on unmasked, pred on masked
        temp_pred = pred.clone()
        unmasked = ~mask
        temp_pred[unmasked] = target[unmasked]

        N = temp_pred.size(0)

        # 2) Voltage and REF indicator
        Vm = temp_pred[:, VM]             # (N,)
        ref_indicator = x[:, REF] # (N,)
        ref_idx = torch.argmax(ref_indicator).item()

        # 3) Build connectivity graph using only healthy edges
        edge_index_healthy = self.pruned_edge_index(edge_index, edge_attr)

        # 4) Per-graph K from BFS on healthy-edge graph
        K, dist, reachable = self.choose_K_for_graph(edge_index_healthy, N, ref_idx)
        unreachable = ~reachable

        # 5) Node health from voltage
        failed = (Vm < self.voltage_threshold).float()  # (N,)
        healthy_nodes = 1.0 - failed                    # (N,)

        # 6) Degree-normalized adjacency on healthy-edge graph
        A_norm = self.build_normalized_adj(edge_index_healthy, N).to(device)  # (N, N)

        # 7) Initial connectivity: only REF is connected
        h = torch.zeros_like(healthy_nodes)
        h[ref_idx] = 1.0
        h = h.unsqueeze(-1)  # (N, 1)

        # 8) Propagate connectivity K steps through healthy nodes & edges
        for _ in range(K):
            neigh = torch.matmul(A_norm.t(), h)  # (N, 1)
            # soft AND with node health
            h = torch.sigmoid(self.beta * (neigh * healthy_nodes.unsqueeze(-1)))
            h[ref_idx] = 1.0
        h = h.squeeze(-1)  # (N,)

        # 9) Hinge margin on voltage
        margin = self.margin_factor * self.voltage_threshold
        excess = torch.clamp(Vm - margin, min=0.0)

        # 10) Reachable nodes: disconnected-but-energized violations
        reachable_violation_mask = reachable & (h < 0.5) & (excess > 0.0)
        violation_reach = excess[reachable_violation_mask] ** 2
        loss_reach = (
            violation_reach.mean()
            if violation_reach.numel() > 0
            else torch.tensor(0.0, device=device)
        )

        # 11) Unreachable (islanded) nodes: any non-zero voltage is a violation
        excess_unreach = excess[unreachable]
        loss_unreach = (
            (excess_unreach**2).mean()
            if excess_unreach.numel() > 0
            else torch.tensor(0.0, device=device)
        )

        loss = self.penalty_scale * (loss_reach + loss_unreach)

        if self.visualization:
            return {
                "loss": loss,
                "Voltage Detector Loss": loss.detach(),
                "Loss of reachable nodes.": loss_reach.detach(),
                "Loss of unreachable nodes.": loss_unreach.detach()
            }
        else:
            return {
                "loss": loss,
                "Voltage Detector Loss": loss.detach(),
                "Loss of reachable nodes.": loss_reach.detach(),
                "Loss of unreachable nodes.": loss_unreach.detach(),
                "K_used": torch.tensor(K, device=device)
            }
