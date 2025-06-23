from gridfm_graphkit.datasets.globals import PD, QD, PG, QG, VM, VA, G, B

import torch.nn.functional as F
import torch
from torch_geometric.utils import to_torch_coo_tensor
import torch.nn as nn


class MaskedMSELoss(nn.Module):
    """
    Mean Squared Error loss computed only on masked elements.
    """

    def __init__(self, reduction="mean"):
        super(MaskedMSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, target, edge_index=None, edge_attr=None, mask=None):
        loss = F.mse_loss(pred[mask], target[mask], reduction=self.reduction)
        return {"loss": loss, "Masked MSE loss": loss.item()}


class MSELoss(nn.Module):
    """Standard Mean Squared Error loss."""

    def __init__(self, reduction="mean"):
        super(MSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, target, edge_index=None, edge_attr=None, mask=None):
        loss = F.mse_loss(pred, target, reduction=self.reduction)
        return {"loss": loss, "MSE loss": loss.item()}


class SCELoss(nn.Module):
    """Scaled Cosine Error Loss with optional masking and normalization."""

    def __init__(self, alpha=3):
        super(SCELoss, self).__init__()
        self.alpha = alpha

    def forward(self, pred, target, edge_index=None, edge_attr=None, mask=None):
        if mask is not None:
            pred = F.normalize(pred[mask], p=2, dim=-1)
            target = F.normalize(target[mask], p=2, dim=-1)
        else:
            pred = F.normalize(pred, p=2, dim=-1)
            target = F.normalize(target, p=2, dim=-1)

        loss = ((1 - (pred * target).sum(dim=-1)).pow(self.alpha)).mean()

        return {
            "loss": loss,
            "SCE loss": loss.item(),
        }


class PBELoss(nn.Module):
    """
    Loss based on the Power Balance Equations.
    """

    def __init__(self, visualization=False):
        super(PBELoss, self).__init__()

        self.visualization = visualization

    def forward(self, pred, target, edge_index, edge_attr, mask):
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
                "Power power loss in p.u.": loss.item(),
                "Active Power Loss in p.u.": real_loss_power.item(),
                "Reactive Power Loss in p.u.": imag_loss_power.item(),
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
                "Power power loss in p.u.": loss.item(),
                "Active Power Loss in p.u.": real_loss_power.item(),
                "Reactive Power Loss in p.u.": imag_loss_power.item(),
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

    def forward(self, pred, target, edge_index=None, edge_attr=None, mask=None):
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
