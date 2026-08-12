import torch.nn.functional as F
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from gridfm_graphkit.io.registries import LOSS_REGISTRY
from torch_scatter import scatter_add
from torch_geometric.utils import to_torch_coo_tensor

from gridfm_graphkit.datasets.globals import (
    # Bus feature indices
    QG_H,
    VM_H,
    VA_H,
    QD_H,
    PD_H,
    GS,
    BS,
    # Output feature indices
    VM_OUT,
    VA_OUT,
    QG_OUT,
    PG_OUT,
    PD_OUT,
    QD_OUT,
    # Generator feature indices
    PG_H,
    # Edge feature indices
    YFF_TT_R,
    YFF_TT_I,
    YFT_TF_R,
    YFT_TF_I,
    # Qg Limits
    MIN_QG_H,
    MAX_QG_H,
)


class BaseLoss(nn.Module, ABC):
    """
    Abstract base class for all custom loss functions.
    """

    @abstractmethod
    def forward(
        self,
        pred,
        target,
        edge_index=None,
        edge_attr=None,
        mask=None,
        model=None,
        x_dict=None,
    ):
        """
        Compute the loss.

        Parameters:
        - pred: Predictions.
        - target: Ground truth.
        - edge_index: Optional edge index for graph-based losses.
        - edge_attr: Optional edge attributes for graph-based losses.
        - mask: Optional mask to filter the inputs for certain losses.
        - model: Optional model reference for accessing internal states.

        Returns:
        - A dictionary with the total loss and any additional metrics.
        """
        pass


@LOSS_REGISTRY.register("MaskedMSE")
class MaskedMSELoss(BaseLoss):
    """
    Mean Squared Error loss computed only on masked elements.
    """

    def __init__(self, loss_args, args):
        super(MaskedMSELoss, self).__init__()
        self.reduction = "mean"

    def forward(
        self,
        pred,
        target,
        edge_index=None,
        edge_attr=None,
        mask=None,
        model=None,
        x_dict=None,
    ):
        loss = F.mse_loss(pred[mask], target[mask], reduction=self.reduction)
        return {"loss": loss, "Masked MSE loss": loss.detach()}


@LOSS_REGISTRY.register("MaskedGenMSE")
class MaskedGenMSE(torch.nn.Module):
    """Compute MSE on generator targets restricted to generator mask entries."""

    def __init__(self, loss_args, args):
        super().__init__()
        self.reduction = "mean"

    def forward(
        self,
        pred_dict,
        target_dict,
        edge_index,
        edge_attr,
        mask_dict,
        model=None,
        x_dict=None,
    ):
        gen_pred = pred_dict["gen"][:, : (PG_H + 1)]
        gen_target = target_dict["gen"][:, : (PG_H + 1)]
        mask = mask_dict["gen"][:, : (PG_H + 1)]
        loss = F.mse_loss(
            gen_pred[mask],
            gen_target[mask],
            reduction=self.reduction,
        )
        return {"loss": loss, "Masked generator MSE loss": loss.detach()}


@LOSS_REGISTRY.register("MaskedBusMSE")
class MaskedBusMSE(torch.nn.Module):
    """Compute MSE on selected bus targets, respecting task-specific output columns."""

    def __init__(self, loss_args, args):
        super().__init__()
        self.reduction = "mean"
        self.args = args

    def forward(
        self,
        pred_dict,
        target_dict,
        edge_index,
        edge_attr,
        mask_dict,
        model=None,
        x_dict=None,
    ):
        if self.args.task == "OptimalPowerFlow":
            pred_cols = [VM_OUT, VA_OUT, QG_OUT]
            target_cols = [VM_H, VA_H, QG_H]
        else:
            pred_cols = [VM_OUT, VA_OUT]
            target_cols = [VM_H, VA_H]

        pred_bus = pred_dict["bus"][:, pred_cols]  # shape: [N, 3]
        target_bus = target_dict["bus"][:, target_cols]

        mask = mask_dict["bus"][:, target_cols]

        loss = F.mse_loss(
            pred_bus[mask],
            target_bus[mask],
            reduction=self.reduction,
        )
        return {"loss": loss, "Masked bus MSE loss": loss.detach()}


@LOSS_REGISTRY.register("MaskedReconstructionMSE")
class MaskedReconstructionMSE(BaseLoss):
    """Unified masked MSE over bus-level quantities [VM, VA, PG, QG, PD, QD].

    Mirrors the homogeneous reference MaskedMSE by combining bus predictions
    and aggregated generator PG into a single prediction/target/mask tensor.
    PG targets are aggregated from generator ground truth onto buses via
    scatter_add; the bus-level PG mask is True when any generator at the bus
    is masked, indicating that the model must reconstruct that quantity.

    Replaces the separate MaskedBusMSE + MaskedGenMSE pair.
    Requires output_bus_dim >= 6 so the bus head predicts
    [VM, VA, PG, QG, PD, QD].
    """

    def __init__(self, loss_args, args):
        super().__init__()
        self.reduction = "mean"

    def forward(
        self,
        pred_dict,
        target_dict,
        edge_index_dict,
        edge_attr_dict,
        mask_dict,
        model=None,
        x_dict=None,
    ):
        pred_bus = pred_dict["bus"]
        target_bus = target_dict["bus"]
        num_bus = target_bus.size(0)
        gen_to_bus_ei = edge_index_dict[("gen", "connected_to", "bus")]

        # --- Build target: [VM, VA, PG_agg, QG, PD, QD] ---
        target_pg_agg = scatter_add(
            target_dict["gen"][:, PG_H],
            gen_to_bus_ei[1],
            dim=0,
            dim_size=num_bus,
        )
        target = torch.stack(
            [
                target_bus[:, VM_H],
                target_bus[:, VA_H],
                target_pg_agg,
                target_bus[:, QG_H],
                target_bus[:, PD_H],
                target_bus[:, QD_H],
            ],
            dim=1,
        )

        # --- Build mask: [N_bus, 6] ---
        # PG bus-level mask: True if any generator at the bus has PG masked
        gen_pg_masked = mask_dict["gen"][:, PG_H].float()
        any_gen_masked = (
            scatter_add(
                gen_pg_masked,
                gen_to_bus_ei[1],
                dim=0,
                dim_size=num_bus,
            )
            > 0
        )

        mask = torch.stack(
            [
                mask_dict["bus"][:, VM_H],
                mask_dict["bus"][:, VA_H],
                any_gen_masked,
                mask_dict["bus"][:, QG_H],
                mask_dict["bus"][:, PD_H],
                mask_dict["bus"][:, QD_H],
            ],
            dim=1,
        )

        # --- Prediction: [VM, VA, PG, QG, PD, QD] from bus head ---
        pred = pred_bus[:, [VM_OUT, VA_OUT, PG_OUT, QG_OUT, PD_OUT, QD_OUT]]

        loss = F.mse_loss(pred[mask], target[mask], reduction=self.reduction)
        return {"loss": loss, "Masked reconstruction MSE loss": loss.detach()}


@LOSS_REGISTRY.register("MSE")
class MSELoss(BaseLoss):
    """Standard Mean Squared Error loss."""

    def __init__(self, loss_args, args):
        super(MSELoss, self).__init__()
        self.reduction = "mean"

    def forward(
        self,
        pred,
        target,
        edge_index=None,
        edge_attr=None,
        mask=None,
        model=None,
        x_dict=None,
    ):
        loss = F.mse_loss(pred, target, reduction=self.reduction)
        return {"loss": loss, "MSE loss": loss.detach()}


class MixedLoss(BaseLoss):
    """
    Combines multiple loss functions with weighted sum.

    Args:
        loss_functions (list[nn.Module]): List of loss functions.
        weights (list[float]): Corresponding weights for each loss function.
        warmup_indices (list[int], optional): Indices into `loss_functions` whose
            weight should ramp linearly from 0 up to its target value over
            `warmup_epochs` epochs, instead of being constant. Used to delay
            physics-loss terms until the model has learned a stable supervised
            baseline. Defaults to no warmup (all weights constant at their target).
        warmup_epochs (int, optional): Number of epochs over which `warmup_indices`
            weights ramp up. 0 disables warmup (default, unchanged behavior).
    """

    def __init__(self, loss_functions, weights, warmup_indices=None, warmup_epochs=0):
        super(MixedLoss, self).__init__()

        if len(loss_functions) != len(weights):
            raise ValueError(
                "The number of loss functions must match the number of weights.",
            )

        self.loss_functions = nn.ModuleList(loss_functions)
        self.target_weights = list(weights)
        self.weights = list(weights)
        self.warmup_indices = warmup_indices or []
        self.warmup_epochs = warmup_epochs

    def set_epoch(self, epoch):
        """Update ramped weights for the given (0-indexed) training epoch."""
        if not self.warmup_indices or self.warmup_epochs <= 0:
            return
        alpha = min(1.0, (epoch + 1) / self.warmup_epochs)
        for i in self.warmup_indices:
            self.weights[i] = self.target_weights[i] * alpha

    def forward(
        self,
        pred,
        target,
        edge_index=None,
        edge_attr=None,
        mask=None,
        model=None,
        x_dict=None,
    ):
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
                edge_index,
                edge_attr,
                mask,
                model,
                x_dict,
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


@LOSS_REGISTRY.register("LayeredWeightedPhysics")
class LayeredWeightedPhysicsLoss(BaseLoss):
    """Combine intermediate physics residuals using normalized geometric weights."""

    def __init__(self, loss_args, args) -> None:
        super().__init__()
        self.base_weight = loss_args.base_weight

    def forward(
        self,
        pred,
        target,
        edge_index=None,
        edge_attr=None,
        mask=None,
        model=None,
        x_dict=None,
    ):
        total_loss = 0.0
        loss_details = {}

        layer_keys = sorted(model.layer_residuals.keys())
        L = len(layer_keys)

        # Compute raw weights (geometric decay)
        raw_weights = [self.base_weight ** (L - idx - 1) for idx in range(L)]

        # Normalize so weights sum to 1
        weight_sum = sum(raw_weights)
        norm_weights = [w / weight_sum for w in raw_weights]

        for key, weight in zip(layer_keys, norm_weights):
            residual = model.layer_residuals[key]
            total_loss = total_loss + weight * residual
            loss_details[f"layer_{key}_residual"] = residual.item()
            loss_details[f"layer_{key}_weight"] = weight

        loss_details["loss"] = total_loss
        loss_details["Layered Weighted Physics Loss"] = total_loss.item()
        return loss_details


@LOSS_REGISTRY.register("LossPerDim")
class LossPerDim(BaseLoss):
    """Compute MAE/MSE for one named physical dimension of bus outputs."""

    def __init__(self, loss_args, args):
        super(LossPerDim, self).__init__()
        self.reduction = "mean"
        self.loss_str = loss_args.loss_str
        self.dim = loss_args.dim
        if self.dim not in ["VM", "VA", "P_in", "Q_in"]:
            raise ValueError(
                f"LossPerDim initialized with not valid dim: {self.dim}",
            )

        elif self.loss_str not in ["MAE", "MSE"]:
            raise ValueError(
                f"LossPerDim initialized with not valid loss_str: {self.loss_str}",
            )

    def forward(
        self,
        pred_dict,
        target_dict,
        edge_index,
        edge_attr,
        mask_dict,
        model=None,
        x_dict=None,
    ):
        if self.dim == "VM":
            temp_pred = pred_dict["bus"][:, VM_OUT]
            temp_target = target_dict["bus"][:, VM_H]
        elif self.dim == "VA":
            temp_pred = pred_dict["bus"][:, VA_OUT]
            temp_target = target_dict["bus"][:, VA_H]
        elif self.dim == "P_in":
            temp_pred = pred_dict["bus"][:, PG_OUT]
            num_bus = temp_pred.size(0)
            gen_to_bus_index = edge_index[("gen", "connected_to", "bus")]
            temp_gen = scatter_add(
                target_dict["gen"][:, PG_H],
                gen_to_bus_index[1, :],
                dim=0,
                dim_size=num_bus,
            )
            temp_target = temp_gen - target_dict["bus"][:, PD_H]
        elif self.dim == "Q_in":
            temp_pred = pred_dict["bus"][:, QG_OUT]
            temp_target = target_dict["bus"][:, QG_H] - target_dict["bus"][:, QD_H]

        mse_loss = F.mse_loss(temp_pred, temp_target, reduction=self.reduction)
        mae_loss = F.l1_loss(temp_pred, temp_target, reduction=self.reduction)

        loss = mse_loss if self.loss_str == "mse" else mae_loss
        return {
            "loss": loss,
            f"MSE loss {self.dim}": mse_loss.detach(),
            f"MAE loss {self.dim}": mae_loss.detach(),
        }


@LOSS_REGISTRY.register("PBE")
class PBELoss(BaseLoss):
    """
    Loss based on the Power Balance Equations.

    Adapted for the heterogeneous graph convention: predictions and targets
    are passed as dicts (``{"bus": …, "gen": …}``).  Generator active power
    is aggregated onto bus nodes via the ``(gen, connected_to, bus)`` edge
    index before computing the power balance.
    """

    def __init__(self, loss_args, args):
        super(PBELoss, self).__init__()
        self.visualization = getattr(loss_args, "visualization", False)

    def forward(
        self,
        pred_dict,
        target_dict,
        edge_index_dict,
        edge_attr_dict,
        mask_dict,
        model=None,
        x_dict=None,
    ):
        pred_bus = pred_dict["bus"]  # [N_bus, output_bus_dim]
        target_bus = target_dict["bus"]  # [N_bus, bus_feat_dim]
        num_bus = target_bus.size(0)

        bus_edge_index = edge_index_dict[("bus", "connects", "bus")]
        bus_edge_attr = edge_attr_dict[("bus", "connects", "bus")]
        mask_bus = mask_dict["bus"]

        # --- Clamp known values to ground truth ---
        # In power flow, certain variables are "known" (unmasked) at each
        # bus type (e.g. VM at PV buses, VA at REF).  The model only needs
        # to predict *masked* unknowns; for everything else we substitute
        # the ground truth so that errors in non-target outputs do not
        # pollute the physics loss.  This matches the reference's
        # ``temp_pred[unmasked] = target[unmasked]`` convention.

        Vm_pred = pred_bus[:, VM_OUT]
        Va_pred = pred_bus[:, VA_OUT]
        Vm_target = target_bus[:, VM_H]
        Va_target = target_bus[:, VA_H]

        mask_Vm = mask_bus[:, VM_H]
        mask_Va = mask_bus[:, VA_H]

        V_m = torch.where(mask_Vm, Vm_pred, Vm_target)
        V_a = torch.where(mask_Va, Va_pred, Va_target)

        # Complex voltage
        V = V_m * torch.exp(1j * V_a)
        V_conj = torch.conj(V)

        # --- Admittance matrix from bus-bus edge attrs ---
        # The Y-bus matrix has off-diagonal AND diagonal entries.
        #
        # Off-diagonal: Y[from][to] = Yft, Y[to][from] = Ytf, stored in the
        # YFT_TF columns of the edge attributes.
        #
        # Diagonal: Y[k][k] = sum of Yff/Ytt for all branches at bus k.
        # The dataset stores Yff (forward edges) and Ytt (reverse edges) in
        # the YFF_TT columns.  For every edge, YFF_TT at the *source* bus
        # gives that branch's diagonal contribution at the source.  Summing
        # over all edges with source == k yields the full branch-diagonal.
        #
        # The reference project loads a pre-built Y-bus (y_bus_data.parquet)
        # that includes self-loops for diagonal entries.  Here we reconstruct
        # the same structure from per-branch pi-model parameters.

        # Off-diagonal admittance values
        edge_offdiag = bus_edge_attr[:, YFT_TF_R] + 1j * bus_edge_attr[:, YFT_TF_I]

        # Diagonal: aggregate Yff/Ytt to source bus of each edge
        Y_diag_r = scatter_add(
            bus_edge_attr[:, YFF_TT_R],
            bus_edge_index[0],
            dim=0,
            dim_size=num_bus,
        )
        Y_diag_i = scatter_add(
            bus_edge_attr[:, YFF_TT_I],
            bus_edge_index[0],
            dim=0,
            dim_size=num_bus,
        )
        Y_diag = Y_diag_r + 1j * Y_diag_i

        # Add bus shunt admittance (Gs + jBs) to the diagonal
        if x_dict is not None:
            bus_orig = x_dict["bus"]
            Y_diag = Y_diag + bus_orig[:, GS] + 1j * bus_orig[:, BS]

        # Build complete Y-bus: off-diagonal edges + self-loops for diagonal
        diag_idx = torch.arange(num_bus, device=bus_edge_index.device)
        full_edge_index = torch.cat(
            [bus_edge_index, torch.stack([diag_idx, diag_idx])],
            dim=1,
        )
        full_edge_values = torch.cat([edge_offdiag, Y_diag])

        Y_bus_sparse = to_torch_coo_tensor(
            full_edge_index,
            full_edge_values,
            size=(num_bus, num_bus),
        )
        Y_bus_conj = torch.conj(Y_bus_sparse)

        # Complex power injection:  S_inj = V .* (conj(Y) @ conj(V))
        S_injection = V * (Y_bus_conj @ V_conj)

        # --- Net power from predictions/targets ---
        # Pg: use bus head prediction where masked, ground truth where known.
        # Ground truth is aggregated from generator targets onto buses.
        gen_to_bus_ei = edge_index_dict[("gen", "connected_to", "bus")]
        target_pg_agg = scatter_add(
            target_dict["gen"][:, PG_H],
            gen_to_bus_ei[1],
            dim=0,
            dim_size=num_bus,
        )
        gen_pg_masked = mask_dict["gen"][:, PG_H].float()
        any_gen_masked = (
            scatter_add(
                gen_pg_masked,
                gen_to_bus_ei[1],
                dim=0,
                dim_size=num_bus,
            )
            > 0
        )
        Pg_per_bus = torch.where(any_gen_masked, pred_bus[:, PG_OUT], target_pg_agg)

        # Pd, Qd, Qg: same clamp-to-ground-truth logic.  The size guard
        # (``pred_bus.size(1) > *_OUT``) handles models with a narrow bus
        # head (e.g. output_bus_dim=4) that don't predict PD/QD/QG; in that
        # case the target is always used.
        if pred_bus.size(1) > PD_OUT:
            Pd = torch.where(
                mask_bus[:, PD_H],
                pred_bus[:, PD_OUT],
                target_bus[:, PD_H],
            )
        else:
            Pd = target_bus[:, PD_H]
        if pred_bus.size(1) > QD_OUT:
            Qd = torch.where(
                mask_bus[:, QD_H],
                pred_bus[:, QD_OUT],
                target_bus[:, QD_H],
            )
        else:
            Qd = target_bus[:, QD_H]
        if pred_bus.size(1) > QG_OUT:
            Qg = torch.where(
                mask_bus[:, QG_H],
                pred_bus[:, QG_OUT],
                target_bus[:, QG_H],
            )
        else:
            Qg = target_bus[:, QG_H]

        net_P = Pg_per_bus - Pd
        net_Q = Qg - Qd
        S_net = net_P + 1j * net_Q

        # --- Loss ---
        loss = torch.mean(torch.abs(S_net - S_injection))

        real_loss = torch.mean(
            torch.abs(torch.real(S_net - S_injection)),
        )
        imag_loss = torch.mean(
            torch.abs(torch.imag(S_net - S_injection)),
        )

        result = {
            "loss": loss,
            "Power loss in p.u.": loss.detach(),
            "Active Power Loss in p.u.": real_loss.detach(),
            "Reactive Power Loss in p.u.": imag_loss.detach(),
        }
        if self.visualization:
            result["Nodal Active Power Loss in p.u."] = torch.abs(
                torch.real(S_net - S_injection),
            )
            result["Nodal Reactive Power Loss in p.u."] = torch.abs(
                torch.imag(S_net - S_injection),
            )
        return result


@LOSS_REGISTRY.register("QgViolationPenalty")
class QgViolationPenaltyLoss(BaseLoss):
    """Standard Mean Squared Error loss."""

    def __init__(self, loss_args, args):
        super().__init__()

    def forward(
        self,
        pred,
        target,
        edge_index=None,
        edge_attr=None,
        mask=None,
        model=None,
        x_dict=None,
    ):
        # --- Qg limit violation mask ---
        Qg_pred = pred["bus"][:, QG_OUT]
        Qg_max = x_dict["bus"][:, MAX_QG_H]
        Qg_min = x_dict["bus"][:, MIN_QG_H]

        max_penalty_mask = Qg_pred > Qg_max
        min_penalty_mask = Qg_pred < Qg_min

        loss = 0.0
        # where there are violations, compute penalty loss
        Qg_over = F.relu(Qg_pred - Qg_max)  # amount above max limit
        Qg_under = F.relu(Qg_min - Qg_pred)  # amount below min limit

        Qg_over = Qg_over[max_penalty_mask].mean()
        Qg_under = Qg_under[min_penalty_mask].mean()

        if Qg_over != Qg_over:  # replacing nan with 0
            Qg_over = 0.0
        if Qg_under != Qg_under:  # replacing nan with 0
            Qg_under = 0.0

        penalty_loss = Qg_over + Qg_under
        loss += penalty_loss

        try:
            output = {"loss": loss, "Qg Violation Penalty loss": loss.detach()}
        except Exception:
            output = {"loss": loss, "Qg Violation Penalty loss": loss}

        return output
