import torch.nn.functional as F
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from gridfm_graphkit.io.registries import LOSS_REGISTRY
from torch_scatter import scatter_add

from gridfm_graphkit.datasets.globals import (
    # Bus feature indices
    QG_H,
    VM_H,
    VA_H,
    QD_H,
    PD_H,
    # Output feature indices
    VM_OUT,
    VA_OUT,
    QG_OUT,
    PG_OUT,
    # Generator feature indices
    PG_H,
    #####################
    ## Indices of features of the VLD task
    # Bus feature indices
    BUS_BASE_STATUS_H,
    BUS_CONT_H,
    B_ON,
    # Branch feature indices
    BRANCH_BASE_STATUS_E,
    BRANCH_CONT_E,
    #####################
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
    ):
        loss = F.mse_loss(pred[mask], target[mask], reduction=self.reduction)
        return {"loss": loss, "Masked MSE loss": loss.detach()}


@LOSS_REGISTRY.register("MaskedGenMSE")
class MaskedGenMSE(torch.nn.Module):
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
    ):
        loss = F.mse_loss(
            pred_dict["gen"][mask_dict["gen"][:, : (PG_H + 1)]],
            target_dict["gen"][mask_dict["gen"][:, : (PG_H + 1)]],
            reduction=self.reduction,
        )
        return {"loss": loss, "Masked generator MSE loss": loss.detach()}


@LOSS_REGISTRY.register("MaskedBusMSE")
class MaskedBusMSE(torch.nn.Module):
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
    ):
        loss = F.mse_loss(pred, target, reduction=self.reduction)
        return {"loss": loss, "MSE loss": loss.detach()}


class MixedLoss(BaseLoss):
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

    def forward(
        self,
        pred,
        target,
        edge_index=None,
        edge_attr=None,
        mask=None,
        model=None,
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

#######################
@LOSS_REGISTRY.register("VLDTopologyLoss")
class VLDTopologyLoss(BaseLoss):
    """
    Topology-first voltage loss detection objective.

    Expected bus tensor layouts:
      pred_dict["bus"]   : [Vm, Va, Pg, Qg, status_logit]
      target_dict["bus"] : [Pd, Qd, Qg, Vm, Va, bus_status_target]
      x_dict["bus"]      : standard bus features + [bus_base_status, bus_contingency]
      edge_attr          : standard edge attrs + [branch_base_status, branch_contingency]
    """

    def __init__(self, loss_args, args):
        super().__init__()
        self.args = args
        self.input_state_threshold = getattr(loss_args, "input_state_threshold", 0.5)
        self.prediction_threshold = getattr(loss_args, "prediction_threshold", 0.5)
        self.topology_weight = getattr(loss_args, "topology_weight", 1.0)
        self.target_anchor_weight = getattr(loss_args, "target_anchor_weight", 0.25)
        self.off_vm_weight = getattr(loss_args, "off_vm_weight", 1.0)
        self.on_vm_weight = getattr(loss_args, "on_vm_weight", 0.5)
        self.unreachable_l1_weight = getattr(loss_args, "unreachable_l1_weight", 1.0)
        self.topology_confidence_gamma = getattr(loss_args, "topology_confidence_gamma", 10.0)

    @staticmethod
    def _build_graph_reachability(
        num_bus,
        edge_index,
        edge_attr,
        bus_x,
        device,
        threshold=0.5,
    ):
        """
        Build hard reachability labels from base-status and contingency indicators.

        A bus is considered initially available if base_status is on and it is not
        directly hit by contingency. A branch is traversable if base branch status
        is on and it is not hit by contingency.
        """
        bus_base = (bus_x[:, BUS_BASE_STATUS_H] > threshold)
        bus_hit = (bus_x[:, BUS_CONT_H] > threshold)
        bus_available = bus_base & (~bus_hit)

        src, dst = edge_index
        branch_base = (edge_attr[:, BRANCH_BASE_STATUS_E] > threshold)
        branch_hit = (edge_attr[:, BRANCH_CONT_E] > threshold)
        branch_available = branch_base & (~branch_hit)

        reachable = torch.zeros(num_bus, dtype=torch.bool, device=device)

        # Seeds: all buses that remain available after direct contingency.
        seed_nodes = torch.where(bus_available)[0]
        if seed_nodes.numel() == 0:
            return reachable.float(), bus_available.float()

        reachable[seed_nodes] = True

        changed = True
        while changed:
            prev = reachable.clone()
            active_edges = branch_available & reachable[src]
            reachable[dst[active_edges]] = True
            changed = not torch.equal(prev, reachable)

        return reachable.float(), bus_available.float()

    def forward(
            self,
            pred_dict,
            target_dict,
            edge_index_dict,
            edge_attr_dict,
            mask_dict,
            model=None,
    ):
        bus_pred = pred_dict["bus"]
        bus_target = target_dict["bus"]
        bus_x = model.latest_x_dict["bus"] if hasattr(model, "latest_x_dict") else None

        if bus_x is None:
            raise RuntimeError(
                "VLDTopologyLoss requires model.latest_x_dict['bus']. "
                "Store x_dict on the model inside the forward pass."
            )

        if bus_pred.size(1) <= BUS_STATUS_LOGIT_OUT:
            raise ValueError(
                "VLDTopologyLoss expects bus predictions to include a status logit "
                f"at column {BUS_STATUS_LOGIT_OUT}."
            )

        edge_index = edge_index_dict[("bus", "connects", "bus")]
        edge_attr = edge_attr_dict[("bus", "connects", "bus")]

        num_bus = bus_pred.size(0)
        device = bus_pred.device

        topo_target, bus_available = self._build_graph_reachability(
            num_bus=num_bus,
            edge_index=edge_index,
            edge_attr=edge_attr,
            bus_x=bus_x,
            device=device,
            threshold=self.input_state_threshold,
        )

        status_logit = bus_pred[:, BUS_STATUS_LOGIT_OUT]
        status_prob = torch.sigmoid(status_logit)

        target_status = bus_target[:, BUS_STATUS_TARGET].float()
        target_vm = bus_target[:, VM_H].float()
        pred_vm = bus_pred[:, VM_OUT].float()

        topology_confidence = torch.exp(
            -self.topology_confidence_gamma * torch.abs(bus_available - topo_target)
        )

        topology_bce_raw = F.binary_cross_entropy_with_logits(
            status_logit,
            topo_target,
            reduction="none",
        )
        topology_bce = (topology_confidence * topology_bce_raw).mean()

        target_anchor_bce = F.binary_cross_entropy_with_logits(
            status_logit,
            target_status,
            reduction="mean",
        )

        unreachable_mask = (topo_target < 0.5).float()
        reachable_mask = (topo_target >= 0.5).float()

        off_vm_l2 = ((pred_vm ** 2) * unreachable_mask).sum() / unreachable_mask.sum().clamp_min(1.0)
        off_vm_l1 = (pred_vm.abs() * unreachable_mask).sum() / unreachable_mask.sum().clamp_min(1.0)
        off_vm_loss = off_vm_l2 + self.unreachable_l1_weight * off_vm_l1

        on_vm_sq = ((pred_vm - target_vm) ** 2) * reachable_mask * status_prob.detach()
        on_vm_loss = on_vm_sq.sum() / (reachable_mask * status_prob.detach()).sum().clamp_min(1.0)

        total_loss = (
            self.topology_weight * topology_bce
            + self.target_anchor_weight * target_anchor_bce
            + self.off_vm_weight * off_vm_loss
            + self.on_vm_weight * on_vm_loss
        )

        pred_status = (status_prob >= self.prediction_threshold).float()
        topo_acc = (pred_status == topo_target).float().mean()
        target_acc = (pred_status == target_status).float().mean()

        return {
            "loss": total_loss,
            "VLD Topology BCE": topology_bce.detach(),
            "VLD Target Anchor BCE": target_anchor_bce.detach(),
            "VLD Off Vm Loss": off_vm_loss.detach(),
            "VLD On Vm Loss": on_vm_loss.detach(),
            "VLD Topology Accuracy": topo_acc.detach(),
            "VLD Target Accuracy": target_acc.detach(),
        }
#######################