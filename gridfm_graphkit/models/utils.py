import torch
from torch import nn
from torch_scatter import scatter_add
from gridfm_graphkit.io.registries import PHYSICS_DECODER_REGISTRY

from gridfm_graphkit.datasets.globals import (
    # Bus feature indices
    PD_H,
    QD_H,
    GS,
    BS,
    # Output feature indices
    VM_OUT,
    VA_OUT,
    PG_OUT,
    QG_OUT,
    # Edge feature indices
    YFF_TT_R,
    YFF_TT_I,
    YFT_TF_R,
    YFT_TF_I,
)


class ComputeBranchFlow(nn.Module):
    """Compute sending-end branch flows (Pf, Qf) for all branches."""

    def forward(self, bus_data, edge_index, edge_attr):
        from_idx, to_idx = edge_index

        # Voltage magnitudes and angles
        Vf_mag, Vf_ang = bus_data[from_idx, VM_OUT], bus_data[from_idx, VA_OUT]
        Vt_mag, Vt_ang = bus_data[to_idx, VM_OUT], bus_data[to_idx, VA_OUT]

        # Real & imaginary voltage components
        Vf_r = Vf_mag * torch.cos(Vf_ang)
        Vf_i = Vf_mag * torch.sin(Vf_ang)
        Vt_r = Vt_mag * torch.cos(Vt_ang)
        Vt_i = Vt_mag * torch.sin(Vt_ang)

        # Branch admittance components
        Yfftt_r, Yfftt_i = edge_attr[:, YFF_TT_R], edge_attr[:, YFF_TT_I]
        Yfttf_r, Yfttf_i = edge_attr[:, YFT_TF_R], edge_attr[:, YFT_TF_I]

        # Sending-end currents
        Ift_r = Yfftt_r * Vf_r - Yfftt_i * Vf_i + Yfttf_r * Vt_r - Yfttf_i * Vt_i
        Ift_i = Yfftt_r * Vf_i + Yfftt_i * Vf_r + Yfttf_r * Vt_i + Yfttf_i * Vt_r

        # Sending-end power flows
        Pft = Vf_r * Ift_r + Vf_i * Ift_i
        Qft = Vf_i * Ift_r - Vf_r * Ift_i

        return Pft, Qft


class ComputeNodeInjection(nn.Module):
    """Aggregate branch flows into node-level incoming injections."""

    def forward(self, Pft, Qft, edge_index, num_bus):
        """
        Args:
            Pft, Qft: [num_edges] branch flows
            edge_index: [2, num_edges] (from_bus, to_bus)
            num_bus: number of bus nodes
        Returns:
            P_in, Q_in: aggregated incoming power per bus
        """
        from_idx, _ = edge_index  # Only sending end contributes as "incoming" to node
        P_in = scatter_add(Pft, from_idx, dim=0, dim_size=num_bus)
        Q_in = scatter_add(Qft, from_idx, dim=0, dim_size=num_bus)

        return P_in, Q_in


def compute_shunt_power(bus_data_pred, bus_data_orig):
    p_shunt = -bus_data_orig[:, GS] * bus_data_pred[:, VM_OUT] ** 2
    q_shunt = bus_data_orig[:, BS] * bus_data_pred[:, VM_OUT] ** 2
    return p_shunt, q_shunt


@PHYSICS_DECODER_REGISTRY.register("OptimalPowerFlow")
class PhysicsDecoderOPF(nn.Module):
    def forward(self, P_in, Q_in, bus_data_pred, bus_data_orig, agg_bus, mask_dict):
        mask_pv = mask_dict["PV"]
        mask_ref = mask_dict["REF"]

        mask_pvref = mask_pv | mask_ref

        # Shunt reactive power contribution
        _, q_shunt = compute_shunt_power(bus_data_pred, bus_data_orig)

        # Reactive load
        Qd = bus_data_orig[:, QD_H]

        # ---- COMPUTE Qg FOR PV & REF ----
        # Nodal reactive balance:
        #     Qg = Q_in + Qd - q_shunt
        Qg_physics = Q_in + Qd - q_shunt

        # Use torch.where instead of boolean index-put to avoid aten.nonzero
        # (data-dependent shape) which causes inductor graph breaks under
        # torch.compile.
        Qg_new = torch.where(mask_pvref, Qg_physics, torch.zeros_like(Qg_physics))
        Pg_out = agg_bus  # Active generation (Pg)
        Qg_out = Qg_new  # Reactive gen (Qg)
        Vm_out = bus_data_pred[:, VM_OUT]  # Voltage magnitude
        Va_out = bus_data_pred[:, VA_OUT]  # Voltage angle

        # Concatenate into [num_buses, 4]
        output = torch.stack([Vm_out, Va_out, Pg_out, Qg_out], dim=1)

        return output


@PHYSICS_DECODER_REGISTRY.register("PowerFlow")
class PhysicsDecoderPF(nn.Module):
    def forward(self, P_in, Q_in, bus_data_pred, bus_data_orig, agg_bus, mask_dict):
        """
        PF decoder:
        - Compute Pg at REF bus
        - Compute Qg at PV + REF buses
        - PQ buses: Pg = 0, Qg = 0
        - Return stacked tensor [Vm, Va, Pg, Qg]
        """

        # Masks
        mask_pv = mask_dict["PV"]
        mask_ref = mask_dict["REF"]
        mask_pvref = mask_pv | mask_ref  # Qg computed here

        # --- Shunt contributions ---
        p_shunt, q_shunt = compute_shunt_power(bus_data_pred, bus_data_orig)

        # --- Loads ---
        Pd = bus_data_orig[:, PD_H]
        Qd = bus_data_orig[:, QD_H]

        # ======================
        #   Qg (PV + REF)
        # ======================
        # Use torch.where instead of boolean index-put to avoid aten.nonzero
        # (data-dependent shape) which causes inductor graph breaks under
        # torch.compile.
        Qg_new = torch.where(mask_pvref, Q_in + Qd - q_shunt, torch.zeros_like(Q_in))

        # ======================
        #   Pg (REF only)
        # ======================
        Pg_ref = torch.where(mask_ref, P_in + Pd - p_shunt, torch.zeros_like(P_in))
        Pg_new = torch.where(mask_pv, agg_bus, Pg_ref)  # PV: keep predicted

        # Voltages
        Vm_out = bus_data_pred[:, VM_OUT]
        Va_out = bus_data_pred[:, VA_OUT]

        # Stack into [num_buses, 4] -> [Vm, Va, Pg, Qg]
        output = torch.stack([Vm_out, Va_out, Pg_new, Qg_new], dim=1)

        return output


@PHYSICS_DECODER_REGISTRY.register("StateEstimation")
class PhysicsDecoderSE(nn.Module):
    def forward(self, P_in, Q_in, bus_data_pred, bus_data_orig, agg_bus, mask_dict):
        p_shunt, q_shunt = compute_shunt_power(bus_data_pred, bus_data_orig)
        Vm_out = bus_data_pred[:, VM_OUT]
        Va_out = bus_data_pred[:, VA_OUT]
        output = torch.stack([Vm_out, Va_out, P_in - p_shunt, Q_in - q_shunt], dim=1)
        return output


class ComputeNodeResiduals(nn.Module):
    """Compute net residuals per bus combining branch flows, generators, loads, and shunts."""

    def forward(self, P_in, Q_in, bus_data_pred, bus_data_orig):
        # Shunt contributions
        p_shunt, q_shunt = compute_shunt_power(bus_data_pred, bus_data_orig)

        # Net residuals per bus
        residual_P = bus_data_pred[:, PG_OUT] - bus_data_orig[:, PD_H] + p_shunt - P_in
        residual_Q = bus_data_pred[:, QG_OUT] - bus_data_orig[:, QD_H] + q_shunt - Q_in

        return residual_P, residual_Q


def bound_with_sigmoid(pred, low, high):
    return low + (high - low) * torch.sigmoid(pred)
