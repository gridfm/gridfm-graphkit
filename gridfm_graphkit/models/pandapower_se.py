"""Weighted least squares (WLS) state estimation baseline, solved with pandapower.

`PandapowerSE` converts one grid scenario into a pandapower network, adds the
simulated measurements, and runs a Gauss-Newton WLS estimate with bad-data
removal (largest normalized residual test).

Two parts of the estimator are set here:

- Transformer magnetizing susceptance keeps the sign of `i0_percent`
  (`_calc_y_with_sign`, used by pandapower when it builds the admittance matrix).
  Transformers with capacitive shunts then have the same admittance as in the
  dataset.
- Bad-data removal (`_remove_bad_data`) removes up to `max_bad_data_cycles`
  measurements, one per cycle, and stops when an estimate does not converge.

Install the `se` extra, which pins `pandapower==3.3.0` and `numpy<2.4`.
"""

import logging

import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.build_branch as pp_build_branch
import torch
from pandapower.estimation.ppc_conversion import _initialize_voltage
from pandapower.estimation.state_estimation import StateEstimation
from torch import nn
from torch_scatter import scatter_add

from gridfm_graphkit.datasets.globals import (
    BS,
    GS,
    PG_H,
    VA_OUT,
    VM_OUT,
    YFF_TT_R,
    YFT_TF_I,
)
from gridfm_graphkit.io.registries import MODELS_REGISTRY
from gridfm_graphkit.models.utils import (
    ComputeBranchFlow,
    ComputeNodeInjection,
    compute_shunt_power,
)

logger = logging.getLogger(__name__)

# Column order of the bus features in the processed dataset.
BUS_COLUMNS = [
    "Pd",
    "Qd",
    "Qg",
    "Vm",
    "Va",
    "PQ",
    "PV",
    "REF",
    "min_vm_pu",
    "max_vm_pu",
    "min_q_mvar",
    "max_q_mvar",
    "GS",
    "BS",
    "vn_kv",
]
# Column order of the branch features in the processed dataset, plus the bus indices.
BRANCH_COLUMNS = [
    "pf",
    "qf",
    "Yff_r",
    "Yff_i",
    "Yft_r",
    "Yft_i",
    "tap",
    "ang_min",
    "ang_max",
    "rate_a",
    "from_bus",
    "to_bus",
]
# Smallest standard deviation given to pandapower, in the units of the measurement.
MIN_STD = 1e-2


def _calc_y_with_sign(mode, trafo_df, vn_lv, vn_trafo_lv, net_sn_mva):
    """Transformer shunt admittance in p.u., keeping the sign of `i0_percent`.

    Same as `pandapower.build_branch._calc_y_from_dataframe` in pandapower
    3.3.0, except that the susceptance is multiplied by `sign(i0_percent)`.
    """
    get_trafo_values = pp_build_branch.get_trafo_values
    three_phase = mode == "pf_3ph"
    baseZ = np.square(vn_lv) / (3 * net_sn_mva if three_phase else net_sn_mva)
    vn_lv_kv = get_trafo_values(trafo_df, "vn_lv_kv")
    pfe_mw = get_trafo_values(trafo_df, "pfe_kw") * 1e-3
    i0 = get_trafo_values(trafo_df, "i0_percent")
    if three_phase:
        pfe_mw = pfe_mw / 3
        i0 = i0 / 3
    parallel = get_trafo_values(trafo_df, "parallel")
    trafo_sn_mva = get_trafo_values(trafo_df, "sn_mva")

    vnl_squared = vn_lv_kv**2 / 3 if three_phase else vn_lv_kv**2
    ym_mva = i0 / 100 * trafo_sn_mva
    b_mva_squared = np.square(ym_mva) - np.square(pfe_mw)
    b_mva_squared[b_mva_squared < 0] = 0
    b_mva = -np.sqrt(b_mva_squared)

    g_pu = pfe_mw / vnl_squared * baseZ * parallel / np.square(vn_trafo_lv / vn_lv_kv)
    b_pu = (
        b_mva
        * np.sign(i0)
        / vnl_squared
        * baseZ
        * parallel
        / np.square(vn_trafo_lv / vn_lv_kv)
    )
    return g_pu, b_pu


pp_build_branch._calc_y_from_dataframe = _calc_y_with_sign


def _remove_bad_data(
    net,
    tolerance=1e-6,
    maximum_iterations=100,
    max_bad_data_cycles=1000,
    rn_max_threshold=3.0,
):
    """WLS estimate with largest normalized residual bad-data removal.

    Each cycle runs a WLS estimate and computes the normalized residuals. If the
    largest one is above `rn_max_threshold`, that measurement is removed and the
    estimate runs again.

    Returns:
        net: The network with the estimate in `res_bus_est`. It can be a different
            object from the input, because pandapower may replace it while estimating.
        success: True if an estimate passed the test within `max_bad_data_cycles` cycles.
        cycles: Number of measurements removed.
    """
    wls = StateEstimation(net, tolerance, maximum_iterations, algorithm="wls")
    v_start, delta_start = _initialize_voltage(net, "flat")

    for cycle in range(max_bad_data_cycles + 1):
        # Same arguments as pandapower's perform_rn_max_test. The third
        # positional argument is `zero_injection`.
        wls.estimate(v_start, delta_start, True)
        if not wls.solver.successful:
            return wls.net, False, cycle
        try:
            # Residual covariance Omega = R - H G^-1 H^T, then r_N = |r| / sqrt(diag(Omega)).
            # Computed in the same order as pandapower's perform_rn_max_test.
            R = np.linalg.inv(wls.solver.R_inv)
            omega = R - np.dot(
                wls.solver.H,
                np.dot(np.linalg.inv(wls.solver.Gm), wls.solver.H.T),
            )
            omega = np.sqrt(np.absolute(np.diag(np.diag(omega))))
            r_normalized = np.dot(np.linalg.inv(omega), np.absolute(wls.solver.r))
        except np.linalg.LinAlgError:
            return wls.net, False, cycle

        if r_normalized.max() <= rn_max_threshold:
            return wls.net, True, cycle

        worst = np.argsort(r_normalized, axis=0)[-1]
        meas_idx = wls.solver.pp_meas_indices[worst]
        wls.net.measurement = wls.net.measurement.drop(meas_idx)

    return wls.net, False, max_bad_data_cycles


@MODELS_REGISTRY.register("PandapowerSE")
class PandapowerSE(nn.Module):
    """WLS state estimator with bad-data removal, run on one scenario at a time.

    The input must be denormalized: powers in MW/MVAr, angles in degrees, and
    admittances and shunts in p.u. on the case base `baseMVA_orig`.
    The batch size must be 1.

    The output has the same layout as the neural estimator: per bus
    [Vm (p.u.), Va (degrees), P_inj (MW), Q_inj (MVAr)]. Injections are
    computed from the estimated voltages. If the estimate fails, the output is
    all zeros and `success` is False.
    """

    def __init__(self, args):
        super().__init__()
        self.branch_flow_layer = ComputeBranchFlow()
        self.node_injection_layer = ComputeNodeInjection()

    @staticmethod
    def parse_impedances(Yff_r, Yff_i, Yft_r, Yft_i, Ytf_r, Ytf_i, Ytt_r, Ytt_i, tap):
        """Return series r, x and total line charging b from the branch admittances."""
        assert np.isclose(Yft_r, Ytf_r).all()
        assert np.isclose(Yft_i, Ytf_i).all()
        assert np.isclose(Yff_r * (tap**2), Ytt_r).all()
        assert np.isclose(Yff_i * (tap**2), Ytt_i).all()
        assert np.isclose(-Ytf_r * tap, Ytt_r, atol=1e-5).all()

        y_r = -tap * Ytf_r
        y_i = -tap * Ytf_i
        b = 2 * (Ytt_i - y_i).values
        z = 1 / (y_r + 1j * y_i)
        return np.real(z), np.imag(z), b

    @staticmethod
    def build_net(
        bus_data: pd.DataFrame,
        branch_data: pd.DataFrame,
        baseMVA_orig: float = 100,
    ) -> pp.pandapowerNet:
        """Build a pandapower network with buses, shunts, lines and transformers.

        A branch with tap 1 between buses of the same voltage level becomes a
        line. Every other branch becomes a transformer.
        """
        net = pp.create_empty_network(sn_mva=baseMVA_orig)
        assert (branch_data["tap"] >= 1e-2).values.all()
        pp.create_buses(
            net,
            nr_buses=bus_data.shape[0],
            name=bus_data.index,
            vn_kv=bus_data["vn_kv"],
            index=bus_data.index,
            zone=1,
        )
        shunts = bus_data[(bus_data["BS"] != 0) | (bus_data["GS"] != 0)]
        pp.create_shunts(
            net,
            buses=shunts.index,
            p_mw=shunts["GS"] * baseMVA_orig,
            q_mvar=-shunts["BS"] * baseMVA_orig,
        )
        R, X, B = PandapowerSE.parse_impedances(
            branch_data["Yff_r"],
            branch_data["Yff_i"],
            branch_data["Yft_r"],
            branch_data["Yft_i"],
            branch_data["Ytf_r"],
            branch_data["Ytf_i"],
            branch_data["Ytt_r"],
            branch_data["Ytt_i"],
            branch_data["tap"],
        )

        ext_grid = bus_data[bus_data["REF"] == 1].index.item()
        pp.create_ext_grid(net, bus=ext_grid, vm_pu=1.0, va_degree=0.0)

        for (branch_name, branch), r, x, b in zip(branch_data.iterrows(), R, X, B):
            from_bus, to_bus = branch["from_bus"], branch["to_bus"]
            vn_kv_from = bus_data.at[from_bus, "vn_kv"].item()
            vn_kv_to = bus_data.at[to_bus, "vn_kv"].item()
            tap = branch["tap"]
            if (tap == 1.0) and (vn_kv_from == vn_kv_to):
                zbase = vn_kv_from**2 / baseMVA_orig
                omega = np.pi * 50
                pp.create_line_from_parameters(
                    net,
                    from_bus,
                    to_bus,
                    length_km=1,
                    r_ohm_per_km=zbase * r,
                    x_ohm_per_km=zbase * x,
                    c_nf_per_km=b / zbase / omega * 1e9 / 2,
                    max_i_ka=9999,
                    name=branch_name,
                )
            else:
                sn = branch["rate_a"]
                if sn == 0:
                    sn = 9999
                z = np.sqrt(r**2 + x**2)
                ratio_1 = tap - 1
                hv_bus, lv_bus = from_bus, to_bus
                if vn_kv_to > vn_kv_from:
                    hv_bus, lv_bus = to_bus, from_bus

                pp.create_transformer_from_parameters(
                    net,
                    hv_bus=hv_bus,
                    lv_bus=lv_bus,
                    sn_mva=sn,
                    vn_hv_kv=max(vn_kv_from, vn_kv_to),
                    vn_lv_kv=min(vn_kv_from, vn_kv_to),
                    vk_percent=np.sign(x) * z * 100 * sn / baseMVA_orig,
                    vkr_percent=r * sn * 100 / baseMVA_orig,
                    max_loading_percent=100,
                    pfe_kw=0,
                    i0_percent=-b * baseMVA_orig * 100 / sn,
                    shift_degree=0,
                    tap_step_percent=np.abs(ratio_1) * 100,
                    tap_pos=np.sign(ratio_1),
                    tap_side="hv",
                    tap_neutral=0,
                    name=branch_name,
                    tap_changer_type="Ratio",
                )

        return net

    @staticmethod
    def build_measurements(
        net,
        bus_data: pd.DataFrame,
        branch_data: pd.DataFrame,
        std_bus: pd.DataFrame,
        std_branch: pd.DataFrame,
    ):
        """Add one pandapower measurement per measured value.

        An infinite standard deviation means the value is not measured.
        """
        for bus_name, bus in pd.concat([bus_data, std_bus], axis=1).iterrows():
            if bus["std_Vm"] != float("inf"):
                pp.create_measurement(
                    net,
                    "v",
                    element_type="bus",
                    value=bus["Vm"],
                    std_dev=bus["std_Vm"],
                    element=bus_name,
                )
            if bus["std_Pd"] != float("inf"):
                # pandapower bus measurements use the load sign convention.
                pp.create_measurement(
                    net,
                    "p",
                    element_type="bus",
                    value=bus["Pd"] - bus["Pg"],
                    std_dev=np.sqrt(bus["std_Pd"] ** 2 + bus["std_Pg"] ** 2),
                    element=bus_name,
                )
                pp.create_measurement(
                    net,
                    "q",
                    element_type="bus",
                    value=bus["Qd"] - bus["Qg"],
                    std_dev=np.sqrt(bus["std_Qd"] ** 2 + bus["std_Qg"] ** 2),
                    element=bus_name,
                )

        for branch_name, branch in pd.concat(
            [branch_data, std_branch],
            axis=1,
        ).iterrows():
            from_bus, to_bus = branch["from_bus"], branch["to_bus"]
            vn_kv_from = bus_data.loc[from_bus, "vn_kv"].item()
            vn_kv_to = bus_data.loc[to_bus, "vn_kv"].item()
            line = net.line[net.line["name"] == branch_name].index
            assert line.shape[0] <= 1
            if line.shape[0] == 1:
                element_type, element = "line", line.item()
                # The reactive flow at the "to" side of a line is weighted with the
                # standard deviation of the active flow at that side.
                sides = [
                    (
                        "from",
                        branch["pf"],
                        branch["qf"],
                        branch["std_pf"],
                        branch["std_qf"],
                    ),
                    (
                        "to",
                        branch["pt"],
                        branch["qt"],
                        branch["std_pt"],
                        branch["std_pt"],
                    ),
                ]
            else:
                trafo = net.trafo[net.trafo["name"] == branch_name].index
                assert trafo.shape[0] == 1
                element_type, element = "trafo", trafo.item()
                from_side = (
                    branch["pf"],
                    branch["qf"],
                    branch["std_pf"],
                    branch["std_qf"],
                )
                to_side = (
                    branch["pt"],
                    branch["qt"],
                    branch["std_pt"],
                    branch["std_qt"],
                )
                hv_side, lv_side = (
                    (to_side, from_side)
                    if vn_kv_to > vn_kv_from
                    else (from_side, to_side)
                )
                sides = [("hv", *hv_side), ("lv", *lv_side)]

            for side, p, q, std_p, std_q in sides:
                if std_p == float("inf"):
                    continue
                pp.create_measurement(
                    net,
                    "p",
                    element_type=element_type,
                    value=p,
                    std_dev=std_p,
                    element=element,
                    side=side,
                )
                pp.create_measurement(
                    net,
                    "q",
                    element_type=element_type,
                    value=q,
                    std_dev=std_q,
                    element=element,
                    side=side,
                )
        return net

    def forward(
        self,
        x_dict,
        edge_index_dict,
        edge_attr_dict,
        mask_dict,
        batch_dict=None,
        baseMVA_orig=100,
    ):
        bus_data, branch_data, std_bus, std_branch = self.converter(
            x_dict,
            edge_index_dict,
            edge_attr_dict,
            mask_dict,
        )
        net = self.build_net(bus_data, branch_data, baseMVA_orig=baseMVA_orig)
        net = self.build_measurements(net, bus_data, branch_data, std_bus, std_branch)
        net, success, cycles = _remove_bad_data(net)

        if success:
            # Injections are computed from the estimated voltages with the same
            # branch flow equations as the neural estimator.
            num_bus = x_dict["bus"].size(0)
            bus_temp = torch.tensor(
                net.res_bus_est[["vm_pu", "va_degree"]].values,
                dtype=torch.float64,
            )
            bus_temp[:, VA_OUT] *= np.pi / 180
            bus_edge_attr = edge_attr_dict[("bus", "connects", "bus")].clone()
            bus_edge_attr[:, YFF_TT_R : YFT_TF_I + 1] *= baseMVA_orig
            bus_x = x_dict["bus"].clone()
            bus_x[:, [GS, BS]] *= baseMVA_orig
            bus_edge_index = edge_index_dict[("bus", "connects", "bus")]

            Pft, Qft = self.branch_flow_layer(bus_temp, bus_edge_index, bus_edge_attr)
            P_in, Q_in = self.node_injection_layer(Pft, Qft, bus_edge_index, num_bus)
            p_shunt, q_shunt = compute_shunt_power(bus_temp, bus_x)
            result = torch.stack(
                [
                    bus_temp[:, VM_OUT],
                    bus_temp[:, VA_OUT] * (180 / np.pi),
                    P_in - p_shunt,
                    Q_in - q_shunt,
                ],
                dim=1,
            )
        else:
            result = torch.zeros(x_dict["bus"].shape[0], 4, dtype=torch.float64)

        return {
            "bus": result,
            "success": success,
            "net": net,
            "bad_data_cycles": cycles,
        }

    def converter(self, x_dict, edge_index_dict, edge_attr_dict, mask_dict):
        """Convert one scenario into bus and branch tables for pandapower."""
        num_bus = x_dict["bus"].size(0)
        gen_to_bus = edge_index_dict[("gen", "connected_to", "bus")][1, :]
        agg_gen_on_bus = scatter_add(
            x_dict["gen"][:, PG_H],
            gen_to_bus,
            dim=0,
            dim_size=num_bus,
        )
        bus_data = pd.DataFrame(x_dict["bus"].cpu().numpy(), columns=BUS_COLUMNS)
        bus_data["Pg"] = agg_gen_on_bus.cpu().numpy()

        edge_attr = edge_attr_dict[("bus", "connects", "bus")]
        branch_data = self.edge_attr_to_branch_data(
            edge_attr,
            edge_index_dict[("bus", "connects", "bus")],
        )

        # Summing generator standard deviations is exact only with one generator per bus.
        agg_gen_std_on_bus = scatter_add(
            mask_dict["std_gen"][:, PG_H],
            gen_to_bus,
            dim=0,
            dim_size=num_bus,
        )
        std_bus = torch.cat(
            [mask_dict["std_bus"], agg_gen_std_on_bus.unsqueeze(1)],
            dim=1,
        )
        std_bus = pd.DataFrame(
            torch.clamp(std_bus, min=MIN_STD).cpu().numpy(),
            columns=["std_Pd", "std_Qd", "std_Qg", "std_Vm", "std_Va", "std_Pg"],
        )

        std_branch = mask_dict["std_branch"]
        n_branches = std_branch.size(0) // 2
        std_branch = torch.cat(
            [std_branch[:n_branches, :], std_branch[n_branches:, :]],
            dim=1,
        )
        std_branch = pd.DataFrame(
            torch.clamp(std_branch, min=MIN_STD).cpu().numpy(),
            columns=["std_pf", "std_qf", "std_pt", "std_qt"],
        )

        return bus_data, branch_data, std_bus, std_branch

    def edge_attr_to_branch_data(self, edge_attr, edge_index):
        """One row per branch. The first half of the edges are forward, the second half reverse."""
        # The table keeps the dtype of edge_attr, so the line and transformer
        # parameters are computed in float32.
        table = (
            torch.cat([edge_attr, edge_index.T.to(edge_attr.dtype)], dim=1)
            .cpu()
            .numpy()
        )
        df = pd.DataFrame(table, columns=BRANCH_COLUMNS)
        n_branches = df.shape[0] // 2
        forward = df.iloc[:n_branches].reset_index(drop=True)
        reverse = df.iloc[n_branches:].reset_index(drop=True)

        shared = ["tap", "ang_max", "ang_min", "rate_a"]
        assert (forward[shared].values == reverse[shared].values).all(), (
            "Forward and reverse edges have different branch parameters. The WLS baseline needs batch_size = 1."
        )
        reverse = reverse.rename(
            columns={
                "pf": "pt",
                "qf": "qt",
                "Yff_r": "Ytt_r",
                "Yff_i": "Ytt_i",
                "Yft_r": "Ytf_r",
                "Yft_i": "Ytf_i",
            },
        )
        return pd.concat(
            [forward, reverse[["pt", "qt", "Ytt_r", "Ytt_i", "Ytf_r", "Ytf_i"]]],
            axis=1,
        )
