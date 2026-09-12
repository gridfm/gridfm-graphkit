"""Helpers for the OPF tutorial notebooks (not part of the graphkit package)."""

from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import torch
import yaml
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData

from gridfm_graphkit.cli import _normalize_loaded_state_dict_keys
from gridfm_graphkit.datasets.hetero_powergrid_datamodule import LitGridHeteroDataModule
from gridfm_graphkit.datasets.powergrid_hetero_dataset import HeteroGridDatasetDisk
from gridfm_graphkit.io.param_handler import NestedNamespace, get_task
from gridfm_graphkit.io.registries import DATASET_WRAPPER_REGISTRY

# Processed ``data_index_*.pt`` copies, keyed by (processed_dir, idx).
_RAM_GRAPH_DICTS: dict[tuple[str, int], dict] = {}

# Paper checkpoints shipped in ``examples/notebooks/models/``.
_PAPER_WEIGHT_FILES = (
    (
        "465594abc8eb4c96b908363afc84a0e1/artifacts/model/best_model_state_dict.pt",
        "case118_small_best_model_state_dict.pt",
    ),
    (
        "465594abc8eb4c96b908363afc84a0e1/artifacts/stats/normalizer_stats.pt",
        "case118_small_normalizer_stats.pt",
    ),
    (
        "f227514415974447befc64513a758432/artifacts/model/best_model_state_dict.pt",
        "case2000_small_best_model_state_dict.pt",
    ),
)


def link_paper_weights(repo: Path, experiments_root: Path) -> Path:
    """Symlink shipped ``.pt`` files into the MLflow-style paths the notebook expects."""
    models = Path(repo) / "examples" / "notebooks" / "models"
    experiments_root = Path(experiments_root)
    missing = [str(models / name) for _, name in _PAPER_WEIGHT_FILES if not (models / name).is_file()]
    if missing:
        raise FileNotFoundError("Paper weights missing from clone: " + ", ".join(missing))
    for rel, name in _PAPER_WEIGHT_FILES:
        dest = experiments_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() or dest.is_symlink():
            continue
        dest.symlink_to((models / name).resolve())
    return experiments_root


def enable_ram_graph_cache():
    """Patch ``HeteroGridDatasetDisk.get`` so each graph is loaded from disk once.

    The processed ``.pt`` files stay on disk; this only keeps an in-process copy
    of the un-normalized dict and clones it before the normalizer runs. Safe to
    call more than once (notebook ``importlib.reload``). Does not change the CLI.
    """
    if getattr(HeteroGridDatasetDisk.get, "_tutorial_ram_cached", False):
        return HeteroGridDatasetDisk.get

    def get(self, idx):
        key = (osp.abspath(self.processed_dir), int(idx))
        cached = _RAM_GRAPH_DICTS.get(key)
        if cached is None:
            file_name = osp.join(self.processed_dir, f"data_index_{idx}.pt")
            if not osp.exists(file_name):
                raise IndexError(f"Data file {file_name} does not exist.")
            cached = torch.load(file_name, weights_only=True)
            _RAM_GRAPH_DICTS[key] = cached
        data = HeteroData.from_dict(cached).clone()
        self.data_normalizer.transform(data=data)
        return data

    get._tutorial_ram_cached = True
    HeteroGridDatasetDisk.get = get
    return get


class RamCacheDataset(Dataset):
    """Eager in-RAM copy of a (usually Subset) dataset. CLI ``cache_dir`` is ignored."""

    def __init__(self, dataset, cache_dir=None):
        del cache_dir
        self._items = [dataset[i] for i in range(len(dataset))]

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx].clone()


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _cache_val_batches(task, dm, device):
    batches = []
    with torch.no_grad():
        for batch in dm.val_dataloader():
            batch = task.transfer_batch_to_device(batch, device, 0)
            batch = task.on_after_batch_transfer(batch, 0)
            batches.append(batch)
    return batches


def _layer_mags(task, batches, n_layers):
    buckets = [[] for _ in range(n_layers)]
    idx = [0]

    def _hook(_module, _inputs, out):
        rp, rq = out
        mag = torch.linalg.norm(torch.stack([rp, rq], dim=-1), dim=-1)
        buckets[idx[0]].append(mag.reshape(-1).detach().cpu())
        idx[0] += 1

    handle = task.model.node_residuals_layer.register_forward_hook(_hook)
    with torch.no_grad():
        for batch in batches:
            idx[0] = 0
            task(batch)
    handle.remove()
    return [torch.cat(b).numpy() for b in buckets]


def _style_boxes(bp, color):
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.7)


def plot_val_residual_boxplot(
    config_path,
    data_path,
    mlflow_client,
    run_id,
    random_init=False,
    overlay=False,
    seed=0,
):
    """Box plot of per-bus |ΔS| on the val set, one box per layer.

    Loads the datamodule once. ``overlay=True`` caches transferred val batches
    and runs two forwards (random init, then best checkpoint). Otherwise a
    single forward: checkpoint, or random weights if ``random_init=True``.
    """
    cfg = NestedNamespace(**yaml.safe_load(Path(config_path).read_text()))
    cfg.data.workers = 0
    dm = LitGridHeteroDataModule(cfg, str(data_path))
    dm.setup("fit")
    torch.manual_seed(seed)
    task = get_task(cfg, dm.data_normalizers)
    device = _device()
    task.to(device)
    task.eval()

    n_layers = task.model.num_layers
    batches = _cache_val_batches(task, dm, device)

    if overlay:
        random_data = _layer_mags(task, batches, n_layers)
        art = Path(
            mlflow_client.get_run(run_id).info.artifact_uri.replace("file://", ""),
        )
        ckpt = art / "model" / "best_model_state_dict.pt"
        if not ckpt.is_file():
            raise FileNotFoundError(f"No best checkpoint at {ckpt}")
        task.load_state_dict(
            _normalize_loaded_state_dict_keys(torch.load(ckpt, map_location="cpu")),
        )
        task.eval()
        trained_data = _layer_mags(task, batches, n_layers)

        layers = np.arange(1, n_layers + 1)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        bp_t = ax.boxplot(
            trained_data,
            positions=layers - 0.18,
            widths=0.32,
            showfliers=False,
            patch_artist=True,
            manage_ticks=False,
        )
        bp_r = ax.boxplot(
            random_data,
            positions=layers + 0.18,
            widths=0.32,
            showfliers=False,
            patch_artist=True,
            manage_ticks=False,
        )
        _style_boxes(bp_t, "#4C78A8")
        _style_boxes(bp_r, "#F58518")
        ax.legend(
            [bp_t["boxes"][0], bp_r["boxes"][0]],
            ["best checkpoint", "random init"],
        )
        ax.set_xticks(layers)
        ax.set_xticklabels([str(i) for i in range(n_layers)])
        ax.set_xlabel("layer")
        ax.set_ylabel(r"$|\Delta S_k|$ (p.u.)")
        ax.set_title(r"Validation set: per-bus $|\Delta S|$ by layer")
        plt.show()
        return

    if not random_init:
        art = Path(
            mlflow_client.get_run(run_id).info.artifact_uri.replace("file://", ""),
        )
        ckpt = art / "model" / "best_model_state_dict.pt"
        if not ckpt.is_file():
            raise FileNotFoundError(f"No best checkpoint at {ckpt}")
        task.load_state_dict(
            _normalize_loaded_state_dict_keys(torch.load(ckpt, map_location="cpu")),
        )
        task.eval()

    data = _layer_mags(task, batches, n_layers)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot(data, tick_labels=[str(i) for i in range(n_layers)], showfliers=False)
    ax.set_xlabel("layer")
    ax.set_ylabel(r"$|\Delta S_k|$ (p.u.)")
    ax.set_title(
        r"Validation set, random init: per-bus $|\Delta S|$ by layer"
        if random_init
        else r"Validation set, best checkpoint: per-bus $|\Delta S|$ by layer",
    )
    plt.show()


def _quadratic_cost(pg, gen_x):
    from gridfm_graphkit.datasets.globals import C0_H, C1_H, C2_H

    return (gen_x[:, C2_H] * pg**2 + gen_x[:, C1_H] * pg + gen_x[:, C0_H]).sum()


def _read_raw_scenario(raw_dir: Path, table: str, scenario_id: int):
    import pandas as pd

    df = _read_hive_partition(str(Path(raw_dir) / table), scenario_id // 200)
    out = df[df["scenario"] == scenario_id].copy()
    if out.empty:
        raise ValueError(
            f"No {table} rows for scenario {scenario_id} in {raw_dir / table}",
        )
    return out


@lru_cache(maxsize=64)
def _read_hive_partition(table_path: str, partition: int):
    import pandas as pd

    path = Path(table_path)
    part = path / f"scenario_partition={partition}"
    return pd.read_parquet(part if part.is_dir() else path)


def _load_dc_dispatch(
    data_path,
    network: str,
    scenario_id: int,
    gen_bus,
    pg_ac,
    n_bus: int,
):
    """DC-OPF ``p_mw_dc`` / ``Pg_dc``, aligned to graph gens (in-service only)."""
    raw = Path(data_path) / network / "raw"
    gen = _read_raw_scenario(raw, "gen_data.parquet", scenario_id)
    bus = _read_raw_scenario(raw, "bus_data.parquet", scenario_id)
    if "in_service" in gen.columns:
        gen = gen[gen["in_service"] == 1]
    gen = gen.reset_index(drop=True)
    bus = bus.sort_values("bus")
    if "p_mw_dc" not in gen.columns or "Pg_dc" not in bus.columns:
        raise KeyError(
            f"DC-OPF columns missing in {raw} (need gen p_mw_dc and bus Pg_dc)",
        )
    gen_bus = np.asarray(gen_bus)
    pg_ac = np.asarray(pg_ac, dtype=float)
    buses = gen["bus"].to_numpy()
    pg_ac_raw = gen["p_mw"].to_numpy(dtype=float)
    if gen_bus.shape[0] != buses.shape[0]:
        raise ValueError(
            f"DC table size mismatch for scenario {scenario_id}: "
            f"gen {buses.shape[0]} vs {gen_bus.shape[0]}, bus {len(bus)} vs {n_bus}",
        )
    aligned = np.array_equal(buses, gen_bus) and np.allclose(
        pg_ac_raw,
        pg_ac,
        atol=1e-3,
        rtol=1e-4,
    )
    if not aligned:
        used = set()
        order = []
        for b, p in zip(gen_bus, pg_ac):
            cands = [
                i
                for i in range(len(gen))
                if i not in used and int(buses[i]) == int(b)
            ]
            if not cands:
                raise ValueError(
                    f"No DC gen row for bus {int(b)} in scenario {scenario_id}",
                )
            i = min(cands, key=lambda j: abs(pg_ac_raw[j] - p))
            order.append(i)
            used.add(i)
        gen = gen.iloc[order].reset_index(drop=True)
        pg_ac_raw = gen["p_mw"].to_numpy(dtype=float)
        if not np.allclose(pg_ac_raw, pg_ac, atol=1e-3, rtol=1e-4):
            raise ValueError(
                f"Could not align DC gens to graph for scenario {scenario_id}",
            )
    pg_gen = gen["p_mw_dc"].to_numpy(dtype=float)
    pg_bus = bus["Pg_dc"].to_numpy(dtype=float)
    va_dc = bus["Va_dc"].to_numpy(dtype=float)
    if pg_bus.shape[0] != n_bus:
        raise ValueError(
            f"DC bus size mismatch for scenario {scenario_id}: "
            f"{pg_bus.shape[0]} vs {n_bus}",
        )
    return pg_gen, pg_bus, va_dc


def solve_one_opf(
    config_path,
    data_path,
    mlflow_client=None,
    run_id=None,
    model_path=None,
    normalizer_stats=None,
):
    """One forward pass on the first validation scenario; tables vs IPOPT and DC-OPF."""
    import pandas as pd
    from IPython.display import display

    from gridfm_graphkit.datasets.globals import (
        MAX_PG,
        MIN_PG,
        PG_OUT,
        QG_OUT,
        VA_OUT,
        VM_H,
        VM_OUT,
    )
    from gridfm_graphkit.models.utils import (
        ComputeBranchFlow,
        ComputeNodeInjection,
        ComputeNodeResiduals,
    )

    cfg = NestedNamespace(**yaml.safe_load(Path(config_path).read_text()))
    cfg.data.workers = 0
    cfg.training.batch_size = 1
    dm = LitGridHeteroDataModule(
        cfg,
        str(data_path),
        normalizer_stats_path=(
            None if normalizer_stats is None else str(normalizer_stats)
        ),
    )
    dm.setup("fit")
    task = get_task(cfg, dm.data_normalizers)
    if model_path is None:
        if mlflow_client is None or run_id is None:
            raise ValueError("Pass model_path, or mlflow_client and run_id")
        art = Path(
            mlflow_client.get_run(run_id).info.artifact_uri.replace("file://", ""),
        )
        ckpt = art / "model" / "best_model_state_dict.pt"
    else:
        ckpt = Path(model_path)
    if not ckpt.is_file():
        raise FileNotFoundError(f"No best checkpoint at {ckpt}")
    task.load_state_dict(
        _normalize_loaded_state_dict_keys(torch.load(ckpt, map_location="cpu")),
    )
    device = _device()
    task.to(device)
    task.eval()

    batch = next(iter(dm.val_dataloader()))
    batch = task.transfer_batch_to_device(batch, device, 0)
    batch = task.on_after_batch_transfer(batch, 0)
    with torch.no_grad():
        pred = task(batch)
    task.data_normalizers[0].inverse_transform(batch)
    task.data_normalizers[0].inverse_output(pred, batch)

    bus_x = batch.x_dict["bus"]
    bus_y = batch.y_dict["bus"]
    gen_x = batch.x_dict["gen"]
    gen_y = batch.y_dict["gen"].reshape(-1)
    gen_pred = pred["gen"].reshape(-1)
    _, gen_to_bus = batch.edge_index_dict[("gen", "connected_to", "bus")]
    num_bus = bus_x.size(0)

    Pft, Qft = ComputeBranchFlow()(
        pred["bus"],
        batch.edge_index_dict[("bus", "connects", "bus")],
        batch.edge_attr_dict[("bus", "connects", "bus")],
    )
    P_in, Q_in = ComputeNodeInjection()(
        Pft,
        Qft,
        batch.edge_index_dict[("bus", "connects", "bus")],
        num_bus,
    )
    rP, rQ = ComputeNodeResiduals()(P_in, Q_in, pred["bus"], bus_x)
    pbe = torch.sqrt(rP**2 + rQ**2)

    cost_pred = float(_quadratic_cost(gen_pred, gen_x).detach().cpu())
    cost_ipopt = float(_quadratic_cost(gen_y, gen_x).detach().cpu())
    sid = int(batch["scenario_id"].reshape(-1)[0].item())
    pg_dc_gen, pg_dc_bus, va_dc = _load_dc_dispatch(
        data_path,
        cfg.data.networks[0],
        sid,
        gen_to_bus.detach().cpu().numpy(),
        gen_y.detach().cpu().numpy(),
        num_bus,
    )
    pg_dc_t = torch.as_tensor(pg_dc_gen, dtype=gen_x.dtype, device=gen_x.device)
    cost_dc = float(_quadratic_cost(pg_dc_t, gen_x).detach().cpu())
    dc_bus = pred["bus"].clone()
    dc_bus[:, VM_OUT] = 1.0
    dc_bus[:, VA_OUT] = torch.as_tensor(
        va_dc * (np.pi / 180.0),
        dtype=dc_bus.dtype,
        device=dc_bus.device,
    )
    dc_bus[:, PG_OUT] = torch.as_tensor(
        pg_dc_bus,
        dtype=dc_bus.dtype,
        device=dc_bus.device,
    )
    dc_bus[:, QG_OUT] = 0.0
    Pft_dc, Qft_dc = ComputeBranchFlow()(
        dc_bus,
        batch.edge_index_dict[("bus", "connects", "bus")],
        batch.edge_attr_dict[("bus", "connects", "bus")],
    )
    P_in_dc, Q_in_dc = ComputeNodeInjection()(
        Pft_dc,
        Qft_dc,
        batch.edge_index_dict[("bus", "connects", "bus")],
        num_bus,
    )
    rP_dc, rQ_dc = ComputeNodeResiduals()(P_in_dc, Q_in_dc, dc_bus, bus_x)
    rP_g = rP.detach().cpu().numpy()
    rP_d = rP_dc.detach().cpu().numpy()
    kind = np.where(
        batch.mask_dict["REF"].cpu().numpy(),
        "REF",
        np.where(batch.mask_dict["PV"].cpu().numpy(), "PV", "PQ"),
    )

    print(f"Validation scenario {sid}  (one GENCO forward pass)")
    display(
        pd.DataFrame(
            [
                {
                    "cost GENCO": cost_pred,
                    "cost IPOPT": cost_ipopt,
                    "cost DC": cost_dc,
                    "gap GENCO %": 100.0 * (cost_pred - cost_ipopt) / cost_ipopt,
                    "gap DC %": 100.0 * (cost_dc - cost_ipopt) / cost_ipopt,
                    "mean |ΔS| GENCO (MVA)": float(pbe.mean().detach().cpu()),
                    "mean |ΔP| GENCO (MW)": float(np.mean(np.abs(rP_g))),
                    "mean |ΔP| DC (MW)": float(np.mean(np.abs(rP_d))),
                },
            ],
        ).round(4),
    )

    pmin = gen_x[:, MIN_PG].detach().cpu().numpy()
    pmax = gen_x[:, MAX_PG].detach().cpu().numpy()
    gen_df = pd.DataFrame(
        {
            "gen": np.arange(gen_x.size(0)),
            "Pg GENCO (MW)": gen_pred.detach().cpu().numpy(),
            "Pg IPOPT (MW)": gen_y.detach().cpu().numpy(),
            "Pg DC (MW)": pg_dc_gen,
            "min": pmin,
            "max": pmax,
        },
    )
    dispatch = gen_df.loc[~np.isclose(gen_df["min"], gen_df["max"])].reset_index(
        drop=True,
    )
    print("Generator dispatch (dispatchable only)")
    display(dispatch.round(3))
    gt = dispatch["Pg IPOPT (MW)"].to_numpy()
    d_genco = np.abs(dispatch["Pg GENCO (MW)"].to_numpy() - gt)
    d_dc = np.abs(dispatch["Pg DC (MW)"].to_numpy() - gt)
    nz = np.abs(gt) > 1.0
    pct_genco = 100.0 * np.mean(d_genco[nz] / np.abs(gt[nz]))
    pct_dc = 100.0 * np.mean(d_dc[nz] / np.abs(gt[nz]))
    print(
        f"Average abs. Pg deviation: GENCO {pct_genco:.2f}%, DC {pct_dc:.2f}% "
        f"(vs |Pg_IPOPT|, dispatchable gens with |Pg| > 1 MW)",
    )
    bus_df = pd.DataFrame(
        {
            "bus": np.arange(num_bus),
            "type": kind,
            "Vm GENCO": pred["bus"][:, VM_OUT].detach().cpu().numpy(),
            "Vm IPOPT": bus_y[:, VM_H].detach().cpu().numpy(),
            "|ΔP| GENCO (MW)": np.abs(rP_g),
            "|ΔP| DC (MW)": np.abs(rP_d),
        },
    )
    print("Bus voltages and injections")
    display(bus_df.round(3))


def plot_test_gap_and_violations(
    config_path,
    data_path,
    model_path,
    normalizer_stats,
):
    """Per-sample optimality gap and constraint violations on the test split."""
    import torch.nn.functional as F
    from torch_scatter import scatter_add, scatter_mean

    from gridfm_graphkit.datasets.globals import (
        C0_H,
        C1_H,
        C2_H,
        MAX_QG_H,
        MAX_VM_H,
        MIN_QG_H,
        MIN_VM_H,
        PG_OUT,
        QG_OUT,
        RATE_A,
        VA_OUT,
        VM_OUT,
    )
    from gridfm_graphkit.models.utils import (
        ComputeBranchFlow,
        ComputeNodeInjection,
        ComputeNodeResiduals,
    )
    from gridfm_graphkit.tasks.utils import local_index_per_graph

    cfg = NestedNamespace(**yaml.safe_load(Path(config_path).read_text()))
    cfg.data.workers = 0
    dm = LitGridHeteroDataModule(
        cfg,
        str(data_path),
        normalizer_stats_path=str(normalizer_stats),
    )
    dm.setup("fit")
    task = get_task(cfg, dm.data_normalizers)
    ckpt = Path(model_path)
    if not ckpt.is_file():
        raise FileNotFoundError(f"No checkpoint at {ckpt}")
    task.load_state_dict(
        _normalize_loaded_state_dict_keys(torch.load(ckpt, map_location="cpu")),
    )
    device = _device()
    task.to(device)
    task.eval()

    flow = ComputeBranchFlow()
    inj = ComputeNodeInjection()
    resid = ComputeNodeResiduals()

    genco = {k: [] for k in ("gap", "dP", "dQ", "thermal", "qg", "vm")}
    dc = {k: [] for k in genco}
    network = cfg.data.networks[0]
    loaders = dm.test_dataloader()
    if not isinstance(loaders, (list, tuple)):
        loaders = [loaders]

    def _store(store, gap, bus_out, bus_x, bus_batch, n_graph, ei, ea):
        Pft, Qft = flow(bus_out, ei, ea)
        P_in, Q_in = inj(Pft, Qft, ei, bus_x.size(0))
        rP, rQ = resid(P_in, Q_in, bus_out, bus_x)
        edge_batch = bus_batch[ei[0]]
        therm = F.relu(torch.sqrt(Pft**2 + Qft**2) - ea[:, RATE_A])
        qg = bus_out[:, QG_OUT]
        qg_amt = F.relu(qg - bus_x[:, MAX_QG_H]) + F.relu(bus_x[:, MIN_QG_H] - qg)
        pv_ref = (batch.mask_dict["PV"] | batch.mask_dict["REF"]).to(qg_amt.dtype)
        qg_sum = scatter_add(qg_amt * pv_ref, bus_batch, dim=0, dim_size=n_graph)
        qg_cnt = scatter_add(pv_ref, bus_batch, dim=0, dim_size=n_graph)
        vm = bus_out[:, VM_OUT]
        vm_amt = F.relu(vm - bus_x[:, MAX_VM_H]) + F.relu(bus_x[:, MIN_VM_H] - vm)
        store["gap"].append(gap.detach().cpu())
        store["dP"].append(
            scatter_mean(rP.abs(), bus_batch, dim=0, dim_size=n_graph).cpu(),
        )
        store["dQ"].append(
            scatter_mean(rQ.abs(), bus_batch, dim=0, dim_size=n_graph).cpu(),
        )
        store["thermal"].append(
            scatter_mean(therm, edge_batch, dim=0, dim_size=n_graph).cpu(),
        )
        store["qg"].append((qg_sum / qg_cnt.clamp(min=1)).cpu())
        store["vm"].append(
            scatter_mean(vm_amt, bus_batch, dim=0, dim_size=n_graph).cpu(),
        )

    with torch.no_grad():
        for loader in loaders:
            for batch in loader:
                batch = task.transfer_batch_to_device(batch, device, 0)
                batch = task.on_after_batch_transfer(batch, 0)
                pred = task(batch)
                task.data_normalizers[0].inverse_transform(batch)
                task.data_normalizers[0].inverse_output(pred, batch)

                bus_x = batch.x_dict["bus"]
                gen_x = batch.x_dict["gen"]
                gen_y = batch.y_dict["gen"].reshape(-1)
                gen_pred = pred["gen"].reshape(-1)
                n_graph = int(batch.num_graphs)
                bus_batch = batch.batch_dict["bus"]
                gen_batch = batch.batch_dict["gen"]
                ei = batch.edge_index_dict[("bus", "connects", "bus")]
                ea = batch.edge_attr_dict[("bus", "connects", "bus")]
                _, gen_to_bus = batch.edge_index_dict[("gen", "connected_to", "bus")]
                gen_bus_local = local_index_per_graph(bus_batch)[gen_to_bus]
                sids = batch["scenario_id"].reshape(-1)
                c0, c1, c2 = gen_x[:, C0_H], gen_x[:, C1_H], gen_x[:, C2_H]
                cost_gt = scatter_add(c0 + c1 * gen_y + c2 * gen_y**2, gen_batch, dim=0)
                cost_pr = scatter_add(
                    c0 + c1 * gen_pred + c2 * gen_pred**2,
                    gen_batch,
                    dim=0,
                )
                _store(
                    genco,
                    100.0 * torch.abs((cost_pr - cost_gt) / cost_gt),
                    pred["bus"],
                    bus_x,
                    bus_batch,
                    n_graph,
                    ei,
                    ea,
                )

                dc_bus = pred["bus"].clone()
                dc_bus[:, VM_OUT] = 1.0
                dc_bus[:, QG_OUT] = 0.0
                dc_gen = torch.empty_like(gen_pred)
                for i in range(n_graph):
                    gmask = gen_batch == i
                    bmask = bus_batch == i
                    pg_g, pg_b, va = _load_dc_dispatch(
                        data_path,
                        network,
                        int(sids[i].item()),
                        gen_bus_local[gmask].detach().cpu().numpy(),
                        gen_y[gmask].detach().cpu().numpy(),
                        int(bmask.sum().item()),
                    )
                    dc_gen[gmask] = torch.as_tensor(
                        pg_g,
                        dtype=dc_gen.dtype,
                        device=dc_gen.device,
                    )
                    dc_bus[bmask, PG_OUT] = torch.as_tensor(
                        pg_b,
                        dtype=dc_bus.dtype,
                        device=dc_bus.device,
                    )
                    dc_bus[bmask, VA_OUT] = torch.as_tensor(
                        va * (np.pi / 180.0),
                        dtype=dc_bus.dtype,
                        device=dc_bus.device,
                    )
                cost_dc = scatter_add(
                    c0 + c1 * dc_gen + c2 * dc_gen**2,
                    gen_batch,
                    dim=0,
                )
                _store(
                    dc,
                    100.0 * torch.abs((cost_dc - cost_gt) / cost_gt),
                    dc_bus,
                    bus_x,
                    bus_batch,
                    n_graph,
                    ei,
                    ea,
                )

    def _cat(store, key):
        return torch.cat(store[key]).numpy()

    n = len(_cat(genco, "gap"))
    print(f"Test samples: {n}")
    panels = [
        ("gap", r"Optimality gap (%)"),
        ("dP", r"Mean $|\Delta P|$ (MW)"),
        ("dQ", r"Mean $|\Delta Q|$ (MVar)"),
        ("thermal", r"Mean thermal viol. (MVA)"),
        ("qg", r"Mean $Q_g$ viol. (MVar)"),
        ("vm", r"Mean $|V|$ viol. (p.u.)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2))
    for ax, (key, title) in zip(axes.ravel(), panels):
        g_vals = _cat(genco, key)
        d_vals = _cat(dc, key)
        lo = float(min(g_vals.min(), d_vals.min()))
        hi = float(max(g_vals.max(), d_vals.max()))
        if hi <= lo:
            hi = lo + 1e-6
        bins = np.linspace(lo, hi, 41)
        ax.hist(
            g_vals,
            bins=bins,
            color="#2ca02c",
            alpha=0.55,
            label=f"GENCO mean {np.mean(g_vals):.3g}",
        )
        ax.hist(
            d_vals,
            bins=bins,
            color="#1f77b4",
            alpha=0.55,
            label=f"DC mean {np.mean(d_vals):.3g}",
        )
        ax.set_title(title)
        ax.set_xlabel("per sample")
        ax.set_ylabel("count")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Test-set distributions: GENCO Small vs DC-OPF")
    plt.tight_layout()
    plt.show()


def compare_nb_test_genco_dc(mlflow_client, run_id):
    """Load GENCO and DC-OPF test metrics from this notebook run's MLflow artifacts."""
    import json

    import pandas as pd
    from IPython.display import display

    art = Path(mlflow_client.get_run(run_id).info.artifact_uri.replace("file://", ""))
    test_dir = art / "test"
    genco_csvs = [
        p
        for p in test_dir.glob("*_metrics.csv")
        if not p.name.endswith("_opf_ac_dc_metrics.csv")
    ]
    if not genco_csvs:
        raise FileNotFoundError(f"No GENCO test metrics CSV under {test_dir}")
    genco_csv = genco_csvs[0]
    network = genco_csv.name.removesuffix("_metrics.csv")
    dc_csv = test_dir / f"{network}_opf_ac_dc_metrics.csv"
    if not dc_csv.is_file():
        raise FileNotFoundError(
            f"No DC metrics at {dc_csv}. This run did not log "
            f"`{network}_opf_ac_dc_metrics.csv`.",
        )
    splits_path = art / "stats" / f"{network}_scenario_splits.json"
    n_test = None
    if splits_path.is_file():
        n_test = len(json.loads(splits_path.read_text())["test"])

    genco = pd.read_csv(genco_csv).set_index("Metric")["Value"]
    dc = pd.read_csv(dc_csv).set_index("Metric")["Value"]
    rows = [
        {
            "metric": "Mean optimality gap (%)",
            "GENCO": float(genco["Mean optimality gap (%)"]),
            "DC-OPF": float(dc["DC Mean optimality gap (%)"]),
        },
        {
            "metric": "Avg. active res. (MW)",
            "GENCO": float(genco["Avg. active res. (MW)"]),
            "DC-OPF": float(dc["DC Avg. active res. (MW)"]),
        },
        {
            "metric": "Avg. reactive res. (MVar)",
            "GENCO": float(genco["Avg. reactive res. (MVar)"]),
            "DC-OPF": float("nan"),
        },
        {
            "metric": "Thermal viol. from (MVA)",
            "GENCO": float(genco["Mean branch thermal violation from (MVA)"]),
            "DC-OPF": float(dc["DC Mean branch thermal violation from (MVA)"]),
        },
        {
            "metric": "Thermal viol. to (MVA)",
            "GENCO": float(genco["Mean branch thermal violation to (MVA)"]),
            "DC-OPF": float(dc["DC Mean branch thermal violation to (MVA)"]),
        },
        {
            "metric": "Mean Qg violation (MVar)",
            "GENCO": float(genco["Mean Qg violation"]),
            "DC-OPF": float("nan"),
        },
    ]
    n_txt = f"{n_test} scenarios, " if n_test is not None else ""
    print(f"Test set ({n_txt}run {run_id[:8]}…) from {test_dir}")
    display(pd.DataFrame(rows).round(4))


def _genco_dc_series(mlflow_client, run_id):
    import json

    import pandas as pd

    art = Path(mlflow_client.get_run(run_id).info.artifact_uri.replace("file://", ""))
    test_dir = art / "test"
    genco_csvs = [
        p
        for p in test_dir.glob("*_metrics.csv")
        if not p.name.endswith("_opf_ac_dc_metrics.csv")
    ]
    if not genco_csvs:
        raise FileNotFoundError(f"No GENCO test metrics CSV under {test_dir}")
    genco_csv = genco_csvs[0]
    network = genco_csv.name.removesuffix("_metrics.csv")
    dc_csv = test_dir / f"{network}_opf_ac_dc_metrics.csv"
    if not dc_csv.is_file():
        raise FileNotFoundError(f"No DC metrics at {dc_csv}")
    n_test = None
    splits_path = art / "stats" / f"{network}_scenario_splits.json"
    if splits_path.is_file():
        n_test = len(json.loads(splits_path.read_text())["test"])
    genco = pd.read_csv(genco_csv).set_index("Metric")["Value"]
    dc = pd.read_csv(dc_csv).set_index("Metric")["Value"]
    return genco, dc, n_test, test_dir


def _artifact_dir(mlflow_client, run_id):
    return Path(mlflow_client.get_run(run_id).info.artifact_uri.replace("file://", ""))


def _metric_history(mlflow_client, run_id, name):
    try:
        return sorted(
            mlflow_client.get_metric_history(run_id, name),
            key=lambda m: m.step,
        )
    except Exception:
        return []


def _has_dc_metrics(mlflow_client, run_id):
    return any(_artifact_dir(mlflow_client, run_id).glob("test/*_opf_ac_dc_metrics.csv"))


def _runs_in_same_experiment(mlflow_client, run_id):
    exp_id = mlflow_client.get_run(run_id).info.experiment_id
    return mlflow_client.search_runs(
        [exp_id],
        order_by=["start_time DESC"],
        max_results=50,
    )


def latest_matching_run_id(
    mlflow_client,
    run_id,
    *,
    has_metric=None,
    has_dc_metrics=False,
):
    """Newest sibling of ``run_id`` that has training metrics and/or a DC CSV."""
    for run in _runs_in_same_experiment(mlflow_client, run_id):
        rid = run.info.run_id
        if has_metric and not _metric_history(mlflow_client, rid, has_metric):
            continue
        if has_dc_metrics and not _has_dc_metrics(mlflow_client, rid):
            continue
        return rid
    need = []
    if has_metric:
        need.append(repr(has_metric))
    if has_dc_metrics:
        need.append("DC metrics CSV")
    raise RuntimeError(
        f"No run in the same experiment as {run_id} with {' and '.join(need)}.",
    )


def latest_mlflow_run_id(mlflow_client, name, *, has_metric=None, has_dc_metrics=False):
    """Newest run in an MLflow experiment, or a clear error if it is missing."""
    exp = mlflow_client.get_experiment_by_name(name)
    if exp is None:
        raise RuntimeError(
            f"No MLflow experiment named {name!r}. "
            "The train/finetune CLI may have failed or used a different --exp_name.",
        )
    runs = mlflow_client.search_runs(
        [exp.experiment_id],
        order_by=["start_time DESC"],
        max_results=50,
    )
    if not runs:
        raise RuntimeError(f"Experiment {name!r} has no runs.")
    if has_metric is None and not has_dc_metrics:
        return runs[0].info.run_id
    return latest_matching_run_id(
        mlflow_client,
        runs[0].info.run_id,
        has_metric=has_metric,
        has_dc_metrics=has_dc_metrics,
    )


def plot_scratch_vs_finetune_val_curves(
    mlflow_client,
    scratch_run_id,
    finetune_run_id,
    metrics=("Validation loss", "Validation layer_11_residual"),
):
    """Overlay scratch vs finetune validation histories; bar chart of test opt. gap vs DC."""
    import matplotlib.pyplot as plt

    ylabels = {
        "Validation loss": "validation loss",
        "Validation layer_11_residual": r"mean $|\Delta S|$ (p.u.)",
    }
    curve_metric = metrics[0]
    series = (
        (
            latest_matching_run_id(
                mlflow_client,
                scratch_run_id,
                has_metric=curve_metric,
            ),
            "scratch",
        ),
        (
            latest_matching_run_id(
                mlflow_client,
                finetune_run_id,
                has_metric=curve_metric,
            ),
            "finetune",
        ),
    )
    n_panels = len(metrics) + 1
    fig, axes = plt.subplots(1, n_panels, figsize=(5.2 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]
    for ax, name in zip(axes, metrics):
        for run_id, label in series:
            hist = _metric_history(mlflow_client, run_id, name)
            if not hist:
                raise RuntimeError(f"No {name!r} history for run {run_id}")
            ax.plot(
                [m.step for m in hist],
                [m.value for m in hist],
                "o-",
                label=label,
                markersize=3,
            )
        ax.set_xlabel("step")
        ax.set_ylabel(ylabels.get(name, name))
        ax.set_title(name)
        ax.legend()

    scratch, dc, _, _ = _genco_dc_series(
        mlflow_client,
        latest_matching_run_id(mlflow_client, scratch_run_id, has_dc_metrics=True),
    )
    ft, _, _, _ = _genco_dc_series(
        mlflow_client,
        latest_matching_run_id(mlflow_client, finetune_run_id, has_dc_metrics=True),
    )
    labels = ("scratch", "finetune", "DC-OPF")
    gaps = (
        float(scratch["Mean optimality gap (%)"]),
        float(ft["Mean optimality gap (%)"]),
        float(dc["DC Mean optimality gap (%)"]),
    )
    ax = axes[-1]
    bars = ax.bar(labels, gaps, color=("C0", "C1", "C2"))
    ax.bar_label(bars, fmt="%.2f", padding=3)
    ax.set_ylabel("mean optimality gap (%)")
    ax.set_title("Test set vs DC-OPF")
    ax.set_ylim(0, max(gaps) * 1.15 if max(gaps) > 0 else 1)

    fig.tight_layout()
    plt.show()
    return fig


def compare_scratch_vs_finetune(mlflow_client, scratch_run_id, finetune_run_id):
    """Test metrics: random init vs PF-pretrained finetune vs DC-OPF."""
    import pandas as pd
    from IPython.display import display

    scratch, dc, n_test, _ = _genco_dc_series(mlflow_client, scratch_run_id)
    ft, _, _, _ = _genco_dc_series(mlflow_client, finetune_run_id)
    rows = [
        {
            "metric": "Mean optimality gap (%)",
            "scratch": float(scratch["Mean optimality gap (%)"]),
            "finetune": float(ft["Mean optimality gap (%)"]),
            "DC-OPF": float(dc["DC Mean optimality gap (%)"]),
        },
        {
            "metric": "Avg. active res. (MW)",
            "scratch": float(scratch["Avg. active res. (MW)"]),
            "finetune": float(ft["Avg. active res. (MW)"]),
            "DC-OPF": float(dc["DC Avg. active res. (MW)"]),
        },
        {
            "metric": "Avg. reactive res. (MVar)",
            "scratch": float(scratch["Avg. reactive res. (MVar)"]),
            "finetune": float(ft["Avg. reactive res. (MVar)"]),
            "DC-OPF": float("nan"),
        },
        {
            "metric": "Thermal viol. from (MVA)",
            "scratch": float(scratch["Mean branch thermal violation from (MVA)"]),
            "finetune": float(ft["Mean branch thermal violation from (MVA)"]),
            "DC-OPF": float(dc["DC Mean branch thermal violation from (MVA)"]),
        },
        {
            "metric": "Mean Qg violation (MVar)",
            "scratch": float(scratch["Mean Qg violation"]),
            "finetune": float(ft["Mean Qg violation"]),
            "DC-OPF": float("nan"),
        },
    ]
    n_txt = f"{n_test} test scenarios, " if n_test is not None else ""
    print(
        f"{n_txt}scratch {scratch_run_id[:8]}… vs finetune {finetune_run_id[:8]}…",
    )
    display(pd.DataFrame(rows).round(4))


def _pf_test_artifacts(mlflow_client, run_id):
    """Load PF test RMSE, residual, and optional DC-PF CSVs for one run."""
    import json

    import pandas as pd

    art = _artifact_dir(mlflow_client, run_id)
    test_dir = art / "test"
    rmse_csvs = sorted(test_dir.glob("*_RMSE.csv"))
    if not rmse_csvs:
        raise FileNotFoundError(f"No PF RMSE CSV under {test_dir}")
    rmse_csv = rmse_csvs[0]
    network = rmse_csv.name.removesuffix("_RMSE.csv")
    metrics_csv = test_dir / f"{network}_metrics.csv"
    if not metrics_csv.is_file():
        raise FileNotFoundError(f"No PF residuals CSV at {metrics_csv}")
    dc_csv = test_dir / f"{network}_ac_dc_metrics.csv"
    dc = None
    if dc_csv.is_file():
        dc = pd.read_csv(dc_csv).set_index("Metric")["Value"]
    n_test = None
    splits_path = art / "stats" / f"{network}_scenario_splits.json"
    if splits_path.is_file():
        n_test = len(json.loads(splits_path.read_text())["test"])
    rmse = pd.read_csv(rmse_csv).set_index("Metric")
    metrics = pd.read_csv(metrics_csv).set_index("Metric")["Value"]
    return rmse, metrics, dc, n_test, test_dir


def compare_scratch_vs_finetune_pf(mlflow_client, scratch_run_id, finetune_run_id):
    """Test PF metrics: random init vs OPF-pretrained finetune vs DC-PF."""
    import pandas as pd
    from IPython.display import display

    s_rmse, s_met, s_dc, n_test, _ = _pf_test_artifacts(mlflow_client, scratch_run_id)
    f_rmse, f_met, f_dc, _, _ = _pf_test_artifacts(mlflow_client, finetune_run_id)

    def _rmse(frame, row, col):
        return float(frame.loc[row, col])

    dc_p = float("nan")
    if s_dc is not None and "DC Avg. active res. (MW)" in s_dc.index:
        dc_p = float(s_dc["DC Avg. active res. (MW)"])
    rows = [
        {
            "metric": "RMSE-PQ Vm (p.u.)",
            "scratch": _rmse(s_rmse, "RMSE-PQ", "Vm (p.u.)"),
            "finetune": _rmse(f_rmse, "RMSE-PQ", "Vm (p.u.)"),
            "DC-PF": float("nan"),
        },
        {
            "metric": "RMSE-PQ Va (rad)",
            "scratch": _rmse(s_rmse, "RMSE-PQ", "Va (radians)"),
            "finetune": _rmse(f_rmse, "RMSE-PQ", "Va (radians)"),
            "DC-PF": float("nan"),
        },
        {
            "metric": "Avg. active res. (MW)",
            "scratch": float(s_met["Avg. active res. (MW)"]),
            "finetune": float(f_met["Avg. active res. (MW)"]),
            "DC-PF": dc_p,
        },
        {
            "metric": "Avg. reactive res. (MVar)",
            "scratch": float(s_met["Avg. reactive res. (MVar)"]),
            "finetune": float(f_met["Avg. reactive res. (MVar)"]),
            "DC-PF": float("nan"),
        },
        {
            "metric": "PBE Mean",
            "scratch": float(s_met["PBE Mean"]),
            "finetune": float(f_met["PBE Mean"]),
            "DC-PF": float("nan"),
        },
    ]
    n_txt = f"{n_test} test scenarios, " if n_test is not None else ""
    print(
        f"{n_txt}scratch {scratch_run_id[:8]}… vs finetune {finetune_run_id[:8]}…",
    )
    display(pd.DataFrame(rows).round(4))


def plot_scratch_vs_finetune_pf_val_curves(
    mlflow_client,
    scratch_run_id,
    finetune_run_id,
    metrics=("Validation loss", "Validation layer_11_residual"),
):
    """Overlay scratch vs PF-finetune val curves; bar chart of test |ΔP| vs DC-PF."""
    import matplotlib.pyplot as plt

    ylabels = {
        "Validation loss": "validation loss",
        "Validation layer_11_residual": r"mean $|\Delta S|$ (p.u.)",
    }
    curve_metric = metrics[0]
    series = (
        (
            latest_matching_run_id(
                mlflow_client,
                scratch_run_id,
                has_metric=curve_metric,
            ),
            "scratch",
        ),
        (
            latest_matching_run_id(
                mlflow_client,
                finetune_run_id,
                has_metric=curve_metric,
            ),
            "finetune",
        ),
    )
    n_panels = len(metrics) + 1
    fig, axes = plt.subplots(1, n_panels, figsize=(5.2 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]
    for ax, name in zip(axes, metrics):
        for run_id, label in series:
            hist = _metric_history(mlflow_client, run_id, name)
            if not hist:
                raise RuntimeError(f"No {name!r} history for run {run_id}")
            ax.plot(
                [m.step for m in hist],
                [m.value for m in hist],
                "o-",
                label=label,
                markersize=3,
            )
        ax.set_xlabel("step")
        ax.set_ylabel(ylabels.get(name, name))
        ax.set_title(name)
        ax.legend()

    _, s_met, s_dc, _, _ = _pf_test_artifacts(mlflow_client, scratch_run_id)
    _, f_met, _, _, _ = _pf_test_artifacts(mlflow_client, finetune_run_id)
    dc_p = 0.0
    if s_dc is not None and "DC Avg. active res. (MW)" in s_dc.index:
        dc_p = float(s_dc["DC Avg. active res. (MW)"])
    labels = ("scratch", "finetune", "DC-PF")
    vals = (
        float(s_met["Avg. active res. (MW)"]),
        float(f_met["Avg. active res. (MW)"]),
        dc_p,
    )
    ax = axes[-1]
    bars = ax.bar(labels, vals, color=("C0", "C1", "C2"))
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_ylabel("mean |ΔP| (MW)")
    ax.set_title("Test set vs DC-PF")
    ax.set_ylim(0, max(vals) * 1.15 if max(vals) > 0 else 1)

    fig.tight_layout()
    plt.show()
    return fig


_HIVE_TABLES = ("bus_data.parquet", "gen_data.parquet", "branch_data.parquet")
_SCEN_PER_PARTITION = 200


def _is_hive_table(path: Path) -> bool:
    return path.is_dir() and any(path.glob("scenario_partition=*"))


def write_flat_parquet_as_hive(
    path: Path,
    scen_per_partition: int = _SCEN_PER_PARTITION,
) -> None:
    """Replace a single parquet file with Hive ``scenario_partition=`` directories."""
    import shutil

    import pandas as pd

    if _is_hive_table(path):
        return
    if not path.is_file():
        raise FileNotFoundError(path)

    df = pd.read_parquet(path)
    df = df.drop(columns=["scenario_partition"], errors="ignore")
    tmp = path.with_name(path.name + ".hive_tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    parts = (df["scenario"].to_numpy() // scen_per_partition).astype("int64")
    df = df.copy()
    df["_part"] = parts
    for part, group in df.groupby("_part"):
        part_dir = tmp / f"scenario_partition={int(part)}"
        part_dir.mkdir()
        group.drop(columns=["_part"]).to_parquet(
            part_dir / "data.parquet",
            index=False,
        )
    path.unlink()
    tmp.rename(path)


def write_runtime_hive_from_bus(
    raw_dir: Path,
    scen_per_partition: int = _SCEN_PER_PARTITION,
) -> None:
    """Write ``runtime_data.parquet`` as Hive dirs from unique bus scenario ids."""
    import shutil

    import pandas as pd

    bus = raw_dir / "bus_data.parquet"
    if not _is_hive_table(bus) and not bus.is_file():
        raise FileNotFoundError(bus)
    scenarios = (
        pd.read_parquet(bus, columns=["scenario"])
        .drop_duplicates("scenario")
        .drop(columns=["scenario_partition"], errors="ignore")
    )
    out = raw_dir / "runtime_data.parquet"
    if out.is_file():
        out.unlink()
    elif out.is_dir():
        shutil.rmtree(out)
    out.mkdir()
    parts = (scenarios["scenario"].to_numpy() // scen_per_partition).astype("int64")
    scenarios = scenarios.copy()
    scenarios["_part"] = parts
    for part, group in scenarios.groupby("_part"):
        part_dir = out / f"scenario_partition={int(part)}"
        part_dir.mkdir()
        group.drop(columns=["_part"]).to_parquet(
            part_dir / "data.parquet",
            index=False,
        )


def copy_hf_hive_partitions(src_raw: Path, dst_raw: Path, n_partitions: int) -> None:
    """Copy the first ``n_partitions`` Hive dirs; never concatenate into one file."""
    import shutil

    dst_raw.mkdir(parents=True, exist_ok=True)
    for table in _HIVE_TABLES:
        src_table = src_raw / table
        dst_table = dst_raw / table
        if not _is_hive_table(src_table):
            raise FileNotFoundError(f"Expected Hive table at {src_table}")
        if dst_table.is_file():
            dst_table.unlink()
        dst_table.mkdir(exist_ok=True)
        for part in range(n_partitions):
            name = f"scenario_partition={part}"
            src_part = src_table / name
            dst_part = dst_table / name
            if not src_part.is_dir():
                raise FileNotFoundError(src_part)
            if dst_part.exists():
                continue
            shutil.copytree(src_part, dst_part)


def link_case118_ieee_alias(data_root: Path, network: str = "nb_opf_case118") -> Path:
    """Point ``data/case118_ieee`` at the tutorial slice so paper YAMLs resolve."""
    data_root = Path(data_root)
    target = (data_root / network).resolve()
    if not target.is_dir():
        raise FileNotFoundError(
            f"Tutorial slice missing: {target}. Run the Hugging Face download cell first."
        )
    alias = data_root / "case118_ieee"
    if alias.is_symlink() or alias.exists():
        if alias.resolve() != target:
            raise FileExistsError(f"{alias} exists and is not {target}")
    else:
        alias.symlink_to(target, target_is_directory=True)
    return alias


def prepare_tutorial_opf_raw(
    data_root: Path,
    network: str = "nb_opf_case118",
    n_scenarios: int = 1_000,
    hf_repo: str = "gridfm/opf_small_case118_ieee",
    hf_dirname: str = "opf_small_case118_ieee",
) -> Path:
    """Download only the Hive partitions needed for ``n_scenarios`` (200 per partition)."""
    from huggingface_hub import snapshot_download

    data_root = Path(data_root)
    if n_scenarios <= 0:
        raise ValueError("n_scenarios must be positive")
    n_partitions = (n_scenarios + _SCEN_PER_PARTITION - 1) // _SCEN_PER_PARTITION
    hf_raw = data_root / hf_dirname / "raw"
    needed = [
        hf_raw / tbl / f"scenario_partition={p}"
        for tbl in _HIVE_TABLES
        for p in range(n_partitions)
    ]
    if not all(part.is_dir() for part in needed):
        snapshot_download(
            repo_id=hf_repo,
            repo_type="dataset",
            local_dir=str(hf_raw),
            allow_patterns=[
                f"{tbl}/scenario_partition={p}/*"
                for tbl in _HIVE_TABLES
                for p in range(n_partitions)
            ],
        )
    dst_raw = data_root / network / "raw"
    copy_hf_hive_partitions(hf_raw, dst_raw, n_partitions)
    write_runtime_hive_from_bus(dst_raw)
    link_case118_ieee_alias(data_root, network)
    return dst_raw

