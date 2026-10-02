"""Time the two optimizer-step layouts described in BATCH_TIMING.md.

Layout 1: two mixed-grid forwards (32 PF + 32 OPF, new random grid mix each step).
Layout 2: six homogeneous forwards (10 graphs each, one grid and one task, fixed).

Graphs are real case14/30/57 PF and OPF scenarios from ``--data_path`` after the
task masks, loaded through the multi-task datamodule.
"""

import argparse
import random
import time

import numpy as np
import torch
import yaml
from torch_geometric.data import Batch

import gridfm_graphkit  # noqa: F401  (registers models and transforms)
from gridfm_graphkit.datasets.hetero_powergrid_datamodule import LitGridHeteroDataModule
from gridfm_graphkit.io.param_handler import NestedNamespace, load_model

GRIDS = ("case14_ieee", "case30_ieee", "case57_ieee")
TASKS = ("PowerFlow", "OptimalPowerFlow")
WARMUP = 2
TIMED = 8
POOL = 64  # graphs kept on the GPU per (grid, task)
REPEATS = 2


def load_pools(config_path, data_path, device):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    # Only a few graphs per grid-task are needed for timing.
    cfg["data"]["scenarios"] = [400] * len(cfg["data"]["networks"])
    args = NestedNamespace(**cfg)
    dm = LitGridHeteroDataModule(args, data_dir=data_path)
    dm.setup("fit")
    pools = {}
    for i, (network, task) in enumerate(zip(cfg["data"]["networks"], cfg["data"]["tasks"])):
        grid = network.rsplit("_", 1)[0]  # case14_ieee_pf -> case14_ieee
        ds = dm.train_datasets[i]
        graphs = [ds[j].to(device) for j in range(POOL)]
        pools[(grid, task)] = graphs
        g = graphs[0]
        print(
            f"{grid:12s} {task:16s} buses={g['bus'].num_nodes} gens={g['gen'].num_nodes} "
            f"bus-bus edges={g['bus', 'connects', 'bus'].num_edges} "
            f"bus x={tuple(g['bus'].x.shape)} gen x={tuple(g['gen'].x.shape)}",
        )
    return args, pools


def random_mix(total, rng):
    """Random composition of ``total`` into three grid counts; zeros allowed."""
    cuts = sorted(rng.randint(0, total) for _ in range(len(GRIDS) - 1))
    return [cuts[0], cuts[1] - cuts[0], total - cuts[1]]


def build_layout1_steps(pools, n_steps, rng):
    steps = []
    for _ in range(n_steps):
        micro = []
        for task in TASKS:
            counts = random_mix(32, rng)
            graphs = []
            for grid, n in zip(GRIDS, counts):
                graphs += rng.sample(pools[(grid, task)], n)
            micro.append((task, Batch.from_data_list(graphs), counts))
        steps.append(micro)
    return steps


def build_layout2_step(pools):
    return [
        (task, Batch.from_data_list(pools[(grid, task)][:10]), grid)
        for task in TASKS
        for grid in GRIDS
    ]


def train_step(model, optimizer, microbatches):
    optimizer.zero_grad(set_to_none=True)
    scale = 1.0 / len(microbatches)
    for task, batch, _ in microbatches:
        model.set_task(task)
        out = model(batch)
        loss = (out["bus"].pow(2).mean() + out["gen"].pow(2).mean()) * scale
        loss.backward()
    optimizer.step()


def time_layout(model, optimizer, steps):
    """steps: list of WARMUP + TIMED microbatch lists. Returns ms per step."""
    for s in steps[:WARMUP]:
        train_step(model, optimizer, s)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for s in steps[WARMUP:]:
        train_step(model, optimizer, s)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / TIMED


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="examples/multitask_multigrid/pf_opf_case14_30_57.yaml")
    p.add_argument("--data_path", required=True, help="Directory with case14_ieee_pf, ... (see AGENT.md)")
    a = p.parse_args()

    assert torch.cuda.is_available(), "needs a GPU"
    device = torch.device("cuda")
    torch.manual_seed(0)
    np.random.seed(0)
    rng = random.Random(0)
    print("GPU:", torch.cuda.get_device_name(0))
    print("torch", torch.__version__)

    args, pools = load_pools(a.config, a.data_path, device)
    model = load_model(args).to(device).train()
    print("model:", type(model).__name__)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, betas=(0.9, 0.999))

    n = WARMUP + TIMED
    layout2 = build_layout2_step(pools)
    results = {1: [], 2: []}
    for r in range(REPEATS):
        steps1 = build_layout1_steps(pools, n, rng)
        mixes = [[m[2] for m in s] for s in steps1[WARMUP:]]
        print(f"run {r + 1} layout 1 timed mixes [PF counts, OPF counts] (14/30/57):", mixes)
        results[1].append(time_layout(model, optimizer, steps1))
        results[2].append(time_layout(model, optimizer, [layout2] * n))
        print(
            f"run {r + 1}: layout 1 (2 mixed forwards, bs 64) = {results[1][-1]:.2f} ms/step | "
            f"layout 2 (6 homogeneous forwards, bs 60) = {results[2][-1]:.2f} ms/step",
        )

    m1, m2 = np.mean(results[1]), np.mean(results[2])
    print("\nSUMMARY")
    print("GPU:", torch.cuda.get_device_name(0))
    print("layout 1 ms/step:", ", ".join(f"{x:.2f}" for x in results[1]), f"(mean {m1:.2f})")
    print("layout 2 ms/step:", ", ".join(f"{x:.2f}" for x in results[2]), f"(mean {m2:.2f})")
    fast, slow = (1, 2) if m1 < m2 else (2, 1)
    mf, ms = min(m1, m2), max(m1, m2)
    print(
        f"layout {fast} is faster by {ms - mf:.2f} ms/step ({ms / mf:.2f}x, "
        f"{100 * (ms - mf) / ms:.1f}% less time per step)",
    )
    # Per-graph view, since the batch sizes differ (64 vs 60).
    print(f"per graph: layout 1 {m1 / 64:.3f} ms, layout 2 {m2 / 60:.3f} ms")


if __name__ == "__main__":
    main()
