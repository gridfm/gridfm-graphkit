"""Load one GENCO checkpoint and run a bare forward pass on real grid data.

Bring-up harness for onboarding GENCO onto IBM Spyre: no Lightning Trainer,
no MLflow, no datamodule, no loss. It only builds the dataset, restores the
normalizer and runs the plain ``nn.Module``. See ``examples/spyre/README.md``.
"""

import argparse
import hashlib
import time

import torch
import yaml
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_add

import gridfm_graphkit  # noqa: F401  (fires the registry decorators)
from gridfm_graphkit.datasets.globals import (
    PG_H,
    PG_OUT,
    QG_H,
    QG_OUT,
    VA_H,
    VA_OUT,
    VM_H,
    VM_OUT,
)
from gridfm_graphkit.datasets.powergrid_hetero_dataset import HeteroGridDatasetDisk
from gridfm_graphkit.io.config_version import resolve_config
from gridfm_graphkit.io.param_handler import (
    NestedNamespace,
    get_task_transforms,
    load_model,
    load_normalizer,
)

INPUT_KEYS = ("x_dict", "edge_index_dict", "edge_attr_dict", "mask_dict")


def load_args(config_path):
    with open(config_path) as f:
        config = resolve_config(yaml.safe_load(f), config_path, "auto_inline")
    return NestedNamespace(**config)


def load_bare_model(args, model_path):
    """Build the registered model and load the Lightning checkpoint's weights into it."""
    model = load_model(args)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    # The release checkpoints are task state_dicts, so every key starts with "model.".
    state_dict = {
        k.removeprefix("model.").removeprefix("_orig_mod."): v
        for k, v in state_dict.items()
    }
    model.load_state_dict(state_dict)
    return model.eval()


def load_dataset(args, data_root, normalizer_stats, network):
    normalizer = load_normalizer(args)
    stats = torch.load(normalizer_stats, map_location="cpu", weights_only=True)
    normalizer.fit_from_dict(stats[network])
    return HeteroGridDatasetDisk(
        root=data_root,
        data_normalizer=normalizer,
        transform=get_task_transforms(args),
    )


def input_fingerprint(batch):
    """sha256 over every tensor the model reads; equal hashes mean an identical batch."""
    h = hashlib.sha256()
    for key in INPUT_KEYS:
        for name, t in sorted(getattr(batch, key).items(), key=lambda kv: str(kv[0])):
            h.update(str(name).encode())
            h.update(t.contiguous().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def rmse(pred, target, mask):
    return torch.sqrt(((pred - target)[mask] ** 2).mean()).item()


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--normalizer_stats", required=True)
    p.add_argument(
        "--data_root",
        required=True,
        help="dir that holds raw/ (and processed/)",
    )
    p.add_argument("--network", default="case118_ieee", help="key in normalizer_stats")
    p.add_argument(
        "--scenarios",
        type=int,
        nargs="+",
        default=list(range(8)),
        help="scenario ids that form the batch, in this order",
    )
    p.add_argument("--device", default="cpu", help="torch device, e.g. cpu, cuda, mps")
    p.add_argument("--repeats", type=int, default=10, help="timed forward passes")
    p.add_argument("--save_io", help="save the batch inputs and outputs here (.pt)")
    cli = p.parse_args()

    args = load_args(cli.config)
    model = load_bare_model(args, cli.model_path).to(cli.device)
    dataset = load_dataset(args, cli.data_root, cli.normalizer_stats, cli.network)
    # The batch is exactly these scenarios in this order: grouping changes float
    # summation order inside message passing, so it must be fixed for a golden compare.
    samples = [dataset[i] for i in cli.scenarios]
    batch = next(iter(DataLoader(samples, batch_size=len(samples))))
    assert batch["scenario_id"].tolist() == cli.scenarios
    fingerprint = input_fingerprint(batch)
    batch = batch.to(cli.device)

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"model    {type(model).__name__}  {n_params:,} params  "
        f"layers={model.num_layers} hidden={model.hidden_dim} heads={model.heads}",
    )
    print(f"dataset  {len(dataset)} scenarios in {cli.data_root}")
    print(f"batch    scenarios {cli.scenarios} on {cli.device}")
    print(f"inputs   sha256 {fingerprint}")
    for key in INPUT_KEYS:
        for name, t in getattr(batch, key).items():
            print(f"  {key:<16}{str(name):<36}{str(tuple(t.shape)):<14}{t.dtype}")

    with torch.no_grad():
        out = model(batch)
        if cli.device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(cli.repeats):
            out = model(batch)
        if cli.device.startswith("cuda"):
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / max(cli.repeats, 1) * 1e3

    print(
        f"output   bus {tuple(out['bus'].shape)} [VM, VA, PG, QG]   gen {tuple(out['gen'].shape)} [PG]",
    )
    print(f"latency  {ms:.2f} ms / forward (mean of {cli.repeats}, after 1 warm-up)")

    # Sanity check against the solver's solution, on the masked (= predicted) entries,
    # in normalized units: VM in p.u., VA in rad, PG/QG in p.u. of the fitted baseMVA.
    # Slack PG is read at bus level (physics decoder output), as pf_task.py scores it.
    bus = out["bus"].cpu()
    y_bus, m_bus = batch["bus"].y.cpu(), batch.mask_dict["bus"].cpu()
    _, gen_bus = batch.edge_index_dict[("gen", "connected_to", "bus")]
    pg_bus = scatter_add(batch["gen"].y[:, PG_H], gen_bus, dim_size=bus.shape[0]).cpu()
    print("rmse     (normalized, masked entries only)")
    print(
        f"  VM {rmse(bus[:, VM_OUT], y_bus[:, VM_H], m_bus[:, VM_H]):.5f}   "
        f"VA {rmse(bus[:, VA_OUT], y_bus[:, VA_H], m_bus[:, VA_H]):.5f}   "
        f"QG {rmse(bus[:, QG_OUT], y_bus[:, QG_H], m_bus[:, QG_H]):.5f}   "
        f"PG(slack) {rmse(bus[:, PG_OUT], pg_bus, batch.mask_dict['REF'].cpu()):.5f}",
    )
    print(
        f"  mean |power-balance residual| at last layer: "
        f"{model.layer_residuals[model.num_layers - 1].item():.5f}",
    )

    if cli.save_io:
        torch.save(
            {
                "inputs": {
                    k: {str(n): t.cpu() for n, t in getattr(batch, k).items()}
                    for k in INPUT_KEYS
                },
                "outputs": {k: v.cpu() for k, v in out.items()},
                "scenarios": cli.scenarios,
                "fingerprint": fingerprint,
            },
            cli.save_io,
        )
        print(f"saved    reference inputs/outputs to {cli.save_io}")


if __name__ == "__main__":
    main()
