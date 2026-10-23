"""Export the library GENCO model to ONNX and check it against PyTorch with onnxruntime.

The model is the one ``genco_forward.py`` loads, so the graph computes exactly what
``gridfm_graphkit predict`` computes. ``--exporter dynamo`` (the default) produces the file
onnx-mlir compiles. The checked batches are saved as a reference ``.npz`` next to the
``.onnx``, for ``check_model_so.py`` to test a compiled model with. See ``README.md``.
"""

import contextlib

import argparse
import os
import sys
import types

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import genco_forward as gf  # noqa: E402

BB = ("bus", "connects", "bus")
GB = ("gen", "connected_to", "bus")
BG = ("bus", "connected_to", "gen")

# (name, how to read it from a PyG batch, dynamic axes)
INPUTS = [
    ("bus_features", lambda b: b["bus"].x, {0: "num_buses"}),
    ("gen_features", lambda b: b["gen"].x, {0: "num_gens"}),
    ("bus_bus_edge_index", lambda b: b.edge_index_dict[BB], {1: "num_bus_edges"}),
    ("gen_bus_edge_index", lambda b: b.edge_index_dict[GB], {1: "num_gens"}),
    ("bus_gen_edge_index", lambda b: b.edge_index_dict[BG], {1: "num_gens"}),
    ("bus_bus_edge_attr", lambda b: b.edge_attr_dict[BB], {0: "num_bus_edges"}),
    ("bus_mask", lambda b: b.mask_dict["bus"], {0: "num_buses"}),
    ("gen_mask", lambda b: b.mask_dict["gen"], {0: "num_gens"}),
    ("pv_mask", lambda b: b.mask_dict["PV"], {0: "num_buses"}),
    ("ref_mask", lambda b: b.mask_dict["REF"], {0: "num_buses"}),
]
OUTPUTS = [("bus_output", {0: "num_buses"}), ("gen_output", {0: "num_gens"})]


class FlatInputs(torch.nn.Module):
    """ONNX takes a flat list of tensors; rebuild the four dicts the model reads."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        bus_x,
        gen_x,
        bb_ei,
        gb_ei,
        bg_ei,
        bb_ea,
        bus_m,
        gen_m,
        pv_m,
        ref_m,
    ):
        batch = types.SimpleNamespace(
            x_dict={"bus": bus_x, "gen": gen_x},
            edge_index_dict={BB: bb_ei, GB: gb_ei, BG: bg_ei},
            edge_attr_dict={BB: bb_ea},
            mask_dict={"bus": bus_m, "gen": gen_m, "PV": pv_m, "REF": ref_m},
        )
        out = self.model(batch)
        return out["bus"], out["gen"]


@contextlib.contextmanager
def squeeze_last_dim():
    """Make ``x.squeeze()`` on a ``[N, 1]`` tensor squeeze only the last dim, during export.

    The model calls ``gen_temp.squeeze()`` on ``[num_gens, 1]``. Exported as a Squeeze with
    no axes, onnx-mlir can't infer its rank when ``num_gens`` is dynamic (it would be a
    scalar if ``num_gens`` were 1). ``squeeze(-1)`` gives the same tensor for 2+ generators.
    """
    original = torch.Tensor.squeeze

    def squeeze(self, *args, **kwargs):
        if not args and not kwargs and self.dim() == 2 and self.shape[-1] == 1:
            return original(self, -1)
        return original(self, *args, **kwargs)

    torch.Tensor.squeeze = squeeze
    try:
        yield
    finally:
        torch.Tensor.squeeze = original


def export(wrapped, inputs, path, opset, exporter, static):
    names = {
        "input_names": [n for n, _, _ in INPUTS],
        "output_names": [n for n, _ in OUTPUTS],
    }
    if exporter == "legacy":
        torch.onnx.export(
            wrapped,
            inputs,
            path,
            opset_version=opset,
            dynamo=False,
            dynamic_axes=None
            if static
            else {n: ax for n, _, ax in INPUTS} | dict(OUTPUTS),
            **names,
        )
        return
    dims = {}
    shapes = None
    if not static:
        shapes = tuple(
            {
                axis: dims.setdefault(name, torch.export.Dim(name))
                for axis, name in ax.items()
            }
            for _, _, ax in INPUTS
        )
    with squeeze_last_dim():
        torch.onnx.export(
            wrapped,
            inputs,
            path,
            opset_version=opset,
            dynamo=True,
            dynamic_shapes=shapes,
            **names,
        )
    # The dynamo exporter writes weights to a side file; onnx-mlir wants one file.
    onnx.save_model(onnx.load(path), path, save_as_external_data=False)
    if os.path.exists(path + ".data"):
        os.remove(path + ".data")


def make_batch(dataset, ids):
    return next(iter(DataLoader([dataset[i] for i in ids], batch_size=len(ids))))


def flat(batch):
    return tuple(get(batch) for _, get, _ in INPUTS)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    m = "data/genco-pf-datakit/case118_ieee/tiny/seed0"
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default=f"{m}/case118_ieee_tiny_seed0.yaml")
    p.add_argument("--model_path", default=f"{m}/best_model_state_dict.pt")
    p.add_argument("--normalizer_stats", default=f"{m}/normalizer_stats.pt")
    p.add_argument("--data_root", default="data/genco_case118_p0")
    p.add_argument("--network", default="case118_ieee")
    p.add_argument("--scenarios", type=int, nargs="+", default=list(range(8)))
    p.add_argument("--opset", type=int, default=18)
    p.add_argument(
        "--exporter",
        choices=["dynamo", "legacy"],
        default="dynamo",
        help="dynamo compiles with onnx-mlir; legacy (TorchScript) adds If nodes it can't infer",
    )
    p.add_argument(
        "--static",
        action="store_true",
        help="fix every size to the export batch (only that batch's sizes will run)",
    )
    p.add_argument(
        "--output",
        help="default: genco_case118_tiny_opset18_<exporter>[_static].onnx",
    )
    cli = p.parse_args()
    if cli.output is None:
        suffix = "_static" if cli.static else ""
        cli.output = os.path.join(
            here,
            f"genco_case118_tiny_opset{cli.opset}_{cli.exporter}{suffix}.onnx",
        )

    args = gf.load_args(cli.config)
    model = gf.load_bare_model(args, cli.model_path)
    dataset = gf.load_dataset(args, cli.data_root, cli.normalizer_stats, cli.network)
    wrapped = FlatInputs(model).eval()

    batch = make_batch(dataset, cli.scenarios)
    with torch.no_grad():
        direct = model(batch)
        via_wrapper = wrapped(*flat(batch))
    assert torch.equal(direct["bus"], via_wrapper[0])
    assert torch.equal(direct["gen"], via_wrapper[1])

    with torch.no_grad():
        export(wrapped, flat(batch), cli.output, cli.opset, cli.exporter, cli.static)
    onnx.checker.check_model(onnx.load(cli.output), full_check=True)
    print(
        f"exported {cli.output}  ({os.path.getsize(cli.output) / 1e6:.1f} MB, opset {cli.opset})",
    )

    # Check on the export batch and on batches with other graph sizes: a traced export can
    # hard-code sizes it saw, even when the axes are declared dynamic.
    n_gen = [dataset[i]["gen"].x.shape[0] for i in range(len(dataset))]
    other = [i for i, n in enumerate(n_gen) if n != n_gen[cli.scenarios[0]]]
    checks = {"export batch": cli.scenarios}
    if not cli.static:
        checks |= {
            "other gen count x8": other[:8],
            "mixed sizes x3": [cli.scenarios[0], other[0], cli.scenarios[-1]],
            "single scenario": [other[1]],
            "32 scenarios": list(range(40, 72)),
        }
    # Reference: the same model in float64, which has almost no rounding error. Each
    # output column is scored as max |error| / max |value|, so VM (~1) and PG (~7) are
    # judged on the same scale. A real export bug shows up as 1e-2 or worse. QG sums
    # large flows of opposite sign, so float32 itself reaches ~1e-4 there on big batches.
    tol = 3e-4
    exact = FlatInputs(gf.load_bare_model(args, cli.model_path).double()).eval()
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    session = ort.InferenceSession(cli.output, opts, providers=["CPUExecutionProvider"])
    cols = ["VM", "VA", "PG", "QG", "gen PG"]
    reference = {}
    print(
        f"worst relative error vs float64, per batch   (PASS if onnxruntime < {tol:.0e})",
    )
    print(
        f"  {'batch':<18} {'shape':<34} {'PyTorch f32':>12} {'onnxruntime':>12}  worst column",
    )
    worst = 0.0
    for label, ids in checks.items():
        b = make_batch(dataset, ids)
        with torch.no_grad():
            f32 = [t.numpy() for t in wrapped(*flat(b))]
            f64 = [
                t.numpy()
                for t in exact(
                    *(t.double() if t.is_floating_point() else t for t in flat(b)),
                )
            ]
        feeds = {n: t.numpy() for (n, _, _), t in zip(INPUTS, flat(b))}
        got = session.run(None, feeds)
        key = label.replace(" ", "_")
        reference |= {f"{key}/{n}": v for n, v in feeds.items()}
        reference |= {f"{key}/f64_bus_output": f64[0], f"{key}/f64_gen_output": f64[1]}
        reference |= {f"{key}/f32_bus_output": f32[0], f"{key}/f32_gen_output": f32[1]}

        def rel(a, ref):
            # one value per output column: bus VM, VA, PG, QG, then gen PG
            return np.concatenate(
                [np.abs(x - r).max(0) / np.abs(r).max(0) for x, r in zip(a, ref)],
            )

        rel_torch, rel_ort = rel(f32, f64), rel(got, f64)
        worst = max(worst, rel_ort.max())
        shape = f"bus {b['bus'].x.shape[0]} gen {b['gen'].x.shape[0]} edges {b.edge_index_dict[BB].shape[1]}"
        print(
            f"  {label:<18} {shape:<34} {rel_torch.max():>12.1e} {rel_ort.max():>12.1e}  "
            f"{cols[int(rel_ort.argmax())]}",
        )
    print(
        f"{'PASS' if worst < tol else 'FAIL'}: worst onnxruntime relative error {worst:.1e}",
    )
    ref_path = os.path.splitext(cli.output)[0] + "_reference.npz"
    np.savez_compressed(ref_path, **reference)
    print(
        f"saved reference batches to {ref_path}  ({os.path.getsize(ref_path) / 1e6:.1f} MB)",
    )


if __name__ == "__main__":
    main()
