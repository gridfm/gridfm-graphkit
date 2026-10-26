"""Run a compiled GENCO model.so on the reference batches and compare with PyTorch.

Needs only numpy and onnx-mlir's Python runtime (PyRuntime), so it runs wherever the model
was compiled. The reference .npz comes from ``export_onnx.py``.

    python check_model_so.py model.so genco_case118_tiny_opset18_dynamo_reference.npz \
        --runtime_dir /workdir/onnx-mlir/build/Debug/lib
"""

import argparse
import os
import sys
import time

import numpy as np

INPUT_ORDER = [
    "bus_features",
    "gen_features",
    "bus_bus_edge_index",
    "gen_bus_edge_index",
    "bus_gen_edge_index",
    "bus_bus_edge_attr",
    "bus_mask",
    "gen_mask",
    "pv_mask",
    "ref_mask",
]
COLUMNS = ["VM", "VA", "PG", "QG", "gen PG"]


def relative_error(outputs, reference):
    # one value per output column: bus VM, VA, PG, QG, then gen PG
    return np.concatenate(
        [np.abs(o - r).max(0) / np.abs(r).max(0) for o, r in zip(outputs, reference)],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("model_so")
    p.add_argument("reference")
    p.add_argument("--runtime_dir", help="directory that holds PyRuntime.py")
    p.add_argument(
        "--tol",
        type=float,
        default=3e-4,
        help="max relative error vs float64: 3e-4 for a CPU build, 3e-2 for NNPA (16-bit)",
    )
    p.add_argument("--repeats", type=int, default=5)
    cli = p.parse_args()

    if cli.runtime_dir:
        sys.path.insert(0, cli.runtime_dir)
    from PyRuntime import OMExecutionSession

    # dlopen treats a bare file name as a library-path search, not the current directory
    session = OMExecutionSession(os.path.abspath(cli.model_so))
    # The .npz is written little-endian; IBM Z is big-endian, and the runtime reads raw bytes.
    with np.load(cli.reference) as npz:
        ref = {
            k: np.ascontiguousarray(v.astype(v.dtype.newbyteorder("=")))
            for k, v in npz.items()
        }
    batches = list(dict.fromkeys(k.split("/")[0] for k in ref))
    session.run([ref[f"{batches[0]}/{n}"] for n in INPUT_ORDER])  # warm-up

    print(
        f"{'batch':<20} {'buses':>6} {'gens':>6}   {'PyTorch f32':>11} {'model.so':>9}  worst   ms/run",
    )
    worst = 0.0
    for b in batches:
        inputs = [ref[f"{b}/{n}"] for n in INPUT_ORDER]
        out = session.run(inputs)
        start = time.perf_counter()
        for _ in range(cli.repeats):
            session.run(inputs)
        ms = (time.perf_counter() - start) / max(cli.repeats, 1) * 1e3
        f64 = [ref[f"{b}/f64_bus_output"], ref[f"{b}/f64_gen_output"]]
        f32 = [ref[f"{b}/f32_bus_output"], ref[f"{b}/f32_gen_output"]]
        err, err_torch = relative_error(out, f64), relative_error(f32, f64)
        worst = max(worst, err.max())
        print(
            f"{b:<20} {inputs[0].shape[0]:>6} {inputs[1].shape[0]:>6}   "
            f"{err_torch.max():>11.1e} {err.max():>9.1e}  {COLUMNS[int(err.argmax())]:<6} {ms:7.1f}",
        )
    ok = worst < cli.tol
    print(
        f"{'PASS' if ok else 'FAIL'}: worst model.so relative error {worst:.1e} (tolerance {cli.tol:.0e})",
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
