"""Run a compiled GENCO model.so on the reference batches and compare with PyTorch.

Needs only numpy and onnx-mlir's Python runtime (PyRuntime), so it runs wherever the model
was compiled. The reference .npz comes from ``export_onnx.py``.

    python check_model_so.py model.so genco_case118_tiny_opset18_dynamo_reference.npz \
        --runtime_dir /workdir/onnx-mlir/build/Debug/lib

Pass or fail is decided on the ``accuracy`` batch (every scenario in the partition): the
compiled model's error against the power-flow solver must stay within ``--max_ratio`` of
PyTorch float32's, for every output column. The float64 comparison is printed as a
diagnostic: a miscompile shows there as 1e-1 or worse.
"""

import argparse
import os
import statistics
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
SOLVER_COLUMNS = ["VM", "VA", "PG (slack)", "QG"]
# Emulated DLFLOAT16 with the CPU config stays within 1.06 of float32 on 200-scenario
# partitions from light to heavy load; default NNPA placement is 1.9 to 6.6.
MAX_RATIO = 1.25


def relative_error(outputs, reference):
    # one value per output column: bus VM, VA, PG, QG, then gen PG
    return np.concatenate(
        [np.abs(o - r).max(0) / np.abs(r).max(0) for o, r in zip(outputs, reference)],
    )


def accuracy_ratios(bus_out, bus_f32, target, predicted):
    """Mean and 99th-percentile |error| vs the solver per column, for the model and float32.

    Returns ``(model, float32, ratio)``, each ``[4 columns, (mean, p99)]``. Only entries the
    model predicts count; the rest are inputs copied through.
    """
    stats = np.zeros((2, target.shape[1], 2))
    for c in range(target.shape[1]):
        p = predicted[:, c]
        for i, out in enumerate((bus_out, bus_f32)):
            err = np.abs(out[p, c] - target[p, c])
            stats[i, c] = err.mean(), np.percentile(err, 99)
    return stats[0], stats[1], stats[0] / stats[1]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("model_so")
    p.add_argument("reference")
    p.add_argument("--runtime_dir", help="directory that holds PyRuntime.py")
    p.add_argument(
        "--max_ratio",
        type=float,
        default=MAX_RATIO,
        help="max model.so / PyTorch float32 error ratio vs the solver, per column",
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=30,
        help="timed runs per batch; the median is reported (the accuracy batch uses at most 5)",
    )
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
    if "accuracy/solver_bus" not in ref:
        sys.exit(
            "The reference file has no accuracy batch; re-export it with export_onnx.py.",
        )
    session.run([ref[f"{batches[0]}/{n}"] for n in INPUT_ORDER])  # warm-up

    print("relative error vs float64, worst column (diagnostic)")
    print(
        f"{'batch':<20} {'buses':>6} {'gens':>6}   {'PyTorch f32':>11} {'model.so':>9}  worst   ms/run",
    )
    outputs = {}
    for b in batches:
        inputs = [ref[f"{b}/{n}"] for n in INPUT_ORDER]
        outputs[b] = session.run(inputs)  # also the per-batch warm-up
        # run() is synchronous (CPU and on-chip NNPA alike), so wall time is the full run.
        # Median: one slow run (page fault, scheduler) moves a mean but not a median.
        times = []
        for _ in range(min(cli.repeats, 5) if b == "accuracy" else cli.repeats):
            start = time.perf_counter()
            session.run(inputs)
            times.append(time.perf_counter() - start)
        ms = statistics.median(times) * 1e3 if times else float("nan")
        f64 = [ref[f"{b}/f64_bus_output"], ref[f"{b}/f64_gen_output"]]
        f32 = [ref[f"{b}/f32_bus_output"], ref[f"{b}/f32_gen_output"]]
        err, err_torch = relative_error(outputs[b], f64), relative_error(f32, f64)
        print(
            f"{b:<20} {inputs[0].shape[0]:>6} {inputs[1].shape[0]:>6}   "
            f"{err_torch.max():>11.1e} {err.max():>9.1e}  {COLUMNS[int(err.argmax())]:<6} {ms:7.1f}",
        )

    target, predicted = ref["accuracy/solver_bus"], ref["accuracy/solver_predicted"]
    model, f32, ratio = accuracy_ratios(
        outputs["accuracy"][0],
        ref["accuracy/f32_bus_output"],
        target,
        predicted,
    )
    scale = np.abs(target).max(0)
    print(
        f"\naccuracy vs power-flow solver, {target.shape[0]} buses "
        f"(|error| / largest |value| per column; ratio = model.so / PyTorch float32)",
    )
    print(
        f"{'column':<11} {'f32 mean':>9} {'so mean':>9} {'ratio':>6}   {'f32 p99':>9} {'so p99':>9} {'ratio':>6}",
    )
    for c, name in enumerate(SOLVER_COLUMNS):
        print(
            f"{name:<11} {f32[c, 0] / scale[c]:>9.2e} {model[c, 0] / scale[c]:>9.2e} {ratio[c, 0]:>6.3f}   "
            f"{f32[c, 1] / scale[c]:>9.2e} {model[c, 1] / scale[c]:>9.2e} {ratio[c, 1]:>6.3f}",
        )
    ok = bool(ratio.max() <= cli.max_ratio)
    print(
        f"{'PASS' if ok else 'FAIL'}: worst error ratio {ratio.max():.3f} (limit {cli.max_ratio})",
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
