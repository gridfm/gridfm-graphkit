# GENCO for onnx-mlir / zDLC

Exports the GENCO power-flow model (`gridfm/genco-pf-datakit`, case118 tiny seed0) to an ONNX file that
onnx-mlir compiles, and checks a compiled `model.so` against PyTorch.

| File | What it does |
|---|---|
| `export_onnx.py` | Exports the model to ONNX, checks it with onnxruntime, and saves reference batches and an NNPA placement config |
| `check_model_so.py` | Runs a compiled `model.so` on the reference batches and reports the error |

## 1. Export

From the repo root, after the downloads in [`../README.md`](../README.md):

```bash
pip install onnx onnxruntime onnxscript
python examples/spyre/onnx_mlir/export_onnx.py
```

This writes three files to `examples/spyre/onnx_mlir/`:

| File | Size | Content |
|---|---:|---|
| `genco_case118_tiny_opset18_dynamo.onnx` | 9.2 MB | The model, opset 18, all graph sizes dynamic |
| `genco_case118_tiny_opset18_dynamo_reference.npz` | 0.3 MB | 5 input batches of different sizes, with the PyTorch outputs in float32 and float64 |
| `genco_case118_tiny_opset18_dynamo_nnpa_config.json` | 12 KB | Pins the voltage head and the physics block to the CPU (see below) |

The script checks the ONNX file in onnxruntime on those 5 batches and ends with a `PASS` or `FAIL` line. The
config matches nodes by name, so it only fits the `.onnx` exported in the same run.

## 2. Compile

```bash
onnx-mlir -O3 --march=z17 --maccel=NNPA \
    --config-file=genco_case118_tiny_opset18_dynamo_nnpa_config.json \
    genco_case118_tiny_opset18_dynamo.onnx                                          # NNPA
onnx-mlir -O3 genco_case118_tiny_opset18_dynamo.onnx                                # CPU only
```

`--opt-report=NNPAUnsupportedOps` lists the ops that stay on the CPU, with the reason for each.

### Why some ops are pinned to the CPU

NNPA computes in 16-bit (DLFLOAT16, about 3 significant digits). Most of GENCO tolerates that, but two parts
don't:

| Part | Why it is sensitive |
|---|---|
| `mlp_bus`, the head that predicts Vm and Va | Vm sits near 1.0 p.u., where 16-bit steps are 0.002 |
| The physics block (`gridfm_graphkit/models/utils.py`) | Multiplies those voltages by branch admittances of ~30 to get branch flows and power balance, at every one of the 12 layers |

Emulating DLFLOAT16 in PyTorch gives the error below, relative to the largest value per output column against
float64. QG is always the worst column:

| 16-bit applied to | QG, export batch | QG, 32 scenarios |
|---|---:|---:|
| Nothing (float32) | 4.8e-05 | 7.5e-05 |
| Every linear layer and the physics | 3.3e-01 | 2.7e-01 |
| Everything except `mlp_bus` and the physics | 9.6e-03 | 1.9e-02 |

The model's own QG error against the power-flow solver is 1.8e-01 on the export batch, so ~1e-2 from 16-bit is
small next to it.

Keeping only the last layers in float32 doesn't work: the 16-bit errors from earlier layers have already reached
the hidden state through the physics feedback. Removing the physics isn't an option either: without its
per-layer feedback the trained model's errors grow by two orders of magnitude.

Effect on placement:

| | Default | With the config |
|---|---:|---:|
| Compute ops on NNPA | 982 | 545 |
| Matrix products on NNPA | 257 | 233 |
| Compute ops on CPU | 952 | 1,389 |
| Stick / unstick | 1,065 | 842 |

The ops moved to the CPU are mostly 1-D vectors (one value per branch or bus), a poor fit for NNPA anyway.

Other flags tried: `--nnpa-enable-scalar-bcast-binary` moves 36 `Div` ops to NNPA, and
`--nnpa-disable-shape-restriction` moves none. `--nnpa-placement-heuristic=FasterOpsWSU` and `MuchFasterOpsWSU`
abort in onnx-mlir 0.5.1 (`PerfModel.cpp:66`, an assertion on unranked operands).

## 3. Check

```bash
python check_model_so.py genco_case118_tiny_opset18_dynamo.so \
    genco_case118_tiny_opset18_dynamo_reference.npz \
    --runtime_dir <onnx-mlir build>/lib --tol 3e-2  # the directory with PyRuntime.py
```

It needs only numpy and onnx-mlir's `PyRuntime`, and converts the reference data to the machine's byte order
(IBM Z is big-endian). It reports the error per batch, relative to the largest value in each output column,
against the float64 reference. PyTorch's own float32 error is shown for comparison.

Use `--tol 3e-4` (the default) for a CPU build and `--tol 3e-2` for NNPA.

## Inputs and outputs

All sizes are dynamic: one compiled file runs any number of scenarios and any grid.

| Input | Shape | Type |
|---|---|---|
| `bus_features` | `[num_buses, 15]` | f32 |
| `gen_features` | `[num_gens, 6]` | f32 |
| `bus_bus_edge_index` | `[2, num_bus_edges]` | i64 |
| `gen_bus_edge_index` | `[2, num_gens]` | i64 |
| `bus_gen_edge_index` | `[2, num_gens]` | i64 |
| `bus_bus_edge_attr` | `[num_bus_edges, 10]` | f32 |
| `bus_mask` | `[num_buses, 15]` | bool |
| `gen_mask` | `[num_gens, 6]` | bool |
| `pv_mask` | `[num_buses]` | bool |
| `ref_mask` | `[num_buses]` | bool |

Outputs: `bus_output` `[num_buses, 4]` (VM, VA, PG, QG) and `gen_output` `[num_gens, 1]` (PG).

## Why this export compiles

Two things stopped onnx-mlir's shape inference with the earlier export:

| Problem | Fix in `export_onnx.py` |
|---|---|
| The legacy TorchScript exporter wraps every `scatter_reduce` in `If` nodes whose branches return different ranks. onnx-mlir can't infer the `Squeeze` inside them (`Can not squeeze from dynamic dimensions`). | Use the `torch.export`-based exporter (`dynamo=True`) with dynamic shapes. The graph has no `If` nodes and shrinks from 6,013 to 2,500 nodes. |
| The model calls `gen_temp.squeeze()` on a `[num_gens, 1]` tensor. With `num_gens` dynamic, onnx-mlir can't tell whether the result is a vector or a scalar. | Export it as `squeeze(-1)`, which gives the same result for any grid with two or more generators. |

Fixing the shapes doesn't help: a static legacy export still has the `If` nodes, and onnxruntime can't load it.

## Tested with

| | |
|---|---|
| Exporter | PyTorch 2.12.1, onnx 1.23.2, onnxscript 0.7.2, onnxruntime 1.31.0 |
| Compiler | `ghcr.io/onnxmlir/onnx-mlir-dev` (amd64), onnx-mlir 0.5.1, Debug build |
| CPU build | Compiles to `model.so`. `check_model_so.py` passes: worst error 1.2e-04, PyTorch float32 7.5e-05. |
| NNPA build, default placement | On z17: worst error 2.5e-01 to 6.6e-01, from 16-bit precision. 81 ms for the export batch, single thread. |
| NNPA build, with the config | All model stages pass, and placement is as in the table above. Not yet run on z17. |
| CPU timings | PyTorch 35 ms, onnxruntime 40 ms for the export batch, single thread, Apple M4 Max (not IBM Z) |
