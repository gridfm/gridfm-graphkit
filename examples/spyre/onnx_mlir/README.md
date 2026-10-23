# GENCO for onnx-mlir / zDLC

Exports the GENCO power-flow model (`gridfm/genco-pf-datakit`, case118 tiny seed0) to an ONNX file that
onnx-mlir compiles, and checks a compiled `model.so` against PyTorch.

| File | What it does |
|---|---|
| `export_onnx.py` | Exports the model to ONNX, checks it with onnxruntime, and saves reference batches |
| `check_model_so.py` | Runs a compiled `model.so` on the reference batches and reports the error |

## 1. Export

From the repo root, after the downloads in [`../README.md`](../README.md):

```bash
pip install onnx onnxruntime onnxscript
python examples/spyre/onnx_mlir/export_onnx.py
```

This writes two files to `examples/spyre/onnx_mlir/`:

| File | Size | Content |
|---|---:|---|
| `genco_case118_tiny_opset18_dynamo.onnx` | 9.2 MB | The model, opset 18, all graph sizes dynamic |
| `genco_case118_tiny_opset18_dynamo_reference.npz` | 0.3 MB | 5 input batches of different sizes, with the PyTorch outputs in float32 and float64 |

The script checks the ONNX file in onnxruntime on those 5 batches and ends with a `PASS` or `FAIL` line.

## 2. Compile

```bash
onnx-mlir -O3 --march=z17 --maccel=NNPA genco_case118_tiny_opset18_dynamo.onnx      # NNPA
onnx-mlir -O3 genco_case118_tiny_opset18_dynamo.onnx                                # CPU only
```

`--opt-report=NNPAUnsupportedOps` lists the ops that stay on the CPU, with the reason for each.

Two NNPA flags barely change placement here: `--nnpa-enable-scalar-bcast-binary` moves 36 `Div` ops to NNPA
(and adds 36 layout conversions), and `--nnpa-disable-shape-restriction` moves none. The elementwise ops left on
the CPU broadcast `[N, 1]` against `[N, 96]` (the `TransformerConv` skip mix) or work on 1-D per-bus vectors in
the physics block, so moving them would take model changes, not flags.

## 3. Check

```bash
python check_model_so.py genco_case118_tiny_opset18_dynamo.so \
    genco_case118_tiny_opset18_dynamo_reference.npz \
    --runtime_dir <onnx-mlir build>/lib          # the directory with PyRuntime.py
```

It needs only numpy and onnx-mlir's `PyRuntime`. It reports the error per batch, relative to the largest value
in each output column, against the float64 reference. PyTorch's own float32 error is shown for comparison.

The default tolerance (`--tol 3e-4`) suits a CPU build. NNPA computes in 16-bit (DLFLOAT16), so expect larger
errors there and set `--tol` accordingly.

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
| NNPA build | All model stages pass. Placement puts 982 compute ops on NNPA (matrix products, elementwise, activations) and 952 on the CPU (gathers, scatters, layer norm). Code generation then needs an LLVM with the IBM Z backend, which the amd64 image doesn't have. |
