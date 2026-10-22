# GENCO forward pass (Spyre bring-up)

`genco_forward.py` loads one released GENCO power-flow checkpoint as a plain `torch.nn.Module` and runs a forward
pass on real grid scenarios. It leaves out the Lightning Trainer, MLflow, the datamodule, losses and splits. Only
the model, one batch and the outputs remain, so the model can be moved to a new backend and checked there.

| | |
|---|---|
| **Model** | [`gridfm/genco-pf-datakit`](https://huggingface.co/gridfm/genco-pf-datakit), `case118_ieee/tiny/seed0`: `GNS_heterogeneous`, 12 layers, hidden 12, 8 heads, 1.28 M params |
| **Data** | [`gridfm/pf_small_case118_ieee`](https://huggingface.co/datasets/gridfm/pf_small_case118_ieee), IEEE 118-bus power-flow scenarios, the dataset the model was trained on |
| **Task** | AC power flow: given loads, generator set-points and the network, predict the bus voltages (magnitude, angle) and the slack/reactive generation |

## What runs

```
HF parquet (raw/)            bus / gen / branch tables, one row per element per scenario
  │ HeteroGridDatasetDisk    builds one HeteroData per scenario, cached as processed/*.pt (first run only)
  │ HeteroDataMVANormalizer  restored from the checkpoint's normalizer_stats.pt (never refitted)
  │ PowerFlow transforms     drop out-of-service elements, build the PF mask, overwrite masked inputs with 0
  ▼
torch_geometric DataLoader   the --scenarios ids, in that order → one block-diagonal graph
  ▼
GNS_heterogeneous.forward    ← the boundary to port
  ▼
{"bus": [N_bus, 4], "gen": [N_gen, 1]}
```

`load_bare_model` builds the model from the config through the registry and loads the checkpoint into it.
The release checkpoint is a Lightning task `state_dict`, so the script strips the `model.` key prefix first. Its
outputs match `task.model` on the CLI path exactly (max abs diff `0.0` on a batch of 8).

## Setup

Same environment as the rest of the repo (see the top-level README / `CLAUDE.md`). The script needs
`torch-scatter` and `huggingface_hub` (for the `hf` CLI) on top of the base install. `main` pins torch below 2.13
to match the prebuilt PyG wheels. On macOS arm64 there is no prebuilt wheel, so install with
`pip install torch-scatter --no-build-isolation`. That compiles against the installed torch, and you must rerun it
whenever torch changes version.

```bash
pip install -e .
TORCH_CUDA_VERSION=$(python -c "import torch; print(torch.__version__ + ('+cpu' if torch.version.cuda is None else ''))")
pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH_CUDA_VERSION}.html
pip install "huggingface_hub[cli]"
```

## Download

The full dataset is ~3.7 GB (219k scenarios across 1000 parquet partitions). A forward pass needs only a few
scenarios, so this fetches partition 0 (200 scenarios, 5 MB). Everything goes under `data/`, which is
git-ignored.

Both downloads are pinned to a commit with `--revision`, so the weights and scenario ids stay the same even if
the Hugging Face repos are updated.

```bash
hf download gridfm/genco-pf-datakit \
  --revision f5eb20ee798b9847b723c91874af85d32a96a468 \
  --include "case118_ieee/tiny/seed0/*" \
  --local-dir data/genco-pf-datakit

hf download gridfm/pf_small_case118_ieee --repo-type dataset \
  --revision 3f5a6e266f759be15e3777bf4d70133ff40e6fa7 \
  --include "bus_data.parquet/scenario_partition=0/*" \
  --include "gen_data.parquet/scenario_partition=0/*" \
  --include "branch_data.parquet/scenario_partition=0/*" \
  --local-dir data/genco_case118_p0/raw
```

Pass one `--include` per pattern. Putting several patterns after a single `--include` makes `hf` treat them as
file names and skip files. The script only reads `bus`, `gen` and `branch_data.parquet`.

## Run

```bash
M=data/genco-pf-datakit/case118_ieee/tiny/seed0
python examples/spyre/genco_forward.py \
  --config           $M/case118_ieee_tiny_seed0.yaml \
  --model_path       $M/best_model_state_dict.pt \
  --normalizer_stats $M/normalizer_stats.pt \
  --data_root        data/genco_case118_p0 \
  --device cpu --save_io data/genco_ref_io.pt
```

| Flag | Default | Meaning |
|---|---|---|
| `--data_root` | (required) | Directory holding `raw/`; `processed/` is written next to it on the first run |
| `--network` | `case118_ieee` | Key into `normalizer_stats.pt` |
| `--scenarios` | `0 1 2 3 4 5 6 7` | Scenario ids that make up the batch, in this order; the batch size is their count |
| `--device` | `cpu` | Any `torch` device string |
| `--repeats` | `10` | Timed forward passes after one warm-up |
| `--save_io` | off | Save the batch's input tensors, model outputs, scenario ids and fingerprint to a `.pt` file |

### Same scenarios on every run

Each run uses the scenarios listed in `--scenarios`, batched in exactly that order. Neither the order nor the
grouping can be left to chance. The batch is one combined graph, so grouping scenarios differently changes the
float summation order inside message passing and moves outputs by ~1e-6 relative. The script asserts the batch's
`scenario_id`s match the list. It also prints `inputs sha256 …`, a hash over every tensor the model reads.
The same hash on two machines means the model received an identical batch, so any output difference comes from
the backend. On the pinned revisions, the default scenarios give `c5c9512bc7d8e9a3`.

The release config is schema v0. The script migrates it in memory, so the "Converted v0 config -> v1" line is
expected. The `mask_ref_vm: true` warning is also expected: this checkpoint was trained with slack-bus Vm masked.

### Reference output (macOS arm64, CPU, torch 2.12, batch 8)

```
model    GNS_heterogeneous  1,276,611 params  layers=12 hidden=12 heads=8
dataset  200 scenarios in data/genco_case118_p0
batch    scenarios [0, 1, 2, 3, 4, 5, 6, 7] on cpu
inputs   sha256 c5c9512bc7d8e9a3
  x_dict          bus                                 (944, 15)     torch.float32
  x_dict          gen                                 (431, 6)      torch.float32
  edge_index_dict ('bus', 'connects', 'bus')          (2, 2962)     torch.int64
  edge_index_dict ('gen', 'connected_to', 'bus')      (2, 431)      torch.int64
  edge_index_dict ('bus', 'connected_to', 'gen')      (2, 431)      torch.int64
  edge_attr_dict  ('bus', 'connects', 'bus')          (2962, 10)    torch.float32
  mask_dict       bus                                 (944, 15)     torch.bool
  mask_dict       gen                                 (431, 6)      torch.bool
  mask_dict       branch                              (2962, 10)    torch.bool
  mask_dict       PQ                                  (944,)        torch.bool
  mask_dict       PV                                  (944,)        torch.bool
  mask_dict       REF                                 (944,)        torch.bool
output   bus (944, 4) [VM, VA, PG, QG]   gen (431, 1) [PG]
latency  59.79 ms / forward (mean of 10, after 1 warm-up)
rmse     (normalized, masked entries only)
  VM 0.00114   VA 0.00445   QG 0.05142   PG(slack) 0.03780
  mean |power-balance residual| at last layer: 0.00074
```

All errors are in normalized units: VM in p.u., VA in radians, PG/QG in p.u. of the fitted `baseMVA` (137.9).
They are computed only on the entries the model has to predict. Read them as a smoke test: a broken port shows up
as VM/VA errors orders of magnitude larger, or a residual far above ~1e-3. A full evaluation is
`gridfm_graphkit evaluate` (see the model card).

## Agreement with `gridfm_graphkit predict`

The script's outputs are bit-identical to the CLI's `predict` route when both see the same batch:

| Comparison | Vm | Va | Pg | Qg |
|---|---|---|---|---|
| Script vs `predict` on CPU, same scenarios and order | 0 | 0 | 0 | 0 |
| Script vs `predict` on CPU, same scenarios grouped differently | 4e-6 | 5e-6 | 6e-3 MW | 2e-2 MVar |
| `predict` on CPU vs `predict` on MPS | 6e-6 | 5e-6 | 7e-3 MW | 3e-2 MVar |

Max absolute difference over the 22 test scenarios of partition 0, batch size 8. Two things differ between
the routes by default:

- **Scenarios.** `predict` runs only on the test split, which it draws with a seeded shuffle. On partition 0
  that is ids 110–131, and its first batch is `131 110 117 112 116 123 118 128`. Passing those ids to
  `--scenarios` reproduces that batch exactly.
- **Units.** The script prints normalized outputs. `predict` multiplies PG/QG by `baseMVA` (137.93) to get
  MW/MVar. VM (p.u.) and VA (rad) are the same in both.

The last row gives a sense of scale for comparing backends: two PyTorch backends on the same batch already
differ by ~1e-5 relative.

## What the model receives and returns

`model(batch)` reads exactly four attributes of the PyG batch, and all of them are saved by `--save_io`:

| Input | Shape (batch 8) | Content |
|---|---|---|
| `x_dict["bus"]` | `[944, 15]` f32 | Pd, Qd, Qg, Vm, Va, PQ/PV/REF one-hot, Vm/Qg limits, Gs, Bs, kV (`datasets/globals.py`); masked entries are 0 |
| `x_dict["gen"]` | `[431, 6]` f32 | Pg, Pg limits, cost coefficients; slack-generator Pg masked |
| `edge_index_dict[...]` | `[2, E]` i64 | bus↔bus branches (both directions), gen→bus, bus→gen |
| `edge_attr_dict[("bus","connects","bus")]` | `[2962, 10]` f32 | flows (masked), admittances Yff/Yft, tap, angle limits, rating |
| `mask_dict` | bool | per-feature masks for `bus`, `gen`, `branch`, and per-bus `PQ`/`PV`/`REF` flags |

Outputs: `bus` `[N_bus, 4]` = `[VM, VA, PG, QG]`, where PG/QG come from the physics decoder (power balance), not
a learned head. `gen` `[N_gen, 1]` = per-generator PG.

## Things that matter for Spyre

- **Shapes vary per scenario.** The dataset's topology perturbation takes one branch or one generator out of
  service per scenario. Partition 0 has two graph shapes: `(118 bus, 54 gen, 370 arcs)` × 158 and
  `(118, 53, 372)` × 42. A batch's `N_gen` and `E` depend on which scenarios it holds. For a static-shape
  compile, either batch scenarios of one shape (filter the dataset on `num_nodes` / `edge_index.shape`) or pad
  the out-of-service elements back in with zeroed features.
- **Ops in the forward:** `Linear`, `LayerNorm`, `LeakyReLU`, PyG `TransformerConv` (gather by `edge_index`,
  per-destination softmax, scatter-add aggregation) inside `HeteroConv(aggr="sum")`, plus at every one of the 12
  layers the physics block in `models/utils.py`: `cos`/`sin` gathers over branches, `torch_scatter.scatter_add`
  (= `Tensor.scatter_add_`) onto buses, and `torch.where` on the masks. The gather/scatter/segment-softmax ops are
  the ones to check first on a new backend.
- **Golden reference.** Run once on CPU with `--save_io`, then run on Spyre with the same `--scenarios`. Check
  that the `inputs sha256` lines match, then compare the outputs against the saved `outputs` tensors.
- `--device` accepts any `torch` device string. A Spyre-specific path (a `torch.compile` backend or device
  name) is not wired in yet.
