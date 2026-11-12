# Add HH-MPNN model and effective-resistance positional encoding

## Summary

Adds HH-MPNN (Arowolo & Cremer, [arXiv:2510.06860](https://arxiv.org/abs/2510.06860)), the OPF baseline that the GENCO paper compares against, as a registered model: `model.type: HH_MPNN`. It comes with the positional encoding the model needs, as a registered transform.

The change is additive. No existing model, task, loss, datamodule or CLI code path is modified, and there are no new dependencies.

## Changes

| File | Change |
|---|---|
| `gridfm_graphkit/models/hh_mpnn.py` (new) | `HHMPNN` registered as `HH_MPNN`, with its layers `MLP`, `HeteroInteractionNetwork`, `HeteroPerformerLayer`, `MaskedPerformerAttention`. |
| `gridfm_graphkit/datasets/effective_resistance.py` (new) | `effective_resistance_pe`, `connected_components` and the `EffectiveResistancePE` transform. |
| `gridfm_graphkit/models/__init__.py` | Imports and exports `HHMPNN`, which registers the model. |
| `gridfm_graphkit/datasets/task_transforms.py` | Registers `EffectiveResistancePE` in `TRANSFORM_REGISTRY` with the other primitive transforms. |
| `examples/config/HHMPNN_OPF_datakit_case14.yaml` (new) | Example OPF config. |
| `tests/config/hh_mpnn_test_config.yaml` (new) | Small config used by the tests. |
| `tests/test_hh_mpnn.py`, `tests/test_effective_resistance.py` (new) | 36 and 18 tests. |
| `docs/quick_start/yaml_config.md` | Documents `data.transforms`, `EffectiveResistancePE` and the `HH_MPNN` model keys. |

## Model

Each of the `num_layers` layers runs a heterogeneous interaction network (type-specific edge and node MLPs, messages summed into both end nodes of an edge) and Performer attention over all nodes of a graph, then merges the two. Node states start from type-specific encoders concatenated with the 5-dimensional positional encoding. The layers were ported from the authors' training script; `test_layers_match_the_original_implementation` loads the same weights into verbatim copies of the original layers and checks equal node states.

Config keys under `model`: `hidden_size`, `num_layers`, `input_bus_dim`, `input_gen_dim`, `edge_dim`, `attention_head` (default 1), and two switches:

- `load_shunt_nodes` (default `true`). `true` rebuilds the load and shunt node types of the original HH-MPNN inside the model: one load node per bus with non-zero `Pd`/`Qd`, one shunt node per bus with non-zero `GS`/`BS`, each joined to its bus by a connector edge, and those four columns removed from the bus features. `false` runs on graphkit's `bus` and `gen` nodes only.
- `physics_decoder` (default `false`). `false` is the original HH-MPNN decoding: `Vm`, `Va`, `Qg` per bus and `Pg` per generator are predicted directly and bounded by their limits with a sigmoid. It supports `OptimalPowerFlow` only and raises a `ValueError` for other tasks. `true` predicts `Vm`, `Va`, `Pg` and obtains the remaining quantities from the task's physics decoder, as `GNS_heterogeneous` does, which supports `PowerFlow`, `OptimalPowerFlow` and `StateEstimation`.

Output follows the existing contract: `{"bus": [N, 4] = (Vm, Va, Pg, Qg), "gen": [Ng, 1] = Pg}`, and `forward(batch, return_embeddings=True)` also returns the bus and generator states fed to the output heads.

## Positional encoding

`EffectiveResistancePE` stores `data["bus"].pestat_ER`: per bus, the mean, median, standard deviation, maximum and minimum of its effective resistance to the other buses. The graph is the in-service bus graph weighted by series susceptance, `|Im(Yft)| * tap`, which equals `x / (r^2 + x^2)` for branches without phase shift.

Graph structure is not constant across samples: `tests/data/case14_ieee` has 72 scenarios and 6 distinct in-service topologies. The transform therefore caches by a hash of bus count, in-service edges and weights. That gives one computation for a fixed grid and one per distinct topology otherwise (6 for the test data, checked by `test_one_computation_per_distinct_topology`). The cache is in memory per process, plus an optional on-disk cache (`cache_dir`) with atomic writes. Islanded components are encoded separately and an isolated bus gets zeros.

It is used through `data.transforms`, after the task's default steps:

```yaml
data:
  transforms:
  - RemoveInactiveBranches
  - RemoveInactiveGenerators
  - AddOPFHeteroMask
  - ApplyMasking
  - EffectiveResistancePE
```

## Things to discuss with reviewers

1. **Graph schema (`load_shunt_nodes`).** graphkit stores loads and shunts as bus columns; but HH-MPNN uses separate node types. The default follows the original HH-MPNN. Are you okay with the default this repo, or should the native bus + gen graph be the default, or the only option? I choose the option to be as close to the original HH-MPNN as possible
2. **Reactive power (`physics_decoder`).** graphkit has no per-generator `Qg` or Q limits in the graph, so the bounded `Qg` head sits on the bus and uses the bus-aggregated limits `MIN_QG_H`/`MAX_QG_H`. The original HH-MPNN predicts `Qg` per generator. Should both decoding modes stay? The dafault behaviour right now does not use the physics decoding, again to be faithful to the original HH-MPNN implementation
3. **PE through `data.transforms`.** This follows PR#149 and needs no datamodule changes, but a config has to restate the default pipeline. The alternative is a `data.posenc_ER` block in the datamodule like `posenc_RWSE`/`posenc_RRWP`. Those are applied before `RemoveInactiveBranches`: on the case14 test data the RWSE encoding is identical for all 72 scenarios although their in-service topologies differ. That is outside this PR.
4. **Branch edges.** graphkit stores each branch in both directions with direction-specific features, and has a single branch edge type. The model uses both directions and sums every message into both end nodes, so each bus receives two messages per branch. The original HH-MPNN has one edge per branch and separate line and transformer types.
5. **Base of the PE.** Weights are read after normalization, so the encoding is on the normalizer's power base, like the admittance features. The cache key therefore changes if the normalizer statistics change.
6. **Voltage angle bound.** For OPF, `Va` is bounded to ±2 rad (`VA_BOUND`), as in the original HH_MPNN implementation.

## Testing

Run on CPU, Python 3.12, torch 2.12.1, torch-geometric 2.8.1:

- `MLFLOW_ALLOW_FILE_STORE=true pytest tests/`: 235 passed (178 before this change).
- `ruff` 0.12.0 check and format, `flake8`, `add-trailing-comma` with the arguments from `.pre-commit-config.yaml`: clean on the changed Python files.
- `mkdocs build`: succeeds.

The new tests cover: all eight supported (task, `load_shunt_nodes`, `physics_decoder`) combinations for output shape and for every parameter receiving a gradient (the CLI runs DDP with `find_unused_parameters=False`); CLI `train` in the same eight modes; CLI `evaluate` and `predict --get_embeddings`; limits respected with saturated heads; task inputs passed through; batch independence; bfloat16; the PE against closed forms, against the original script's NumPy implementation and against raw `r`, `x` from the parquet files; cache behaviour in memory and on disk.

Integration tests: not run on the standard 10,000-scenario datasets, but was run on the 72-scenario `tests/data/case14_ieee` set 

