# YAML configuration reference

Every experiment is driven by one YAML file in `examples/config/`.


## Full example (current style)

```yaml
task:
  task_name: OptimalPowerFlow
data:
  baseMVA: 100
  mask_value: 0.0
  normalization: HeteroDataMVANormalizer
  networks:
  - case14_ieee
  scenarios:
  - 300000
  workers: 32
  split_by_load_scenario_idx: false
  split_from_existing_files: "/dccstor/gridfm/march_opf_exp/opfdata_olay_splits/"
model:
  attention_head: 8
  edge_dim: 10
  hidden_size: 48
  input_bus_dim: 15
  input_gen_dim: 6
  output_bus_dim: 2
  output_gen_dim: 1
  num_layers: 12
  type: GNS_heterogeneous
optimizer:
  beta1: 0.9
  beta2: 0.999
  learning_rate: 0.0005
  lr_decay: 0.7
  lr_patience: 5
training:
  batch_size: 64
  epochs: 200
  loss_weights: [0.1, 0.1, 0.75, 0.001]
  losses: [LayeredWeightedPhysics, MaskedGenMSE, MaskedBusMSE, QgViolationPenalty]
  loss_args:
  - base_weight: 0.5
  - {}
  - {}
  - {}
  accelerator: auto
  devices: auto
  strategy: auto
seed: 0
verbose: true
callbacks:
  patience: 100
  tol: 0
```

---

## Top-level keys

- `task`: task-specific settings (`OptimalPowerFlow` or `PowerFlow`).
- `data`: dataset selection, normalization, splits, and loading behavior.
- `model`: model architecture and dimensions.
- `optimizer`: optimizer and scheduler parameters.
- `training`: epochs, loss composition, and accelerator strategy.
- `callbacks`: early stopping behavior.
- `seed`: random seed used for reproducible shuffling/splits.
- `verbose`: enables extra outputs (for example additional test plots/log artifacts).

---

## `task` section

### `task.task_name`

Task name registered in the framework:

- `OptimalPowerFlow`
- `PowerFlow`

---

## `data` section

### `data.networks`

List of dataset folders under your data root.
Examples: `case14_ieee`, `case118_ieee`, `case2000_goc`, `Texas2k_case1_2016summerpeak`.

### `data.scenarios`

List of scenario counts, one value per network in `data.networks`.
Example: with two networks, use two scenario entries in matching order.

### `data.normalization`

Normalizer class name:

- `HeteroDataMVANormalizer`: fit one normalization scale from training data.
- `HeteroDataPerSampleMVANormalizer`: fit per-scenario scales across the selected dataset.

### `data.baseMVA`

Base MVA reference value (default in examples: `100`). Used by normalizers for per-unit scaling.

### `data.mask_value`

Fill value used when masking unavailable measurements/features (examples use `0.0`).

### `data.test_ratio`, `data.val_ratio`

Fractions for validation and test splits when split files are not supplied.

### `data.workers`

Number of dataloader workers.

### `data.split_by_load_scenario_idx`

- `true`: split train/val/test by load scenario identifiers.
- `false`: perform standard random split.

### `data.split_from_existing_files`

Optional path to precomputed split files. When provided:

- split IDs are loaded from this folder,
- `data.scenarios` is ignored for split construction,
- do **not** combine with `split_by_load_scenario_idx: true`.

## `model` section

Current configs use the heterogeneous GNS model:

- `type`: model registry name (examples use `GNS_heterogeneous`).
- `input_bus_dim`: bus-node input feature dimension.
- `input_gen_dim`: generator-node input feature dimension.
- `output_bus_dim`: bus-node output dimension.
- `output_gen_dim`: generator-node output dimension.
- `edge_dim`: edge feature dimension.
- `hidden_size`: hidden feature width.
- `num_layers`: number of stacked message-passing layers.
- `attention_head`: attention head count per layer.

---

## `training` section

### Core training controls

- `batch_size`: mini-batch size.
- `epochs`: number of epochs.
- `accelerator`: Lightning accelerator (`auto`, `mps`, `cpu`, `gpu`, etc.).
- `devices`: Lightning device selection (`auto`, integer, list).
- `strategy`: Lightning strategy (`auto`, `ddp`, etc.).

### Multi-loss configuration

- `losses`: list of registered loss names.
- `loss_weights`: scalar weight per loss.
- `loss_args`: list of argument objects matching `losses` by position.

All three lists must be aligned (same length and same order).

Registered loss names in current code:

- `LayeredWeightedPhysics`
- `MaskedGenMSE`
- `MaskedBusMSE`
- `QgViolationPenalty`
- `LossPerDim`
- `MaskedMSE`
- `MSE`

Common `loss_args` patterns:

- `LayeredWeightedPhysics`: `{base_weight: <float>}`
- `LossPerDim`: `{dim: VM|VA|P_in|Q_in, loss_str: MAE|MSE}`
- `MaskedGenMSE`, `MaskedBusMSE`, `QgViolationPenalty`, `MaskedMSE`, `MSE`: `{}`

---

## `optimizer` section

- `learning_rate`: initial learning rate.
- `beta1`, `beta2`: Adam betas.
- `lr_decay`: scheduler decay factor (e.g., ReduceLROnPlateau factor).
- `lr_patience`: epochs to wait before applying LR decay.

---

## `callbacks` section

- `patience`: early stopping patience (epochs without sufficient improvement).
- `tol`: minimum required improvement threshold to reset patience.

---

## Practical validation checklist

Before launching a run, verify:

- `len(data.networks) == len(data.scenarios)`.
- `len(training.losses) == len(training.loss_weights) == len(training.loss_args)`.
- `split_by_load_scenario_idx` and `split_from_existing_files` are not both active.
