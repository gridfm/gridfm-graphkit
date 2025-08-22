# The YAML configuration file

Every experiment in **`gridfm-graphkit`** is defined through a single YAML configuration file.
This file specifies which networks to load, how to normalize the data, which model architecture to build, and how different stages of the workflow should be executed.

Rather than modifying the source code, you simply adjust the YAML file to describe your experiment. This approach makes results reproducible and easy to share: all the important details are stored in one place.

The configuration is divided into sections (`data`, `model`, `training`, `optimizer`, etc.), with each section grouping related options.
We will explain these fields one by one and show how to use them effectively.

For ready-to-use examples, check the folder [**`examples/config/`**](https://github.com/gridfm/gridfm-graphkit/tree/main/examples/config), which contains valid configuration files you can adapt for your own experiments.

---

## Data

The `data` section defines **which networks and scenarios to use**, as well as **how to prepare and mask the input features**.

Example:

```yaml
data:
  networks: ["case300_ieee", "case30_ieee"]
  scenarios: [8500, 4000]
  normalization: baseMVAnorm
  baseMVA: 100
  mask_type: rnd
  mask_value: 0.0
  mask_ratio: 0.5
  mask_dim: 6
  learn_mask: false
  val_ratio: 0.1
  test_ratio: 0.1
  workers: 4
```

**Key fields:**

- **`networks`**: List of network topologies (e.g., IEEE test cases) used.
- **`scenarios`**: Number of scenarios (samples) for each network.
- **`normalization`**: Method to scale features. Options:
    - `minmax`: scale between min and max.
    - `standard`: zero mean, unit variance.
    - `baseMVAnorm`: divide by base MVA value (see `baseMVA`).
    - `identity`: no normalization.
- **`baseMVA`**: Base MVA value from the case file (default: 100).
- **`mask_type`**: Defines how input features are masked:
    * `rnd` = random masking (controlled by `mask_ratio` and `mask_dim`).
    * `pf` = power flow problem setup.
    * `opf` = optimal power flow setup.
    * `none` = no masking.
* **`mask_value`**: Numerical value used to mask inputs (default: 0.0).
* **`mask_ratio`**: Probability of masking a feature (only used when `mask_type=rnd`).
* **`mask_dim`**: Number of features that can masked (default: the first 6 → Pd, Qd, Pg, Qg, Vm, Va).
* **`learn_mask`**: If true, the mask value becomes learnable.
* **`val_ratio` / `test_ratio`**: Fractions of the dataset used for validation and testing.
* **`workers`**: Number of data-loading workers

---

## Model

The `model` section specifies the neural network architecture and its hyperparameters.

Example:

```yaml
model:
  type: GPSconv
  input_dim: 9
  output_dim: 6
  edge_dim: 2
  pe_dim: 20
  num_layers: 6
  hidden_size: 256
  attention_head: 8
  dropout: 0.0
```

**Key fields:**

* **`type`**: Model architecture (e.g., `"GPSconv"`).
* **`input_dim`**: Input feature dimension (default: 9 → Pd, Qd, Pg, Qg, Vm, Va, PQ, PV, REF).
* **`output_dim`**: Output feature dimension (default: 6 → Pd, Qd, Pg, Qg, Vm, Va).
* **`edge_dim`**: Dimension of edge features (default: 2 → G, B).
* **`pe_dim`**: Size of positional encoding (e.g., random walk length).
* **`num_layers`**: Number of layers in the network.
* **`hidden_size`**: Width of hidden layers.
* **`attention_head`**: Number of attention heads.
* **`dropout`**: Dropout probability (default: 0.0).

---

## Training

The `training` section defines how the model is optimized and which loss functions are used.

Example:

```yaml
training:
  batch_size: 16
  epochs: 100
  losses: ["MaskedMSE", "PBE"]
  loss_weights: [0.01, 0.99]
  accelerator: auto
  devices: auto
  strategy: auto
```

**Key fields:**

* **`batch_size`**: Number of samples per training batch.
* **`epochs`**: Number of training epochs.
* **`losses`**: List of losses to combine. Options:
    * `MSE` = Mean Squared Error.
    * `MaskedMSE` = Masked Mean Squared Error.
    * `SCE` = Scaled Cosine Error.
    * `PBE` = Power Balance Equation loss.
* **`loss_weights`**: Relative weights applied to each loss term.
* **`accelerator`**: Device type used for training (cpu, gpu, mps, or auto).
* **`devices`**: Number of devices (GPUs/CPUs) to use (or auto)
* **`strategy`**: Training strategy (e.g., ddp for distributed data parallel, or auto).

!!! note
    On macOS, using accelerator: `cpu` is often the most stable choice.
---

## Optimizer

Defines the optimizer and learning rate scheduling.

Example:

```yaml
optimizer:
  learning_rate: 0.0001
  beta1: 0.9
  beta2: 0.999
  lr_decay: 0.5
  lr_patience: 3
```

**Key fields:**

* **`learning_rate`**: Initial learning rate.
* **`beta1`**, **`beta2`**: Adam optimizer parameters (defaults: 0.9, 0.999).
* **`lr_decay`**: Factor to decay the learning rate.
* **`lr_patience`**: Number of epochs to wait before reducing the LR.

---

## Callbacks

Callbacks add additional behavior during training, such as early stopping.

Example:

```yaml
callbacks:
  patience: 100
  tol: 0
```

**Key fields:**

* **`patience`**: Number of epochs to wait before early stopping.
* **`tol`**: Minimum improvement required in validation loss to reset patience.

---
