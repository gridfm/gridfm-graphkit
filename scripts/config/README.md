# Configuration Parameters

This document provides a detailed explanation of the parameters used in the YAML configuration file. The configuration is structured into different sections for ease of use.

---


## Verbose
- **`verbose`**: *(Boolean)* Provides detailed analysis after training.
  - Default: `False`

## Data
- **`networks`**: *(List of Strings)* Specifies the network topologies to use during training
  - Example: `["case300_ieee", "case30_ieee"]`

- **`scenarios`**: *(List of Integers)* Defines the number of scenarios to use for each network specified.
  - Example: `[8500, 4000]`

- **`normalization`**: *(String)* Normalization method for data.
  - Options:
    - `minmax`: Scales data between the minimum and maximum values.
    - `standard`: Standardizes data to have zero mean and unit variance.
    - `baseMVAnorm`: Divides data by a baseMVA value, which is the maximum active/reactive power across the network.
    - `identity`: Leaves data unchanged.
  - Example: `"baseMVAnorm"`

- **`baseMVA`**: *(Integer)* The base MVA value specified in the original matpower casefile, needed for `baseMVAnorm` normalization.
  - Default: `100`

- **`mask_value`**: *(Float)* Value used to mask data during training.
  - Default: `0.0`

- **`mask_ratio`**: *(Float)* Propability of each feature to be masked.
  - Default: `0.5`

- **`mask_dim`**: *(Integer)* Number of features to mask.
  - Default: `6` (Pd, Qd, Pg, Qg, Vm, Va)

- **`learn_mask`**: *(Boolean)* Specifies whether the mask value is learnable.
  - Default: `False`

- **`val_ratio`**: *(Float)* Fraction of data used for validation.
  - Default: `0.1`

- **`test_ratio`**: *(Float)* Fraction of data used for testing.
  - Default: `0.1`

## Model
- **`type`**: *(String)* Specifies the type of model architecture.
  - Example: `"GPSconv"`

- **`input_dim`**: *(Integer)* Input dimensionality of the model.
  - Default: `9` (Pd, Qd, Pg, Qg, Vm, Va, PQ, PV, REF)

- **`output_dim`**: *(Integer)* Output dimensionality of the model.
  - Default: `6` (Pd, Qd, Pg, Qg, Vm, Va)

- **`edge_dim`**: *(Integer)* Dimensionality of edge features.
  - Default: `2` (G, B)

- **`pe_dim`**: *(Integer)* Dimensionality of positional encoding.
  - Example: `20` (Length of random walk)

- **`num_layers`**: *(Integer)* Number of layers in the model.
  - Example: `6`

- **`hidden_size`**: *(Integer)* Size of hidden layers.
  - Example: `256`

- **`attention_head`**: *(Integer)* Number of attention heads in the model.
  - Example: `8`

- **`dropout`**: *(Float)* Model dropout probability
  - Default: `0.0`

## Training
- **`batch_size`**: *(Integer)* Number of samples per training batch.
  - Example: `16`

- **`epochs`**: *(Integer)* Number of training epochs.
  - Example: `100`

- **`losses`**: *(List of Strings)* Specifies the loss functions to use during training.
  - Available options:
    - `MSE`: Mean Squared Error.
    - `MaskedMSE`: Masked Mean Squared Error.
    - `SCE`: Scaled Cosine Error.
    - `PBE`: Power Balance Equation loss.
  - Example: `["MaskedMSE", "PBE"]`

- **`loss_weights`**: *(List of Floats)* Specifies the relative weights for each loss function.
  - Example: `[0.01, 0.99]`

## Optimizer
- **`learning_rate`**: *(Float)* Learning rate for the optimizer.
  - Example: `0.0001`

- **`beta1`**: *(Float)* Beta1 parameter for the Adam optimizer.
  - Default: `0.9`

- **`beta2`**: *(Float)* Beta2 parameter for the Adam optimizer.
  - Default: `0.999`

- **`lr_decay`**: *(Float)* Learning rate decay factor.
  - Example: `0.7`

- **`lr_patience`**: *(Integer)* Number of epochs to wait before applying learning rate decay.
  - Example: `3`

## Callbacks
- **`patience`**: *(Integer)* Number of epochs to wait before early stopping. A value of `-1` disables early stopping.
  - Default: `-1`

- **`tol`**: *(Float)* Tolerance for validation loss comparison during early stopping.
  - Default: `0`

---

