# Power Flow Task

The `PowerFlowTask` class is a concrete implementation of `ReconstructionTask` for **power flow analysis**. It computes voltage profiles and power flows from given injections and adds physics-based validation using Power Balance Error (PBE) and per-bus-type metrics.

## Overview

`PowerFlowTask` extends `ReconstructionTask` and provides:

- **Physics-based validation**: Branch flow computation, node injection, and power balance residuals (P, Q)
- **Per-bus-type metrics**: Separate MSE and residual statistics for PQ, PV, and REF buses (PG, QG, VM, VA)
- **Power Balance Error (PBE)**: Mean and max PBE across the test set
- **Optional verbose output**: Residual histograms and correlation plots when `args.verbose` is true

## PowerFlowTask Class

::: gridfm_graphkit.tasks.pf_task.PowerFlowTask
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - test_step
        - on_test_end

## Configuration Example

Use the task by setting `task_name: PowerFlow` in your YAML:

```yaml
task:
  task_name: PowerFlow

model:
  type: GNS_heterogeneous
  hidden_size: 48
  num_layers: 12
  attention_head: 8

training:
  batch_size: 64
  epochs: 100
  losses:
    - MaskedMSE
  loss_weights:
    - 1.0

data:
  networks:
    - case14_ieee
    - case118_ieee

verbose: true
```

## Test Metrics

During evaluation, `PowerFlowTask` logs (per dataset):

| Metric | Description |
|--------|-------------|
| Test loss | Main reconstruction loss |
| Active Power Loss | Mean absolute active power residual |
| Reactive Power Loss | Mean absolute reactive power residual |
| PBE Mean | Mean power balance error magnitude |
| PBE Max | Maximum power balance error (reduced with max across batches) |
| MSE PQ/PV/REF nodes - PG/QG/VM/VA | MSE per bus type and output dimension |

With `verbose: true`, CSV reports and residual histograms are written to MLflow artifacts.

## Related

- [Reconstruction Task](reconstruction_task.md): Base class for reconstruction tasks
- [Base Task](base_task.md): Abstract base class for all tasks
- [Loss Functions](../training/loss.md): Available loss functions (e.g. MaskedMSE, LayeredWeightedPhysics)
