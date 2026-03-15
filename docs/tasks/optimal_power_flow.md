# Optimal Power Flow Task

The `OptimalPowerFlowTask` class is a concrete implementation of `ReconstructionTask` for **optimal power flow (OPF)**. It adds economic optimization metrics (generation cost, optimality gap), constraint violation tracking (thermal, voltage angle, reactive power limits), and the same physics-based validation as the power flow task.

## Overview

`OptimalPowerFlowTask` extends `ReconstructionTask` and provides:

- **Economic metrics**: Generation cost from quadratic cost coefficients (C0, C1, C2) and **optimality gap** (relative difference between predicted and ground-truth cost)
- **Constraint violations**: Branch thermal limits (RATE_A), branch angle limits (ANG_MIN, ANG_MAX), and reactive power limits (Qg min/max) for PV and REF buses
- **Physics validation**: Same branch flow, node injection, and power balance residuals as PowerFlowTask
- **Per-bus-type MSE**: Separate MSE for PQ, PV, and REF buses (PG, QG, VM, VA)

## OptimalPowerFlowTask Class

::: gridfm_graphkit.tasks.opf_task.OptimalPowerFlowTask
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - test_step
        - on_test_end

## Configuration Example

Use the task by setting `task_name: OptimalPowerFlow` in your YAML:

```yaml
task:
  task_name: OptimalPowerFlow

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

During evaluation, `OptimalPowerFlowTask` logs (per dataset):

| Metric | Description |
|--------|-------------|
| Test loss | Main reconstruction loss |
| Opt gap | Mean absolute percentage difference between predicted and ground-truth generation cost |
| MSE PG | MSE on generator active power |
| Active / Reactive Power Loss | Mean absolute P/Q residuals |
| Branch thermal violation from | Mean thermal limit excess on forward branch (apparent power vs RATE_A) |
| Branch thermal violation to | Mean thermal limit excess on reverse branch (apparent power vs RATE_A) |
| Branch voltage angle difference violations | Mean angle limit violation (degrees) |
| Mean Qg violation PV buses | Mean reactive power limit violation on PV buses |
| Mean Qg violation REF buses | Mean reactive power limit violation on REF buses |
| MSE PQ/PV/REF nodes - PG/QG/VM/VA | MSE per bus type and output dimension |

With `verbose: true`, CSV reports and plots are written to MLflow artifacts.

## Related

- [Reconstruction Task](reconstruction_task.md): Base class for reconstruction tasks
- [Power Flow Task](power_flow.md): Power flow analysis (no cost or constraint metrics)
- [Base Task](base_task.md): Abstract base class for all tasks
- [Loss Functions](../training/loss.md): Available loss functions
