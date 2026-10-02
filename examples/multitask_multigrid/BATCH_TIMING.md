# Batch layout timing

Time two optimizer-step layouts on one cluster GPU. This is a timing test only. Do not start the 50-epoch runs in `AGENT.md`. Do not edit `gridfm_graphkit/models/gnn_heterogeneous_gns.py`. Do not change the training code from the result.

Check out branch `multitask-multigrid`. From the repo root:

```bash
export PYTHONPATH="$PWD"
```

## Model

Use `GNS_heterogeneous_MultiTask` with the trunk from `pf_opf_case14_30_57.yaml`: hidden size 12, 12 layers, 8 heads, edge dim 10, bus input 15, generator input 6, bus output 2, generator output 1. AdamW at 5e-4, betas 0.9 and 0.999. Train mode.

Each microbatch is one task, so call `set_task` and then the model forward. That path is the plain GENCO forward. Do not use a mixed-task batch here.

## Graphs

Prefer real case14, case30, and case57 graphs from the power-flow and OPF trees, after the masks for that task. If those files are not on the machine, build synthetic graphs with these sizes:

| Grid | Buses | Generators | Undirected branches |
|---|---:|---:|---:|
| case14 | 14 | 5 | 20 |
| case30 | 30 | 6 | 41 |
| case57 | 57 | 7 | 80 |

Bus features are 15-wide, generator features are 6-wide, branch features are 10-wide. Store both directions of each branch. Put the graphs on the GPU before timing.

## The two layouts

Both layouts do one AdamW step. Scale each microbatch loss by `1 / number of microbatches`, call `backward` on each, then `optimizer.step` once.

1. **Two mixed-grid forwards.** Batch size 64. One microbatch of 32 power-flow graphs and one of 32 OPF graphs. Each 32 contains case14, case30, and case57. Draw a new grid mix every step, including lopsided mixes. A fixed 11/11/10 mix is not this layout.

2. **Six homogeneous forwards.** Batch size 60. Six microbatches of 10 graphs. Each microbatch is one grid and one task: case14 PF, case30 PF, case57 PF, case14 OPF, case30 OPF, case57 OPF. Reuse the same ten graphs so the tensor shape of each microbatch stays fixed across steps.

Loss can be the mean square of the bus and generator outputs. The GNN dominates the step.

## How to time

Warm up 2 steps, then time 8 steps. Synchronize the GPU before the timed loop and after it. Divide by 8. Run that measurement a second time and report both. Print the GPU name.

Report milliseconds per optimizer step for layout 1 and layout 2. Also report which layout is faster and by how much. Do not copy a timing from another machine into the result.
