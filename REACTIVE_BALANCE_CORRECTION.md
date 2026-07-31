# Generating a dataset with corrected reactive-power balance

This explains how to build a dataset whose **ground-truth** reactive-power balance has
been reconciled per bus, using the offline preprocessing script. It is opt-in: nothing
changes unless you explicitly ask for the correction.

## Why

Some raw scenarios carry a reactive-power (Q) imbalance in the ground truth — i.e. the
recorded voltages, loads, generation and shunts don't perfectly satisfy the nodal
reactive balance. The per-bus residual is exactly what the training loss computes:

```
residual_Q = Qg - Qd + q_shunt - Q_in
```

where `Q_in` is the reactive power flowing out through the branches (a function of
voltages `Vm`, `Va` and branch admittances only). Because `Q_in` does **not** depend on
`Qd`/`Qg`, we can drive `residual_Q → 0` on every bus by absorbing the mismatch into the
reactive load `Qd` (or into the generator `Qg`). This:

- leaves **voltage magnitudes and angles untouched** (no power-flow re-solve needed),
- leaves **branch flows and active-power balance untouched**,
- is **local** — each bus's correction affects only that bus, not its neighbours.

The correction is done **once, at dataset-creation time**, and cached to disk, so there is
no per-epoch cost during training.

## What stays the default (nothing changes unless you ask)

- The normal training dataset build (`HeteroGridDatasetDisk.process()`) does **not** apply
  any correction. Datasets built the usual way are byte-for-byte identical to before.
- The offline script `scripts/process_hetero_dataset_parallel.py` defaults to
  `--reactive-correction none`, i.e. also no correction. You must pass a mode explicitly.

## How to generate a corrected dataset

Run the offline preprocessing script on a dataset root that contains `raw/` (and where
`processed/` will be written):

```bash
python scripts/process_hetero_dataset_parallel.py /path/to/dataset_root \
    --reactive-correction qd_all \
    --force
```

- `/path/to/dataset_root` — the folder containing `raw/` (the parquet tables) and
  `processed/` (created if absent). Example: `.../case14_ieee` or `.../case10000_goc`.
- `--force` — run even if a `processed_raw_files.done` marker already exists. Use this when
  regenerating an already-processed dataset.
- `--no-skip-existing` — also recompute individual `data_index_<scenario>.pt` files that
  already exist. Combine with `--force` to fully rebuild a corrected dataset from scratch.
- `--workers N` — parallel scenario workers per partition (defaults to CPU count).

> **Tip:** to build a corrected dataset without disturbing your existing one, copy the
> dataset's `raw/` into a new root first, then run the script there:
> ```bash
> mkdir -p /path/to/corrected_root && cp -r /path/to/dataset_root/raw /path/to/corrected_root/
> python scripts/process_hetero_dataset_parallel.py /path/to/corrected_root \
>     --reactive-correction qd_all --force
> ```

### Correction modes

| Mode              | What it does                                                                 |
|-------------------|------------------------------------------------------------------------------|
| `none` (default)  | No correction. Dataset built as-is.                                          |
| `qd_all`          | Absorb the residual into `Qd` on **every** bus.                              |
| `qd_pq_qg_pvref`  | Absorb into `Qd` on **PQ** buses and into `Qg` on **PV/REF** buses.          |

Both `qd_all` and `qd_pq_qg_pvref` drive the reactive residual to ~0 and leave voltages
untouched. They differ only on generator (PV/REF) buses:

- `qd_all` writes the whole correction into the **load** column `Qd`, even on generator
  buses. Simpler; but on a PV/REF bus the recorded `Qd` then includes reactive power that
  is physically generator output. Prefer this when `Qd` on generator buses is not used as
  a meaningful input feature.
- `qd_pq_qg_pvref` keeps `Qd` meaning "load" everywhere and lets the generator column `Qg`
  carry the correction on PV/REF buses. This matches how the PF physics decoder already
  treats these buses (`Qg = Q_in + Qd - q_shunt`). Prefer this if the model reads `Qd` on
  PV/REF buses as a genuine load/boundary condition.

## Provenance marker

When (and only when) a correction is applied, the script writes a marker file so a
corrected dataset is distinguishable from a raw one:

```
processed/reactive_correction.json      # e.g. {"mode": "qd_all"}
```

If you run with `--reactive-correction none`, no marker is written and the output is
identical to the standard build.

## Verifying the result

After building, you can confirm the reactive residual is ~0. Load a scenario and recompute
the balance with the same layers the training loss uses:

```python
import torch
from torch_geometric.data import HeteroData
from gridfm_graphkit.datasets.globals import VM_H, VA_H, QG_H, VM_OUT, VA_OUT, QG_OUT
from gridfm_graphkit.models.utils import (
    ComputeBranchFlow, ComputeNodeInjection, ComputeNodeResiduals,
)

d = HeteroData.from_dict(
    torch.load("/path/to/corrected_root/processed/data_index_0.pt", weights_only=True)
)
bus = d["bus"].x                      # physical units; angle in degrees on disk
ei  = d["bus", "connects", "bus"].edge_index
ea  = d["bus", "connects", "bus"].edge_attr
n   = bus.size(0)

out = torch.zeros((n, 4), dtype=bus.dtype)
out[:, VM_OUT] = bus[:, VM_H]
out[:, VA_OUT] = bus[:, VA_H] * torch.pi / 180.0   # deg -> rad for the physics
out[:, QG_OUT] = bus[:, QG_H]

Pft, Qft = ComputeBranchFlow()(out, ei, ea)
_, Q_in  = ComputeNodeInjection()(Pft, Qft, ei, n)
_, residual_Q = ComputeNodeResiduals()(torch.zeros(n), Q_in, out, bus)
print("max |residual_Q|:", residual_Q.abs().max().item())   # ~1e-5 after correction
```

Automated checks live in `tests/test_reactive_correction.py`:

```bash
python -m pytest tests/test_reactive_correction.py -q
```

## Bug fix: baseMVA unit mismatch

`ComputeBranchFlow`/`compute_shunt_power` return `Q_in`/`q_shunt` in **per-unit**, but
`Qd`/`Qg` are in **Mvar** at creation time. The original residual mixed the two, inflating
it by `baseMVA` (≈159 Mvar of phantom imbalance on case14). Fixed by scaling the per-unit
terms: `Q_in, q_shunt = Q_in * base_mva, q_shunt * base_mva` (with `base_mva` from
`data.baseMVA`, default 100). Post-fix the residual on converged data is ~0.

Max reactive imbalance over all 19,957 raw case14 scenarios:

| | Max imbalance |
|---|---|
| Without the fix (unscaled) | 159.07 Mvar |
| With the fix (`base_mva=100`) | 0.001 Mvar |

## Implementation pointers

- Core correction: `reconcile_reactive_balance(...)` in
  `gridfm_graphkit/datasets/hetero_preprocess.py`.
- Applied per scenario inside `build_hetero_data_for_scenario(...)`; threaded through
  `process_scenarios(...)` / `process_partition(...)` via a `reactive_correction` keyword
  (default `None`).
- CLI flag + marker file: `scripts/process_hetero_dataset_parallel.py`.
