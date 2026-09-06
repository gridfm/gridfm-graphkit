Place the HH-MPNN 90/5/5 split tensors here (`train.pt`, `val.pt`, `test.pt`, `shuffled_indices.pt`).

They are not stored in git. Download from the model repo:

```bash
hf download gridfm/genco-opfdata-base --include "splits/*" --local-dir /tmp/genco-opfdata-base
cp /tmp/genco-opfdata-base/splits/*.pt scripts/opfdata/splits/
```

Configs set `split_from_existing_files: scripts/opfdata/splits/` (run from the graphkit repo root).
