**Winning case2k PF recipe — branch `paper`**

Scope: **case2000_goc @ 20k**. Config: [`HGNS_PF_datakit_case2000_canonical_best.yaml`](HGNS_PF_datakit_case2000_canonical_best.yaml).  
Best observed: **0.91 MW** test active (val l11 **0.0086**, seed0).

---

### Branch (required)

Use **`paper`**, not `main`:
- **Checkpoint** on `Validation layer_11_residual` (last-layer power balance — closer to the metric we care about).
- **LR schedule** on `Validation loss` — **not** layer_11 (too noisy → decay ~**3×** too early: first drop ~ep12 vs ~ep34; LR fell to **~1.7e-4 by ep24** instead of staying at **5e-4 until ~ep58**).
- Restore **best checkpoint before test** (old path tested last-epoch weights).

---

### Config knobs (why)

| Knob | Value | Evidence (seed0 unless noted) |
|------|--------|-------------------------------|
| Physics / MSE | **20% / 80%** | **1.03 MW** vs 10% **~1.27–1.44** (~**20%** better) and 5% **2.03**. 30% catastrophic (**509 / 76 MW**). **Unstable across seeds** (below). |
| `base_weight` | **0.5** | bw0.3 hurts 10% baseline (**1.64 vs 1.27**); on 20% phys **1.06 vs 1.03**. Winners used 0.5. |
| `lr_patience` | **10** | On 20% phys: **0.98 vs 1.03** (lrp5) → **−0.05 (~5%)**. No gain on 10% baseline. Does **not** fix seed divergence. |
| `lr_decay` | **0.85** | On 20% phys + lrp10: **0.91 vs 0.98** (decay 0.7) → **−0.07 (~7%)**. Neutral on 10% baseline. |
| LR | **5e-4** | Fine mid-run when LR monitor is correct; don’t raise first. |
| Arch | **tiny h12** | At 10% phys, size doesn’t help (tiny **1.44** / small **1.48** / base **1.51**). At 20% phys, small ≈ tiny (**0.92 vs 0.91**). Skip **h48**. |
| Batch | **global 16** (`bs=8` × 2 GPU) | case2k tiny: bs16 **1.36** vs bs4 **2.27** MW. |
| Epochs | **300** | 300 vs 200 on 10% baseline: **1.27 vs 1.37** only; jump to ~1 MW was **20% phys**. Diminishing returns ~ep250 (**0.92** at early-stop ≈ full run). |
| Data | **20k**, split by load scenario, norm **121 / 345** | Winning stack; preset avoids heavy fit on large data. |
| Seed | **0** (then **42**) | Same 20% recipe: seed0 **1.03** vs seed1 **505**; seed1 failed **4×**. Never average diverged seeds (**~253 MW** fake mean). |

**Stacked (20% phys @ lrp5/d0.7 → this file):** 1.03 → 0.98 (lrp10) → **0.91** (−**0.12 / ~12%** from LR; physics still dominant).

**Track:** test **Avg. active res. (MW)** on best ckpt. Mid-run val l11: **kill if >1 by ~ep8–20**; healthy **<0.05 by ~ep20**.

---

### Launch

```bash
cd <repo> && source venv/bin/activate && unset MLFLOW_TRACKING_URI

# 2× GPU unpinned, 64G (OOM → 128G), ~7h @ case2k tiny bs16
gridfm_graphkit train \
  --config experiments/winning_recipe_case2k/HGNS_PF_datakit_case2000_canonical_best.yaml \
  --data_path <pf_datakit_root> \
  --exp_name <exp> --run_name seed0 \
  --log_dir <mlflow_store> \
  --compile reduce-overhead
```

`--compile reduce-overhead` on train. ALWAYS USE IT !!! IT SAVES A LOT OF TIME

---

### case10k — do not copy this file

| case2k (this recipe) | case10k |
|----------------------|---------|
| global **bs16** | global **bs4** (small bs4 **~0.82** vs bs16 **~2.8** MW) |
| **20%** phys | **10%** phys — 20% wins residuals on case2k but seed-diverges often |
| tiny ≈ small | **small** often ≥ tiny |
| ~64G | **128–256G** |


---

### Normalizer presets

- Set `normalizer_preset` (`baseMVA` / `vn_kv_max`) to **skip fit-from-train** (saves RAM/time on large pools because no need to then load the entire parquet dataset in memory)
- In the future we should fit the normalizer on the processed .pt files instead of using the whole parquet files and having to load everything (1TB+ for case10k in memory)
- Fit-from-split vs preset can differ slightly (**~1e−1–1e−2** relative), but that's usually fine
- remember to always adapt the values to your network -> you can fit once for the first run, then read the values from the mlflow artifacts, and use them in the config file of your next runs

---

### Stability (20% physics)

- We could consider different weight init distrutions 
- We should start training with 10 epochs with only MSE to have stable training, and then add physics loss later -> needs to be implemented
---

### Dataset size vs steps (hypothesis)

Winning case2k used N=**20k** scenarios, not 250k. Intuition: smaller N can be offset by **more epochs** and **smaller batch** (more steps/epoch). Data is fairly homogeneous, so more unique scenarios may help less than raw step count — WHAT DO YOU THINK @GUANG/PANOS?

---

### Tiny vs larger architecture

For large grids I think we havent mastered yet the HPO for base and small architecture, so that so far tiny works best. It allows fast training in less than 7 hours for case2k so that's nice. in the future we should try base again and train for longer.


