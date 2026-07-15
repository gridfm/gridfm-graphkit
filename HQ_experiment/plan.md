# HQ experiment plan — scratch vs finetune @ 100 / 1k / 10k / 20k → eval on TEST_DS

Branch: **`paper`**. Configs live in this folder. `networks: [hq1200]` needs to be updated in every yaml to the name you use for the directory of your data.

---

## Data layout

Agreed with data partner:

1. **1 year of snapshots @ 30 min** — OK (~17 520 snapshots).
2. Hold out **10% → `TEST_DS`** (move elsewhere; not under the train path) → ~**1.7k** samples.
3. Remaining **90% → `TRAIN_POOL`** (~**15.8k**). Train/finetune sample N ∈ {100, 1k, 10k, 20k} from `TRAIN_POOL` with `val_ratio: 0.1`, `test_ratio: 0.01` (the 1% train-time test split is **not** used for reporting).
4. Report metrics only from **`evaluate` on `TEST_DS`**. (I set 250k scenarios in the config file but graphkit will just use all available scenarios from `TEST_DS`)
5. **Reindex scenarios** in both `TRAIN_POOL` and `TEST_DS` so IDs **start at 0 and are contiguous** (0…N−1). Gaps or non-zero starts break loading / `scenarios:` sampling.

### Precisions about  N = 20k

With ~15.8k in `TRAIN_POOL`, `scenarios: 20000` silently uses the **full pool**. 
---

## Architecture caveat

Train configs use **tiny (`hidden_size: 12`)** because that works best with small N.

If the pretrained checkpoint is **base (h48)**:

- either **re-pretrain a tiny** model, **or**
- set **`hidden_size: 48` in every** train / finetune / eval config here.

---

## Matrix

| Mode | N | Config |
|------|---|--------|
| scratch | 100 / 1k / 10k / 20k† | `HGNS_PF_datakit_hq1200_jul15_{100,1k,10k,20k}_2_gpus.yaml` |
| finetune | same N, same seed, same yaml | same yaml — CLI is `finetune` (not `train`) plus `--model_path` |
| eval | each of the 8 runs | `HGNS_PF_datakit_hq1200_jul15_eval_run_on_1_gpu.yaml` |

† See “N = 20k vs TRAIN_POOL size” above.

**Compile:** `--compile reduce-overhead` on train + finetune. **No compile** on evaluate.

**GPUs:** train/FT configs assume **2 GPUs** (`batch_size: 8` → global 16). Eval is **1 GPU** (`batch_size: 64`).

---

## Commands

```bash
REPO=/path/to/gridfm_model_evaluation   # clone of this repo (paper branch)
HQ=$REPO/HQ_experiment
source $REPO/venv/bin/activate
unset MLFLOW_TRACKING_URI               # if set, can override --log_dir

TRAIN_POOL=/path/to/TRAIN_POOL          # 90% snapshots; 
TEST_DS=/path/to/TEST_DS                # held-out 10%; 
MLFLOW=/path/to/mlflow_store            # file: MLflow root for --log_dir
PRETRAINED=/path/to/best_model_state_dict.pt

# pick N: 100 | 1k | 10k | 20k
CFG_N=$HQ/HGNS_PF_datakit_hq1200_jul15_1k_2_gpus.yaml
CFG_EVAL=$HQ/HGNS_PF_datakit_hq1200_jul15_eval_run_on_1_gpu.yaml
```

### 1) Train from scratch (2 GPU + compile)

```bash
gridfm_graphkit train \
  --config $CFG_N \
  --data_path $TRAIN_POOL \
  --exp_name jul15_scratch_<N> \
  --run_name seed0 \
  --log_dir $MLFLOW \
  --compile reduce-overhead
```

### 2) Finetune (same config / seed / data; CLI + model path differ)

```bash
gridfm_graphkit finetune \
  --config $CFG_N \
  --model_path $PRETRAINED \
  --data_path $TRAIN_POOL \
  --exp_name jul15_ft_<N> \
  --run_name seed0 \
  --log_dir $MLFLOW \
  --compile reduce-overhead
```

Note: `finetune` has **no** `--normalizer_stats` — it **refits** the normalizer on the FT train split.

### 3) Evaluate on TEST_DS (1 GPU, no compile)

```bash
gridfm_graphkit evaluate \
  --config $CFG_EVAL \
  --model_path /path/to/run/artifacts/model/best_model_state_dict.pt \
  --normalizer_stats /path/to/run/artifacts/stats/normalizer_stats.pt \
  --data_path $TEST_DS \
  --exp_name jul15_eval_<scratch|ft>_<N> \
  --run_name seed0_on_TEST_DS \
  --log_dir $MLFLOW
```

Use **`--normalizer_stats` from the same run** as `--model_path` (scratch→scratch stats, FT→FT stats). Do not mix pretrain stats with an FT checkpoint.

Repeat for N ∈ {100, 1000, 10000, 20000}.

---

## Other considerations

- **Branch `paper`:** checkpoint = `Validation layer_11_residual`, LR = `Validation loss`, restore best before test.
- **Same seed** in yaml for scratch vs FT pairs; prefer seed0 (+42 if lottery).
- **Normalizer:** scratch and FT both **fit on their train split** (`finetune` cannot load pretrain stats today). Eval must use that run’s **`normalizer_stats.pt`**.
- **Don’t** report in-training test residual; only **TEST_DS evaluate**.
- **Scenario IDs** in `TRAIN_POOL` and `TEST_DS` must be **reindexed to 0…N−1 contiguous** after the split.
- **`split_by_load_scenario_idx: false`** on all configs (HQ snapshots; no `load_scenario_idx` required).
- **Early kill:** val l11 **≫ 1** for many early epochs → retry seed. Healthy runs usually drop quickly.

---



## Files in this folder

| File | Role |
|------|------|
| `HGNS_PF_datakit_hq1200_jul15_100_2_gpus.yaml` | train/FT · 100 scenarios |
| `HGNS_PF_datakit_hq1200_jul15_1k_2_gpus.yaml` | train/FT · 1k |
| `HGNS_PF_datakit_hq1200_jul15_10k_2_gpus.yaml` | train/FT · 10k |
| `HGNS_PF_datakit_hq1200_jul15_20k_2_gpus.yaml` | train/FT · “20k” (= full TRAIN_POOL if pool &lt; 20k) |
| `HGNS_PF_datakit_hq1200_jul15_eval_run_on_1_gpu.yaml` | evaluate on TEST_DS |
| `plan.md` | this plan |
