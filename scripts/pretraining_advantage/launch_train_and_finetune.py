#!/usr/bin/env python3
"""
python /u/apu/gridfm_model_evaluation/scripts/pretraining_advantage/launch_train_and_finetune.py \
/u/apu/gridfm_model_evaluation/examples/config/HGNS_PF_datakit_case118.yaml \
case118
"""

import sys
import subprocess
from pathlib import Path
import yaml

# Scenarios to run
SCENARIOS = [100, 1000, 10000, 20000, 50000, 100000]
# SCENARIOS = [250000]


# Base directories
BASE_DIR = Path("/u/apu/gridfm_model_evaluation")
DATA_PATH = "/dccstor/gridfm/powermodels_data/v4/finetuning/pf"

TRAIN_LOG_DIR = "/dccstor/gridfm/mlflow_alban_pretraining_scaling"

# Pretrained model to finetune from
PRETRAINED_MODEL = (
    "/dccstor/gridfm/mlflow_alban_pretraining/"
    "335178916922704189/"
    "11d08e25cdb64a0d8c62d6f6179355a6/"
    "artifacts/model/best_model_state_dict.pt"
)

# Directory to store persistent configs
TMP_DIR = BASE_DIR / "tmp_configs"
TMP_DIR.mkdir(exist_ok=True)


def submit(cmd: str):
    """Submit a shell command and print it."""
    print("\nSubmitting job:\n", cmd, "\n")
    subprocess.run(cmd, shell=True, check=True)


def main():
    if len(sys.argv) != 3:
        print("Usage: python launch_train_and_finetune.py <CONFIG_PATH> <CASE>")
        sys.exit(1)

    config_path = Path(sys.argv[1])
    case = sys.argv[2]

    if not config_path.exists():
        raise FileNotFoundError(config_path)

    for scen in SCENARIOS:

        print(f"\n=== Scenario {scen} ===")

        # ------------------------------------------------------------
        # Create persistent config with updated scenario
        # ------------------------------------------------------------
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        config["data"]["scenarios"] = [scen]

        tmp_config = TMP_DIR / f"{case}_{scen}_config.yaml"
        with open(tmp_config, "w") as f:
            yaml.dump(config, f)

        exp_name = f"scaling_{case}_a100_3_to_del4"

        # ------------------------------------------------------------
        # 1️⃣ TRAIN FROM SCRATCH
        # ------------------------------------------------------------
        train_job = f"train_from_scratch_{case}_{scen}"
        train_run = f"{case}_{scen}"

        train_cmd = f"""
bsub -q normal \
-gpu "num=1:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB" \
-M 32G -n 16 \
-J {train_job} \
-o ~/.lsbatch/%J.out \
"cd {BASE_DIR}; \
source venv/bin/activate && \
gridfm_graphkit train \
--config {tmp_config} \
--data_path {DATA_PATH} \
--exp_name {exp_name} \
--run_name {train_run} \
--log_dir {TRAIN_LOG_DIR}"
"""
        submit(train_cmd)

        # ------------------------------------------------------------
        # 2️⃣ FINETUNE (independent, 1 GPU)
        # ------------------------------------------------------------
        finetune_job = f"finetune_{case}_{scen}"
        finetune_run = f"finetune_{case}_{scen}"

        finetune_cmd = f"""
bsub -q normal \
-gpu "num=1:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB" \
-M 32G -n 16 \
-J {finetune_job} \
-o ~/.lsbatch/%J.out \
"cd {BASE_DIR}; \
source venv/bin/activate && \
gridfm_graphkit finetune \
--config {tmp_config} \
--data_path {DATA_PATH} \
--exp_name {exp_name} \
--run_name {finetune_run} \
--model_path {PRETRAINED_MODEL} \
--log_dir {TRAIN_LOG_DIR}"
"""
        submit(finetune_cmd)


if __name__ == "__main__":
    main()