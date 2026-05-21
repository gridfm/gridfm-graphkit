# python run_on_new_dataset.py \
#     <CONFIG_PATH> \
#     <DATA_PATH> \
#     <EXPERIMENT_DIR> \
#     <NEW_EXP_NAME>

# Arguments:
#     CONFIG_PATH     Path to evaluation YAML config file
#     DATA_PATH       Path to dataset directory
#     EXPERIMENT_DIR  Path to MLflow experiment directory (contains run folders)
#     NEW_EXP_NAME    Name of the new MLflow experiment for evaluation

#!/usr/bin/env python3

"""
python /u/apu/gridfm_model_evaluation/scripts/pretraining_advantage/run_on_new_dataset.py \
/u/apu/gridfm_model_evaluation/scripts/pretraining_advantage/HGNS_PF_datakit_case30_eval_new_dataset.yaml \
/dccstor/gridfm/powermodels_data/v4/evaluation_pretraining/pf \
/dccstor/gridfm/mlflow_alban_pretraining_scaling/431383760655839309 \
case30_eval_new_dataset_to_del2

python /u/apu/gridfm_model_evaluation/scripts/pretraining_advantage/run_on_new_dataset.py \
/u/apu/gridfm_model_evaluation/scripts/pretraining_advantage/HGNS_PF_datakit_case500_eval_new_dataset.yaml \
/dccstor/gridfm/powermodels_data/v4/evaluation_pretraining/pf \
/dccstor/gridfm/mlflow_alban_pretraining_scaling/510717248406872224 \
case500_eval_new_dataset


python /u/apu/gridfm_model_evaluation/scripts/pretraining_advantage/run_on_new_dataset.py \
/u/apu/gridfm_model_evaluation/scripts/pretraining_advantage/HGNS_PF_datakit_case118_eval_new_dataset.yaml \
/dccstor/gridfm/powermodels_data/v4/evaluation_pretraining/pf \
/dccstor/gridfm/mlflow_alban_pretraining_scaling/652860222610121090 \
case118_eval_new_dataset_paper # need to relaunch this

# evaluation on k=5
python /u/apu/gridfm_model_evaluation/scripts/pretraining_advantage/run_on_new_dataset.py \
/u/apu/gridfm_model_evaluation/scripts/pretraining_advantage/HGNS_PF_datakit_case118_eval_new_dataset.yaml \
/dccstor/gridfm/powermodels_data/v4/evaluation_pretraining_k_5/pf \
/dccstor/gridfm/mlflow_alban_pretraining_scaling/652860222610121090 \
case118_eval_new_dataset_k_5

# evaluation on k=10
python /u/apu/gridfm_model_evaluation/scripts/pretraining_advantage/run_on_new_dataset.py \
/u/apu/gridfm_model_evaluation/scripts/pretraining_advantage/HGNS_PF_datakit_case118_eval_new_dataset.yaml \
/dccstor/gridfm/powermodels_data/v4/evaluation_pretraining_k_10/pf \
/dccstor/gridfm/mlflow_alban_pretraining_scaling/652860222610121090 \
case118_eval_new_dataset_k_10
"""



import sys
import subprocess
from pathlib import Path
import yaml

# ---------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------

BASE_DIR = Path("/u/apu/gridfm_model_evaluation")
LOG_DIR = "/dccstor/gridfm/mlflow_alban_pretraining_scaling"



# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def submit(cmd: str):
    """Submit a shell command and print it."""
    print("\nSubmitting job:\n", cmd, "\n")
    subprocess.run(cmd, shell=True, check=True)


def load_run_name(meta_path: Path) -> str:
    with open(meta_path, "r") as f:
        meta = yaml.safe_load(f)
    return meta["run_name"]


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    if len(sys.argv) != 5:
        print(
            "Usage:\n"
            "python launch_evaluations_from_mlflow.py "
            "<CONFIG_PATH> <DATA_PATH> <EXPERIMENT_DIR> <NEW_EXP_NAME>"
        )
        sys.exit(1)

    config_path = Path(sys.argv[1])
    data_path = sys.argv[2]
    experiment_dir = Path(sys.argv[3])
    exp_name = sys.argv[4]

    if not config_path.exists():
        raise FileNotFoundError(config_path)

    if not experiment_dir.exists():
        raise FileNotFoundError(experiment_dir)

    # -----------------------------------------------------------------
    # List all runs (subfolders with meta.yaml)
    # -----------------------------------------------------------------

    run_dirs = [
        d for d in experiment_dir.iterdir()
        if d.is_dir() and (d / "meta.yaml").exists()
    ]

    if not run_dirs:
        print("No runs found in experiment.")
        sys.exit(0)

    print(f"\nFound {len(run_dirs)} runs.\n")

    # -----------------------------------------------------------------
    # Iterate over runs
    # -----------------------------------------------------------------

    for run_dir in run_dirs[1:]:

        run_id = run_dir.name
        meta_path = run_dir / "meta.yaml"
        run_name = load_run_name(meta_path)

        model_path = (
            run_dir
            / "artifacts"
            / "model"
            / "best_model_state_dict.pt"
        )

        normalizer_path = (
            run_dir
            / "artifacts"
            / "stats"
            / "normalizer_stats.pt"
        )

        if not model_path.exists():
            print(f"⚠️  Skipping {run_id} (no model found)")
            continue

        if not normalizer_path.exists():
            print(f"⚠️  Skipping {run_id} (no normalizer found)")
            continue

        print(f"Preparing evaluation for run: {run_name}")

        # ------------------------------------------------------------
        # Launch evaluation job
        # ------------------------------------------------------------

        job_name = f"eval_{run_name}"

        eval_cmd = f"""
bsub -q normal \
-gpu "num=1:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB" \
-M 32G -n 16 -R "hname != cccxc584 && hname != cccxc563" \
-J {job_name} \
-o ~/.lsbatch/%J.out \
"cd {BASE_DIR}; \
source venv/bin/activate && \
gridfm_graphkit evaluate \
--config {config_path} \
--data_path {data_path} \
--model_path {model_path} \
--normalizer_stats {normalizer_path} \
--exp_name {exp_name} \
--run_name {run_name}_eval \
--log_dir {LOG_DIR} \
--compute_dc_ac_metrics"
"""
        print(eval_cmd)


if __name__ == "__main__":
    main()