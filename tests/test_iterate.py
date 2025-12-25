import itertools
from pathlib import Path

import yaml
from gridfm_graphkit.__main__ import main
import pytest
import sys

CONFIG_FILES = [
    "configs/iterate_test_case30_ieee_base.yaml",
]

MODELS = ["examples/models/GridFM_v0_2.pth"]
INPUT_TEST_MAIN = list(itertools.product(MODELS, CONFIG_FILES))


def get_test_ids() -> list[str]:
    test_case_ids = list()
    for model, config in INPUT_TEST_MAIN:
        # get the filename
        model = model.split("/")[-1].replace(".pth", "")
        config = config.split("/")[-1].replace(".yaml", "")
        # append to list of test ids
        test_case_ids.append(f"{config}_{model}")
    return test_case_ids


def validate_hpo_results(
    experiment_name: str,
    results_folder: Path,
    n_trials: int,
    n_tasks: int,
    iterate_info: dict,
    ):
    # check that experiment was created
    mlflow_output_path =  / "hpo_mlflow_output"
    assert mlflow_output_path.exists(), f"Error! Directory does not exist: {mlflow_output_path}"
    hpo_exp_path = mlflow_output_path / iterate_info["hpo_experiment_id"]
    assert hpo_exp_path.exists(), f"Error! Directory does not exist: {hpo_exp_path}"
    meta_yaml_path = hpo_exp_path / "meta.yaml"
    assert meta_yaml_path.exists(), (
        f"Error! meta.yaml file {meta_yaml_path} does not exist"
    )

    # open file and check that the experiment name/id is the same
    experiment_name_found: bool = False
    finished_run_id_found: bool = False
    experiment_id_found: bool = False
    experiment_id = iterate_info["hpo_experiment_id"]
    finished_run_id = iterate_info["hpo_finished_run_id"]
    with open(meta_yaml_path, mode="r") as f:
        # read all the lines
        lines = f.readlines()
        # try to find experiment id and name in these lines
    for line in lines:
        if experiment_name in line:
            experiment_name_found = True
        if finished_run_id in line:
            finished_run_id_found = True
        if experiment_id in line:
            experiment_id_found = True
    assert experiment_name_found and experiment_id_found and finished_run_id_found, (
        f"Error! Both experiment name ({experiment_name=}), finished run id ({finished_run_id=}), \
        and experiment id ({experiment_id}) must be in the {meta_yaml_path=}."
    )

    # check number of runs created
    expected_num_runs = (n_trials*n_tasks) + n_tasks + 1
    run_folders = [ f.path for f in os.scandir(folder) if f.is_dir() ]
    assert len(run_folders)==expected_num_runs, (
        f"Error! Expected {expected_num_runs} to be created for HPO experiment. Found {len(run_folders)} runs."
    )




def validate_repeated_results(
    experiment_name: str,
    results_folder: Path,
    n_trials: int,
    n_tasks: int,
    num_repetitions: int,
    iterate_info: dict,
    ):
    # check that epxeriment was created
    mlflow_output_path =  / "repeated_mlflow_output"
    assert mlflow_output_path.exists(), f"Error! Directory does not exist: {mlflow_output_path}"
    hpo_exp_path = mlflow_output_path / iterate_info["hpo_experiment_id"]
    assert hpo_exp_path.exists(), f"Error! Directory does not exist: {hpo_exp_path}"
    meta_yaml_path = hpo_exp_path / "meta.yaml"
    assert meta_yaml_path.exists(), (
        f"Error! meta.yaml file {meta_yaml_path} does not exist"
    )

    # open file and check that the experiment name is the same
    experiment_name_found: bool = False
    experiment_id_found: bool = False
    experiment_id = iterate_info["repeated_experiment_id"]
    with open(meta_yaml_path, mode="r") as f:
        # read all the lines
        lines = f.readlines()
        # try to find experiment id and name in these lines
        
        for line in lines:
            if experiment_name in line:
                experiment_name_found = True
            if experiment_id in line:
                experiment_id_found = True
        assert experiment_name_found and experiment_id_found, (
            f"Error! Both experiment name ({experiment_name=}) and experiment id ({experiment_id=}) \
            must be in the {meta_yaml_path=}."
        )

    # check number of runs created
    expected_num_runs = (n_trials*n_tasks) + 1
    run_folders = [ f.path for f in os.scandir(folder) if f.is_dir() ]
    assert len(run_folders)==expected_num_runs, (
        f"Error! Expected {expected_num_runs} to be created for repeated experiment. Found {len(run_folders)} runs."
    )
    




@pytest.mark.parametrize(
    "model, config",
    INPUT_TEST_MAIN,
    ids=get_test_ids(),
)
def test_iterate(
    model: str,
    config: str,
):
    test_dir = Path(__file__).parents[0]
    home_dir = test_dir.parents[0]
    config_file: Path = test_dir / config
    assert config_file.exists()
    with open(config_file, "r") as file:
        config_data = yaml.safe_load(file)
    experiment_name = config_data["hpo_spec"]["experiment_name"]
    results_folder = config_data["hpo_spec"]["results_folder"]
    results_folder = Path(results_folder)
    num_repetitions = config_data["hpo_spec"]["num_repetitions"]
    n_trials = config_data["hpo_spec"]["n_trials"]
    n_tasks = len(config_data["tasks"])
    

    #send command to sys
    arguments = ["gridfm_graphkit", "iterate", "--config", str(config_file.resolve())]
    model_path = home_dir / model
    results_folder = home_dir / "test_reults"
    arguments.append["--model.model_path", f"{str(model_path)}"]
    arguments.append["--hpo_spec.results_folder", f"{str(results_folder)}"]

    sys.argv = arguments
    iterate_info = main()
    assert isinstance(iterate_info, dict), f"Error! {iterate_info=} is not a dict"
    validate_hpo_results(
        experiment_name=experiment_name,
        results_folder=results_folder,
        n_trials=n_trials,
        n_tasks=n_tasks,
        iterate_info=iterate_info,
    )

    validate_repeated_results(
        experiment_name=experiment_name,
        results_folder=results_folder,
        num_repetitions=num_repetitions,
        n_trials=n_trials,
        n_tasks=n_tasks,
        iterate_info=iterate_info,
    )