import os
from pathlib import Path
import logging
from jsonargparse import Namespace





from functools import partial
from typing import Any, Dict

import mlflow
import optuna
import pandas as pd
import torch
from optuna.pruners import HyperbandPruner
from optuna.samplers import BaseSampler, RandomSampler

from gridfm_graphkit.iterate.model_fitting import fit_model, fit_model_with_hparams

from gridfm_graphkit.utils.types import (
    HyperParameterOptmizerSpec, 
    TaskSpec, 
    CallbackSpec,
    OptimizerSpec, 
    ModelSpec, 
    TrainingSpec, 
    DataSpec,
    direction_type_to_optuna,
    optimization_space_type

    )

from gridfm_graphkit.iterate.utils import (
    parse_optimization_space,
    check_existing_task_parent_runs,
    check_existing_experiments,
    unflatten,
    get_logger,
    sync_mlflow_optuna,
    )



def run_iterate_experiments(
    args, #TODO: remove
    model_spec: ModelSpec,
    training_spec: TrainingSpec,
    optimizer_spec: OptimizerSpec,
    callbacks_spec: CallbackSpec,
    hpo_spec: HyperParameterOptmizerSpec,
    tasks: list[TaskSpec],
    seed: int = 42,
    ) -> bool:
    """
    runs full benchmarking (hpo + repeated) for a model across multiple tasks

    Args:


    Return:
    
    """
    #create folders and initialize logger
    base = Path(args.hpo_spec.results_folder)
    HPO_EXP_FOLDER = base / "hpo_mlflow_output"
    REPEATED_EXP_FOLDER = base / "repeated_mlflow_output"
    REPEATED_CSV_FOLDER = base / "repeated_csv_output"
    LOG_FOLDER = base / "logs"
    folders = [HPO_EXP_FOLDER, REPEATED_EXP_FOLDER, REPEATED_CSV_FOLDER, LOG_FOLDER]
    for f in folders:
        os.makedirs(str(f), exist_ok=True)
    logger = get_logger(log_folder=str(LOG_FOLDER))

    benchmarking_completed = False
    try:
        # run hpo on model across multiple tasks 
        hpo_output = run_hpo_experiments(
            args=args, #TODO: remove args
            logger=logger,
            model_spec=model_spec,
            training_spec=training_spec,
            optimizer_spec=optimizer_spec,
            callbacks_spec=callbacks_spec,
            hpo_spec=hpo_spec,
            tasks=tasks,
            seed=seed,
            storage_uri=HPO_EXP_FOLDER,
            )

        if args.hpo_spec.num_repetitions >= 1:
            # run repeated experiments
            run_repeated_experiments(
                args=args, #TODO: remove args
                logger=logger,
                model_spec=model_spec,
                training_spec=training_spec,
                optimizer_spec=optimizer_spec,
                callbacks_spec=callbacks_spec,
                hpo_spec=hpo_spec,
                tasks=tasks,
                seed=seed,
                repeated_storage_uri=REPEATED_EXP_FOLDER,
                hpo_storage_uri=HPO_EXP_FOLDER,
                csv_folder=REPEATED_CSV_FOLDER,
                experiment_id=hpo_output["experiment_id"],
                parent_run_id=hpo_output["finished_run_id"],
            )   
    except Exception as e:
        logger.info(f"Could not complete due to error {e}")
        raise



def run_hpo_experiments(
    args: Namespace, #TODO: remove
    logger: logging.RootLogger,
    model_spec: ModelSpec,
    training_spec: TrainingSpec,
    optimizer_spec: OptimizerSpec,
    callbacks_spec: CallbackSpec,
    hpo_spec: HyperParameterOptmizerSpec,
    tasks: list[TaskSpec],
    seed: int,
    storage_uri: Path,
) -> Dict[str, str]:
    """Highest level function to run hpo only for a model across multiple tasks

    Args:
        args: Namespace, #TODO: remove
        logger: logging.RootLogger,
        model_spec: ModelSpec,
        training_spec: TrainingSpec,
        optimizer_spec: OptimizerSpec,
        callbacks_spec: CallbackSpec,
        hpo_spec: HyperParameterOptmizerSpec,
        tasks: list[TaskSpec],
        seed: int,
        storage_uri: Path,


    Return:


    """
    # https://mlflow.org/docs/latest/ml/tracking/system-metrics/#using-the-environment-variable-to-control-system-metrics-logging
    if os.getenv("MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING") is None:
        os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"

    model_type: str = model_spec.type
    run_id: str = hpo_spec.run_id
    experiment_name: str = hpo_spec.experiment_name
    task_names = [task.name for task in tasks]
    run_name = f"top_run_{hpo_spec.experiment_name}" if hpo_spec.run_name is None else hpo_spec.run_name
    optimization_space = parse_optimization_space(hpo_spec.optimization_space)
    completed_task_run_names = []
    optimize_hyperparams = True
    task_run_to_id_match = {}

    storage_uri = str(storage_uri)
    logger.info(f"Setting tracking URI: {storage_uri}")
    mlflow.set_tracking_uri(storage_uri)
    logger.info(f"Setting experiment name: {experiment_name}")
    mlflow.set_experiment(experiment_name)
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    if hpo_spec.continue_existing_experiment:
        # find status of existing runs, and delete incomplete runs except one with the most complete tasks
        existing_experiments = check_existing_experiments(
            logger=logger,
            storage_uri=storage_uri,
            experiment_name=experiment_name,
            exp_parent_run_name=run_name,
            task_names=task_names,
            n_trials=hpo_spec.n_trials,
        )
        if existing_experiments["no_existing_runs"]:
            logger.info("\nStarting new experiment from scratch")
        else:
            if (existing_experiments["incomplete_run_to_finish"] is not None) and (
                run_id is None
            ):
                logger.info("Continuing previous experiment parent run")
                run_id = existing_experiments["incomplete_run_to_finish"]
                logger.debug(f"incomplete_run_to_finish: {run_id=}")
                experiment_id = existing_experiments["experiment_id"]
                optimize_hyperparams = True

            if existing_experiments["finished_run"] is not None:
                optimize_hyperparams = False
                finished_run_id = existing_experiments["finished_run"]
                logger.debug(f"finished_run: {run_id=}")
                run_id = existing_experiments["finished_run"]

            # get previously completed tasks
            completed_task_run_names, _, task_run_to_id_match = (
                check_existing_task_parent_runs(
                    logger, run_id, storage_uri, experiment_name, hpo_spec.n_trials
                )
            )
    else:
        logger.info("Starting new experiment from scratch")

    # only run hyperparameter optimization (HPO) if there are no experiments with finished HPO
    if optimize_hyperparams:
        if hpo_spec.bayesian_search:
            sampler = None # defaults to TPESampler
        else:
            sampler = RandomSampler()
        experiment_id, finished_run_id = _run_hpo(
            args=args, #TODO
            model_spec=model_spec,
            training_spec=training_spec,
            optimizer_spec=optimizer_spec,
            callbacks_spec=callbacks_spec,    
            run_name=run_name,
            run_id=run_id,
            tasks=tasks,
            task_names=task_names,
            completed_task_run_names=completed_task_run_names,
            task_run_to_id_match=task_run_to_id_match,
            storage_uri=storage_uri,
            experiment_name=experiment_name,
            n_trials=hpo_spec.n_trials,
            save_models=hpo_spec.save_models,
            sampler=sampler,
            test_models=hpo_spec.test_models,
            optimization_space=optimization_space,
            logger=logger,
            seed=seed,
        )
        logger.info("HPO complete")
    return {"experiment_id": experiment_id, "finished_run_id": finished_run_id}




def _run_hpo(
    args: Namespace,
    model_spec: ModelSpec,
    training_spec: TrainingSpec,
    optimizer_spec: OptimizerSpec,
    callbacks_spec: CallbackSpec,
    tasks: list,
    task_names: list[str],
    completed_task_run_names: list[str],
    task_run_to_id_match: dict,
    storage_uri: str,
    experiment_name: str,
    optimization_space: optimization_space_type,
    n_trials: int,
    logger: logging.RootLogger,
    sampler: BaseSampler | RandomSampler,
    description: str | None = None,
    run_name: str | None = None,
    run_id: str | None = None,
    save_models: bool = False,
    test_models: bool = False,
    seed: int = 42,
) -> tuple[str, str]:
    """
    run HPO for multiple tasks under a single experiment.

    Args:
        arg: Namespace, contains all parameters to be passed to model and datamodule. To be removed
        model_spec: ModelSpec, contains all parameters to intiali model
        training_spec: TrainingSpec,
        optimizer_spec: OptimizerSpec,
        callbacks_spec: CallbackSpec,
        tasks: list,
        task_names: list[str],
        completed_task_run_names: list[str],
        task_run_to_id_match: dict,
        storage_uri: str,
        experiment_name: str,
        optimization_space: optimization_space_type,
        n_trials: int,
        logger: logging.RootLogger,
        sampler: BaseSampler | RandomSampler,
        description: str | None = None,
        run_name: str | None = None,
        run_id: str | None = None,
        save_models: bool = False,
        test_models: bool = False,
        seed: int = 42,


    """
    logger.info(
        f"Running hyperparameter optimization: {run_name=} {run_id=}"
    )
    if run_id is not None:
        run_name = None

    with mlflow.start_run(
        run_name=run_name, run_id=run_id, description=description
    ) as run:
        for task in tasks:
            # only run task if it was not completed before
            task_run_name = task.name
            if task_run_name in completed_task_run_names:
                logger.info(f"{task_run_name} already completed")
                continue
            else:
                logger.info(f"{task_run_name} not completed. starting now")

            task_run_id = (
                task_run_to_id_match[task_run_name]
                if task_run_name in task_run_to_id_match
                else None
            )
            best_value, metric_name, hparams = _run_hpo_per_task(
                args=args, #TODO
                model_spec=model_spec,
                training_spec=training_spec,
                optimizer_spec=optimizer_spec,
                callbacks_spec=callbacks_spec,  
                logger=logger,
                task=task,
                storage_uri=str(storage_uri),
                experiment_name=experiment_name,
                experiment_run_id=run.info.run_id,
                task_run_id=task_run_id,
                optimization_space=optimization_space,
                n_trials=n_trials,
                save_models=save_models,
                sampler=sampler,
                test_models=test_models,
                seed=seed,
            )
        experiment_id = run.info.experiment_id

        # check completion of HPO for all tasks before proceeding to next stage
        existing_experiments = check_existing_experiments(
            logger=logger,
            storage_uri=storage_uri,
            experiment_name=experiment_name,
            exp_parent_run_name=run_name,
            task_names=task_names,
            n_trials=n_trials,
        )
        if existing_experiments["finished_run"] is not None:
            finished_run_id = existing_experiments["finished_run"]
        else:
            logger.info("HPO is not complete. Please re-run this experiment")
            raise RuntimeError

    return experiment_id, finished_run_id






def _run_hpo_per_task(
    args: Namespace, #TODO: remove args
    model_spec: ModelSpec,
    training_spec: TrainingSpec,
    optimizer_spec: OptimizerSpec,
    callbacks_spec: CallbackSpec,
    logger: logging.RootLogger,
    task: TaskSpec,
    storage_uri: str,
    experiment_name: str,
    experiment_run_id: str,
    task_run_id: str | None = None,
    optimization_space: optimization_space_type | None = None,
    n_trials: int = 1,
    save_models: bool = False,
    sampler: BaseSampler | None = None,
    test_models: bool = False,
    seed: int = 42,
):     
    """
    Performs HPO on a single task

    Args:
        args: Namespace, #TODO: remove args
        model_spec: ModelSpec,
        training_spec: TrainingSpec,
        optimizer_spec: OptimizerSpec,
        callbacks_spec: CallbackSpec,
        logger: logging.RootLogger,
        task: TaskSpec,
        storage_uri: str,
        experiment_name: str,
        experiment_run_id: str,
        task_run_id: str | None = None,
        optimization_space: optimization_space_type | None = None,
        n_trials: int = 1,
        save_models: bool = False,
        sampler: BaseSampler | None = None,
        test_models: bool = False,
        seed: int = 42,

    """
    logger.info(
        f"starting backbone benchmark on task {task.name} {task_run_id=} {experiment_name=}"
    )
    if storage_uri.startswith("http"):
        optuna_db_path = Path(".") / "optuna_db"
    else:
        optuna_db_path = Path(storage_uri).parents[0] / "optuna_db"

    if not os.path.exists(optuna_db_path):
        os.makedirs(optuna_db_path)
    optuna_db_path = optuna_db_path / f"{experiment_name}_{experiment_run_id}"
    optuna_db_path = str(optuna_db_path)

    task_run_id = sync_mlflow_optuna(
        optuna_db_path=optuna_db_path,
        storage_uri=storage_uri,
        experiment_name=experiment_name,
        task_run_id=task_run_id,
        task=task,
        n_trials=n_trials,
        logger=logger,
    )
    if task_run_id is not None:
        # run_name is used only when run_id is unspecified.
        run_name = None
    else:
        run_name = task.name
    logger.info(f"start run: {run_name=} {task_run_id=}")
    with mlflow.start_run(run_name=run_name, nested=True, run_id=task_run_id) as run:
        logger.info(f"starting task run with id: {run.info.run_id}")
        if training_spec.epochs is None:
            raise Exception("Must specify epochs for training")
        
        # if no optimization params, just run it
        if optimization_space is None:
            return (
                *fit_model(
                    args=args,
                    model_spec=model_spec,
                    training_spec=training_spec,
                    optimizer_spec=optimizer_spec,
                    callbacks_spec=callbacks_spec,
                    task=task,
                    run_name=f"{run_name}_no_optim",
                    experiment_name=experiment_name,
                    storage_uri=storage_uri,
                    parent_run_id=run.info.run_id,
                    save_models=save_models,
                    test_models=test_models,
                    seed=seed,
                    logger=logger,
                ),
            )

        # if optimization parameters specified, do hyperparameter tuning
        study = optuna.create_study(
            sampler=sampler,
            direction=direction_type_to_optuna[
                task.direction
            ],  # in the future may want to allow user to specify this
            pruner=HyperbandPruner(),
            study_name=task.name,
            storage="sqlite:///{}.db".format(optuna_db_path),
            load_if_exists=True,
        )

        objective = partial(
            fit_model_with_hparams,
            args,
            model_spec,
            training_spec,
            optimizer_spec,
            callbacks_spec,
            task,
            optimization_space,
            run_name,
            experiment_name,
            storage_uri,
            run.info.run_id,
            logger,
            save_models,
            test_models,
            seed,
        )

        n_trials = n_trials - len(study.trials)
        for trial in study.trials:
            if (trial.state == optuna.trial.TrialState.FAIL) | (
                trial.state == optuna.trial.TrialState.RUNNING
            ):
                n_trials = n_trials + 1

        study.optimize(
            objective,
            n_trials=n_trials,
            catch=[torch.cuda.OutOfMemoryError],
        )

        tags = {
            "seed": str(seed),
            "n_trials": str(n_trials),
            "model_spec": vars(model_spec),
            "training_spec": vars(training_spec),
            "optimizer_spec": vars(optimizer_spec),
            "callbacks_spec": vars(callbacks_spec),
            "data": vars(task.data),
        }
        mlflow.set_tags(tags)

        best_params = unflatten(study.best_trial.params)
        mlflow.log_params(best_params)  # unflatten
        mlflow.log_metric(f"best_{task.metric}", study.best_value)
        return study.best_value, task.metric, best_params



def run_repeated_experiments(
    args: Namespace, #TODO: remove args
    logger: logging.RootLogger,
    model_spec: ModelSpec,
    training_spec: TrainingSpec,
    optimizer_spec: OptimizerSpec,
    callbacks_spec: CallbackSpec,
    hpo_spec: HyperParameterOptmizerSpec,
    tasks: list[TaskSpec],
    seed: int,
    repeated_storage_uri: Path,
    hpo_storage_uri: Path,
    csv_folder: Path,
    experiment_id: str,
    parent_run_id: str,
):
    """Repeat best experiments from a benchmark run. Only works with a ray cluster.

    Args:


    """
    

    # if backbone_import:
    #     importlib.import_module(backbone_import)

    experiment_name = hpo_spec.experiment_name
    num_repetitions = hpo_spec.num_repetitions
    #find completed HPO tasks
    mlflow.set_tracking_uri(str(hpo_storage_uri))
    mlflow.set_experiment(experiment_name)

    runs: list[mlflow.entities.Run] = mlflow.search_runs(
        filter_string=f"tags.mlflow.parentRunId='{parent_run_id}'", output_format="list"
    )  # type: ignore
    logger.info(f"parent_run_id {parent_run_id}")
    logger.info(f"Found runs: {[run.info.run_name for run in runs]}")

    task_names = [task.name for task in tasks]
    logger.info(f"Will only run the following: {task_names}")

    table_columns = [
        "Task",
        "Metric",
        "Score",
        "mlflow_run_name",
        "mlflow_run_id",
        "mlflow_run_status",
    ]
    table_entries = []

    mlflow.set_tracking_uri(repeated_storage_uri)
    mlflow.set_experiment(experiment_name)
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    output_path = csv_folder / f"{experiment_name}_repeated_exp_mlflow.csv"
    if not os.path.isabs(output_path):
        raise Exception(
            f"output_path must be absolute."
        )

    # backbone_name = defaults.terratorch_task["model_args"]["backbone"]
    with mlflow.start_run(run_name=experiment_name, run_id=None) as run:
        for task in tasks:
            logger.info(f"\n\ntask: {task.name}")
            matching_runs = [
                run for run in runs if run.info.run_name.endswith(task.name)
            ]  # type: ignore
            if len(matching_runs) == 0:
                msg = f"No runs found for task {task.name}. Skipping."
                warnings.warn(msg)
                continue
            if len(matching_runs) > 1:
                msg = f"More than 1 run found for task {task.name}"
                raise Exception(msg)

            # check if there are already results for this task and exp in the folder
            if os.path.exists(output_path):
                logger.info("there are previous results from repeated experiments")
                existing_output = pd.read_csv(output_path, index_col=False)
                existing_output = existing_output[table_columns]
                existing_task_output = existing_output.loc[
                    existing_output["Task"] == task.name
                ].copy()
                rows, cols = existing_task_output.shape
                logger.info(f"rows: {rows} \t cols: {cols}")
                if rows > num_repetitions:
                    logger.info("task has complete results, will not re-run")
                    continue
                past_seeds = [
                    int(item.split("_")[-1])
                    for item in existing_task_output["mlflow_run_name"].tolist()
                ]
            else:
                past_seeds = []
            logger.info(f"past_seeds for task: {past_seeds}")

            # get best parameters
            best_params = matching_runs[0].data.params
            best_params = {k: literal_eval(v) for k, v in best_params.items()}

            training_spec = combine_with_defaults(task, defaults)
            lightning_task_class = training_spec.task.type.get_class_from_enum()
            
            experiment_info = mlflow.get_experiment_by_name(experiment_name)
            seeds = [randint(1, 5000) for i in range(num_repetitions * 5)]
            seeds = [seed for seed in seeds if seed not in past_seeds]

            for seed in seeds:
                if len(past_seeds) >= num_repetitions:
                    break

                seed_run_name = f"{task.name}_{seed}"
                logger.info(f"now trying: {seed_run_name}")
                seed_run_data = mlflow.search_runs(
                    experiment_ids=[experiment_info.experiment_id],
                    filter_string=f'tags."mlflow.runName" LIKE "{seed_run_name}"',
                    output_format="list",
                )  # type: ignore
                if len(seed_run_data) > 0:
                    continue

                score = non_remote_fit(
                    experiment_name=repeated_experiment_name,
                    parent_run_id=run.info.run_id,
                    storage_uri=repeated_storage_uri,
                    task=task,
                    training_spec=training_spec,
                    lightning_task_class=lightning_task_class,
                    best_params=best_params,
                    seed=seed,
                    backbone_import=backbone_import,
                    save_models=save_models,
                    report_on_best_val=report_on_best_val,
                )
                # check if run with name finished successfully
                logger.info(f"score: {score}")
                # TODO improve this sleep command - try to get a better estimate than this
                time.sleep(60)
                seed_run_data = mlflow.search_runs(
                    experiment_ids=[experiment_info.experiment_id],
                    filter_string=f'tags."mlflow.runName" LIKE "{seed_run_name}"',
                    output_format="list",
                )  # type: ignore

                logger.info(f"run for task {task.name} seed {seed} complete")
                if len(seed_run_data) > 0:
                    if seed_run_data[0].info.status != "FINISHED":
                        mlflow.delete_run(seed_run_data[0].info.run_id)
                        continue
                    past_seeds.append(seed)
                    new_data = pd.DataFrame(
                        {
                            "Task": [task.name],
                            "Metric": [task.metric.split("/")[-1]],
                            "Score": [score],
                            "mlflow_run_name": [seed_run_name],
                            "mlflow_run_id": [seed_run_data[0].info.run_id],
                            "mlflow_run_status": [seed_run_data[0].info.status],
                        }
                    )
                    logger.info(
                        f"completed seeds so far for this task: {len(past_seeds)}"
                    )
                    if os.path.exists(output_path):
                        logger.info(
                            "there are previous results from repeated experiments"
                        )

                        existing_output = pd.read_csv(output_path, index_col=False)
                        existing_output = existing_output[table_columns]
                        existing_output.reset_index(inplace=True)
                        existing_task_output = existing_output.loc[
                            existing_output["Task"] == task.name
                        ].copy()
                        rows, cols = existing_task_output.shape
                        logger.info(f"rows: {rows} \t cols: {cols}")
                        if rows == 0:
                            logger.info("no past results for this task")
                        existing_output = pd.concat(
                            [existing_output, new_data], axis=0
                        )
                        existing_output.reset_index(inplace=True)
                        existing_output = existing_output.drop(
                            columns=["index", "level_0"]
                        )
                        existing_output.to_csv(output_path, index=False)
                    else:
                        new_data.to_csv(output_path, index=False)


def _run_repeated_per_task(
    args: Namespace, #TODO: remove args
    model_spec: ModelSpec,
    training_spec: TrainingSpec,
    optimizer_spec: OptimizerSpec,
    callbacks_spec: CallbackSpec,
    logger: logging.RootLogger,
    task: TaskSpec,
    storage_uri: str,
    experiment_name: str,
    experiment_run_id: str,
    task_run_id: str | None = None,
    optimization_space: optimization_space_type | None = None,
    n_trials: int = 1,
    save_models: bool = False,
    sampler: BaseSampler | None = None,
    test_models: bool = False,
    seed: int = 42,
):     
    """
    Performs HPO on a single task

    Args:
        args: Namespace, #TODO: remove args
        model_spec: ModelSpec,
        training_spec: TrainingSpec,
        optimizer_spec: OptimizerSpec,
        callbacks_spec: CallbackSpec,
        logger: logging.RootLogger,
        task: TaskSpec,
        storage_uri: str,
        experiment_name: str,
        experiment_run_id: str,
        task_run_id: str | None = None,
        optimization_space: optimization_space_type | None = None,
        n_trials: int = 1,
        save_models: bool = False,
        sampler: BaseSampler | None = None,
        test_models: bool = False,
        seed: int = 42,

    """
    logger.info(
        f"starting backbone benchmark on task {task.name} {task_run_id=} {experiment_name=}"
    )
    if storage_uri.startswith("http"):
        optuna_db_path = Path(".") / "optuna_db"
    else:
        optuna_db_path = Path(storage_uri).parents[0] / "optuna_db"

    if not os.path.exists(optuna_db_path):
        os.makedirs(optuna_db_path)
    optuna_db_path = optuna_db_path / f"{experiment_name}_{experiment_run_id}"
    optuna_db_path = str(optuna_db_path)

    task_run_id = sync_mlflow_optuna(
        optuna_db_path=optuna_db_path,
        storage_uri=storage_uri,
        experiment_name=experiment_name,
        task_run_id=task_run_id,
        task=task,
        n_trials=n_trials,
        logger=logger,
    )
    if task_run_id is not None:
        # run_name is used only when run_id is unspecified.
        run_name = None
    else:
        run_name = task.name
    logger.info(f"start run: {run_name=} {task_run_id=}")
    with mlflow.start_run(run_name=run_name, nested=True, run_id=task_run_id) as run:
        logger.info(f"starting task run with id: {run.info.run_id}")
        if training_spec.epochs is None:
            raise Exception("Must specify epochs for training")
        

        # if no optimization params, just run it
        if optimization_space is None:
            return (
                *fit_model(
                    args=args,
                    model_spec=model_spec,
                    training_spec=training_spec,
                    optimizer_spec=optimizer_spec,
                    callbacks_spec=callbacks_spec,
                    task=task,
                    run_name=run.info.run_name,
                    experiment_name=experiment_name,
                    storage_uri=storage_uri,
                    parent_run_id=run.info.run_id,
                    save_models=save_models,
                    test_models=test_models,
                    seed=seed,
                    logger=logger,
                ),
            )
