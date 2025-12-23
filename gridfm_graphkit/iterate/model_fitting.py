"""
This module contains all the logic for fitting models
"""

import abc
import copy
import dataclasses
import importlib
import os
import shutil
import types
import uuid
import torch
import warnings
from abc import abstractmethod
from functools import wraps
from typing import Callable
import pandas as pd
import lightning.pytorch as pl
import mlflow
import optuna
from lightning import Callback, Trainer
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    
)
from jsonargparse import Namespace

import logging
from lightning.pytorch.loggers.mlflow import MLFlowLogger

from optuna.integration import PyTorchLightningPruningCallback


from gridfm_graphkit.training.callbacks import get_training_callbacks
from gridfm_graphkit.datasets.powergrid_datamodule import LitGridDataModule

from gridfm_graphkit.utils.types import (
    OptimizerSpec, 
    ModelSpec, 
    TrainingSpec, 
    TaskSpec,
    CallbackSpec,
    valid_task_types,
    ParameterBounds,
    ParameterTypeEnum,
    optimization_space_type,
    recursive_merge,
    )

from gridfm_graphkit.tasks import (
    FeatureReconstructionTask,
    # ContingencyAnalysisTask,
)




from gridfm_graphkit.iterate.utils import get_logger

LOGGER = get_logger()


os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = (
    "1"  # disable tune loggers, will add csv and json manually. If this is not here, it will log to tensorboard automatically
)


class ParameterPicker(abc.ABC):
    @abstractmethod
    def pick_categorical(self, variable, choices):
        pass

    @abstractmethod
    def pick_int(self, variable, low, high):
        pass

    @abstractmethod
    def pick_float(self, variable, low, high, log=False):
        pass


class OptunaParameterPicker(ParameterPicker):
    def __init__(self, trial: optuna.Trial):
        super().__init__()
        self.trial = trial

    def pick_categorical(self, variable, choices):
        return self.trial.suggest_categorical(variable, choices)

    def pick_int(self, variable, low, high):
        return self.trial.suggest_int(variable, low, high)

    def pick_float(self, variable, low, high, log=False):
        return self.trial.suggest_float(variable, low, high, log=log)


def inject_hparams(
    training_spec: TrainingSpec,
    optimizer_spec: OptimizerSpec,
    config: dict):
    assert isinstance(config, dict), (
        f"Error! Unexpected config type: {config}"
    )
    training_spec_with_hparams = copy.deepcopy(training_spec)
    optimizer_spec_with_hparams = copy.deepcopy(optimizer_spec)

    recursive_merge(training_spec_with_hparams, config)
    recursive_merge(optimizer_spec_with_hparams, config)

    return training_spec_with_hparams, optimizer_spec_with_hparams


def generate_parameters(
    parameter_picker: ParameterPicker,
    current_hparams: dict,
    hparam_space: dict,
    ignore_keys: set[str] | None = None,
    dictionary_position: list[str] | None = None,
):
    if ignore_keys is None:
        ignore_keys = set()
    if dictionary_position is None:
        dictionary_position = []
    _generate_parameters(
        parameter_picker,
        current_hparams,
        hparam_space,
        ignore_keys,
        dictionary_position,
    )


def _generate_parameters(
    parameter_picker: ParameterPicker,
    current_hparams: dict,
    hparam_space: dict,
    ignore_keys: set[str],
    dictionary_position: list[str],
):
    for parameter, space in hparam_space.items():
        if parameter in ignore_keys:
            continue
        # if its a dictionary, continue to recurse
        if isinstance(space, dict):
            if parameter not in current_hparams:
                current_hparams[parameter] = {}
            dictionary_position.append(parameter)
            _generate_parameters(
                parameter_picker,
                current_hparams[parameter],
                hparam_space[parameter],
                ignore_keys,
                dictionary_position,
            )
            dictionary_position.pop()
        # if not, get a value from the parameter_picker and insert it with the name prepended by the dictionary position
        # this is important so that the full path of the parameter is used
        # this will avoid confusion between parameters with the same name but from different components
        else:
            full_parameter_name = ".".join(dictionary_position + [parameter])
            if isinstance(space, list):
                suggestion = parameter_picker.pick_categorical(
                    full_parameter_name, space
                )
                current_hparams[parameter] = suggestion
            elif isinstance(space, ParameterBounds):
                match space.type:
                    case ParameterTypeEnum.integer:
                        current_hparams[parameter] = parameter_picker.pick_int(
                            full_parameter_name,
                            int(space.min),
                            int(space.max),
                        )
                    case ParameterTypeEnum.real:
                        current_hparams[parameter] = parameter_picker.pick_float(
                            full_parameter_name, space.min, space.max, log=space.log
                        )
                    case _:
                        raise Exception(
                            f"Type {space.type} not recognized. Suggest one of {[e.value for e in ParameterTypeEnum]}"
                        )
            else:
                raise Exception(
                    "Leaves of optimization space must be lists or ParameterBounds"
                )



            




"""
single node - optuna
"""
def launch_training(
    trainer: Trainer,
    model: FeatureReconstructionTask, #TODO: create basetask in tasks folder
    optimizer_spec: OptimizerSpec,
    datamodule: LitGridDataModule,
    run_name: str,
    experiment_name: str,
    metric: str,
    storage_uri: str,
    parent_run_id: str,
    direction: str,
    test_models: bool,
    delete_models_after_testing: bool,
) -> float:
    
    with mlflow.start_run(run_name=run_name, nested=True) as run:
        mlflow.set_tag("mlflow.parentRunId", parent_run_id)
        # explicitly log batch_size. Since it is not a model param, it will not be logged
        mlflow.log_param("batch_size", datamodule.batch_size)

        trainer.logger = MLFlowLogger(
            experiment_name=experiment_name,
            run_id=run.info.run_id,
            save_dir=storage_uri,
            log_model=not delete_models_after_testing,
        )
        trainer.fit(model, datamodule=datamodule)
        if test_models:
            test_metrics = trainer.test(
                model,
                ckpt_path="best", 
                datamodule=datamodule)
            test_metrics =test_metrics[0]
        if delete_models_after_testing:
            # delete the checkpoints folder in the run
            ckpts_folder = os.path.join(
                trainer.logger.save_dir,
                str(trainer.logger.name),
                trainer.logger.version,
                "checkpoints",
            )
            shutil.rmtree(ckpts_folder)

        client = mlflow.tracking.MlflowClient(
            tracking_uri=storage_uri,
        )

        if not metric.lower().startswith("val"):
            raise Exception(
                f"Metric {metric} does not start with `val`. Please choose a validation metric"
            )
        for_pd_collect = []
        val_metrics_names = []

        print(f'{client.get_run(run.info.run_id)=}')

        for cname in client.get_run(run.info.run_id).data.metrics:
            print(f'{cname=}')
            
        for metric_name in client.get_run(run.info.run_id).data.metrics:
            if metric_name.lower().startswith("val"):
                val_metrics_names.append(metric_name)
                val_metric_history = client.get_metric_history(
                    run.info.run_id, metric_name
                )
                pd_convertible_metric_history = [
                    {
                        "metric_name": mm.key,
                        "step": mm.step,
                        "value": mm.value,
                    }
                    for mm in val_metric_history
                ]
                for_pd_collect += pd_convertible_metric_history
        df_val_metrics = pd.DataFrame.from_records(for_pd_collect)
        df_val_metrics = df_val_metrics.set_index(
            ["metric_name", "step"], verify_integrity=True
        )
        series_val_metrics = df_val_metrics["value"]
        assert metric in series_val_metrics, (
            f"Error! {metric} is not in {series_val_metrics}"
        )
        if direction == "max":
            best_step = series_val_metrics[metric].idxmax()
        elif direction == "min":
            best_step = series_val_metrics[metric].idxmin()
        else:
            raise Exception(
                f"Error! Direction must be either `max` or `min` but got {direction}"
            )

        for val_metric_name in val_metrics_names:
            mlflow.log_metric(
                f"best_step_{val_metric_name}",
                series_val_metrics[(val_metric_name, best_step)],
            )

        return series_val_metrics[(metric, best_step)]


def fit_model(
    args: Namespace,
    model_spec: ModelSpec,
    training_spec: TrainingSpec,
    optimizer_spec: OptimizerSpec,
    callbacks_spec: CallbackSpec,
    task: TaskSpec,
    run_name: str,
    experiment_name: str,
    storage_uri: str,
    parent_run_id: str,
    logger: logging.RootLogger,
    save_models: bool = False,
    test_models: bool = False,
    seed: int = 42,
    trial: optuna.Trial | None = None,
) -> tuple[float, str]:
    pl.seed_everything(seed, workers=True)
    training_spec_copy = copy.deepcopy(training_spec)

    #get callbacks
    callbacks: list[Callback] = get_training_callbacks(callbacks_spec)
    if callbacks_spec.optuna_early_prune and trial is not None:
        callbacks.append(
            PyTorchLightningPruningCallback(trial, monitor="Validation loss")
        )
    if len(callbacks) > 0:
        warnings.warn(
            "Callbacks passed to trainer. Make sure these are stateless, as they will not be reinitialized for each task!"
        )

    delete_models_after_testing = False
    if test_models and not save_models:
        # we need to save the models during training to be able to test but can be deleted afterwards
        save_models = True
        delete_models_after_testing = True
    if save_models:
        callbacks.append(
            ModelCheckpoint(monitor=task.metric, mode=task.direction)
        )
    enable_checkpointing = False
    if any([isinstance(cb, ModelCheckpoint) for cb in callbacks]):
        enable_checkpointing=True

    # # initialize datamodule 
    args.data = task.data
    datamodule = LitGridDataModule(args, task.data.data_path)

    #initialize model
    lightning_task_class: valid_task_types = task.type.get_class_from_enum()
    model = lightning_task_class(
        args,#TODO: load model, training, optim separataly
        datamodule.node_normalizers,
        datamodule.edge_normalizers,
    )
    logger.info(f"Loading model weights from {model_spec.model_path}")
    state_dict = torch.load(model_spec.model_path)
    model.load_state_dict(state_dict)

    # initialize trainer
    trainer = Trainer(
        accelerator=training_spec_copy.accelerator,
        devices=training_spec_copy.devices,
        strategy=training_spec_copy.strategy,
        log_every_n_steps=training_spec_copy.log_every_n_steps,
        # default_root_dir=args.log_dir,
        max_epochs=training_spec_copy.epochs,
        callbacks=callbacks,
        enable_checkpointing=enable_checkpointing,
        enable_progress_bar=training_spec_copy.enable_progress_bar,
    )

    logger.info(
        f"launch_training {trainer=} {lightning_task_class=} {datamodule=} \
        {run_name=} {experiment_name=} {task.metric=} {storage_uri=} {task.direction=}"
    )
    return (
        launch_training(
            trainer=trainer,
            model=model,
            optimizer_spec=optimizer_spec,
            datamodule=datamodule,
            run_name=run_name,
            experiment_name=experiment_name,
            metric=task.metric,
            storage_uri=storage_uri,
            parent_run_id=parent_run_id,
            direction=task.direction,
            test_models=test_models,
            delete_models_after_testing=delete_models_after_testing,
        )
    )


def fit_model_with_hparams(
    args: Namespace,
    model_spec: ModelSpec,
    training_spec: TrainingSpec,
    optimizer_spec: OptimizerSpec,
    callbacks_spec: CallbackSpec,
    task: TaskSpec,
    optimization_space: optimization_space_type,
    run_name: str,
    experiment_name: str,
    storage_uri: str,
    parent_run_id: str,
    logger: logging.RootLogger,
    save_models: bool = False,
    test_models: bool = False,
    seed: int = 42,
    trial: optuna.Trial | None = None,
) -> float:
    """
    Generate parameters using the optuna trial from the given parameters.
    Then inject these into the given task.
    It is important to make sure to not overwrite the task passed in the arguments, or these updates may affect
    subsequent trials.
    """
    current_hparams: dict[str, int | float | str | bool] = {}
    generate_parameters(
        OptunaParameterPicker(trial),
        current_hparams,
        optimization_space,
    )

    training_spec_with_hparams, optimizer_spec_with_hparams = inject_hparams(
        training_spec, optimizer_spec, current_hparams
    )

    output =  fit_model(
        args=args,
        model_spec=model_spec,
        training_spec=training_spec,
        optimizer_spec=optimizer_spec,
        callbacks_spec=callbacks_spec,
        task=task,
        run_name=f"{run_name}_{trial.number}",
        experiment_name=experiment_name,
        storage_uri=storage_uri,
        parent_run_id=parent_run_id,
        save_models=save_models,
        test_models=test_models,
        seed=seed,
        logger=logger,
        trial=trial,
    )  # return only the metric value for optuna

    print(f'{output=}')

    return output

