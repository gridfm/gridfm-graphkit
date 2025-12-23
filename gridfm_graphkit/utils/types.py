"""
This module defines all the types expected at input. Used for type checking by jsonargparse.
"""

from ast import Dict
from typing import Literal
import copy
import enum
from dataclasses import dataclass, field, replace
from typing import Any, Optional, Union
from gridfm_graphkit.tasks import (
    FeatureReconstructionTask,
    # ContingencyAnalysisTask,
)
from gridfm_graphkit.datasets.powergrid_datamodule import LitGridDataModule


import logging


valid_task_types = type[
    FeatureReconstructionTask
    # | ContingencyAnalysisTask
]

direction_type_to_optuna = {"min": "minimize", "max": "maximize"}



@dataclass
class TaskTypeEnum(enum.Enum):
    """
    Enum for the type of task to be performed. segmentation, regression or classification.
    """

    feature_reconstruction = "feature_reconstruction"
    # contingency_analysis = "contingency_analysis"

    def get_class_from_enum(
        self,
    ) -> valid_task_types:
        match self.value:
            case TaskTypeEnum.feature_reconstruction.value:
                return FeatureReconstructionTask
            case TaskTypeEnum.contingency_analysis.value:
                return ContingencyAnalysisTask
            case _:
                raise TypeError("Task type does not exist")


class ParameterTypeEnum(enum.Enum):
    """
    Enum for the type of parameter allowed in ParameterBounds. integer or real.
    """

    integer = "int"
    real = "real"


@dataclass
class ParameterBounds:
    """
    Dataclass defining a numerical range to search over.

    Args:
        min (float | int): Minimum.
        max (float | int): Maximum.
        type (ParameterTypeEnum): Whether the range is in the space of integers or real numbers.
        log (bool): Whether to search over the log space (useful for parameters that vary wildly in scale, e.g. learning rate)
    """

    min: float | int
    max: float | int
    type: ParameterTypeEnum
    log: bool = False

    def __post_init__(self):
        if not isinstance(self.type, ParameterTypeEnum):
            self.type = ParameterTypeEnum(self.type)



optimization_space_type = dict[
    str, Union[list, dict, ParameterBounds, "optimization_space_type"]
]





@dataclass
class HyperParameterOptmizerSpec:
    """
    Parameters passed to define hyperparameter optimization. Only used with 'iterate' subcommand.

    These parameters are combined with any specified defaults to generate the final task parameters.

    Args:
        name (str): Name for this task
        type (TaskTypeEnum): Type of task.
        terratorch_task (dict): Arguments for the Terratorch Task.
        datamodule (BaseDataModule  | GeoBenchDataModule): Datamodule to be used.
        direction (str): One of min or max. Direction to optimize the metric in.
        metric (str): Metric to be optimized. Defaults to "val/loss".
        early_prune (bool): Whether to prune unpromising runs early. Defaults to False.
        early_stop_patience (int, None): Whether to use Lightning early stopping of runs. Defaults to None, which does not do early stopping.
        optimization_except (str[str]): HyperParameters from the optimization space to be ignored for this task.
        max_run_duration (str, None): maximum allowed run duration in the form DD:HH:MM:SS; will stop a run after this
            amount of time. Defaults to None, which doesn't stop runs by time.
    """
    experiment_name: str
    run_name: str
    
    results_folder: str
    save_models: bool = False
    n_trials: int = 5
    num_repetitions: int = 2
    repeat_on_best: bool = True
    bayesian_search: bool = True
    continue_existing_experiment: bool = True
    test_models: bool = False
    report_on_best_val: bool = True
    run_id: str | None = None
    optimization_space: dict | None = None




@dataclass
class TrainingSpec:
    """
    Parameters passed to define lightning trainer
        
    """
    batch_size: int
    epochs: int
    losses: list[str]
    loss_weights: list[float]
    accelerator: str
    devices: str
    strategy: str
    log_every_n_steps: int = 1
    enable_progress_bar: bool = False




@dataclass
class ModelSpec:
    """
    Parameters passed to define Model
      
    """
    attention_head: int
    dropout: float
    edge_dim: int
    hidden_size: int
    input_dim: int
    num_layers: int
    output_dim: int
    pe_dim: int
    type: str
    model_path: str




@dataclass
class OptimizerSpec:
    """
    Parameters passed to define Optimization and Scheduling parameters. Learning rate will be overwritten for 'iterate' subcommand. 
    
    """
    learning_rate: float
    type: str
    optimizer_params: dict
    scheduler_type: str | None
    scheduler_params: dict | None



@dataclass
class DataSpec:
    """
    Parameters passed to define training data. Ignored for 'iterate' subcommand. 
    
    """
    networks: list[str]
    scenarios: list[int]
    normalization: str
    baseMVA: int
    mask_type: str
    mask_value: float
    mask_ratio: float
    mask_dim: int
    learn_mask: bool
    val_ratio: float
    test_ratio: float
    workers: int
    data_path: str



@dataclass
class CallbackSpec:
    """
    Parameters passed to define training callbacks

    Args:
        patience (int): patience for early stopping
        tol (int): ...
        
    """
    #TODO: use dicts for each callback type
    patience: int | None = None
    tol: int | None = None
    max_run_duration: int | None = None
    monitor_learning_rate: bool = True
    optuna_early_prune: bool = False #only processed with iterate command




@dataclass
class TaskSpec:
    """
    Parameters passed to define each of the tasks. Including DataSpec per task. Only used with 'iterate' subcommand.

    These parameters are combined with any specified defaults to generate the final task parameters.

    Args:
        name (str): Name for this task
        type (TaskTypeEnum): Type of task.
        metric (str): Metric to be optimized. Defaults to "val/loss".
        direction (str): One of min or max. Direction to optimize the metric in.
        data: datamodule (BaseDataModule  | GeoBenchDataModule): Datamodule to be used.

    """
    name: str
    type: TaskTypeEnum = field(repr=False)
    data: DataSpec # = field(repr=False)
    metric: str = "val/constraint_violations"
    direction: Literal["min", "max"] = "min"


def recursive_merge(first_dict: dict[str, Any], second_dict: dict[str, Any]):
    # consider using deepmerge instead of this
    for key, val in second_dict.items():
        if key not in first_dict:
            first_dict[key] = val
        else:
            # if it is a dictionary, recurse deeper
            if isinstance(val, dict):
                recursive_merge(first_dict[key], val)
            # if it is not further nested, just replace the value
            else:
                first_dict[key] = val