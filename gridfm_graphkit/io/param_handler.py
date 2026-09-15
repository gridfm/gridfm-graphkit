import torch
from gridfm_graphkit.training.loss import MixedLoss
from gridfm_graphkit.io.registries import (
    NORMALIZERS_REGISTRY,
    MODELS_REGISTRY,
    LOSS_REGISTRY,
    TRANSFORM_REGISTRY,
    TASK_REGISTRY,
    PHYSICS_DECODER_REGISTRY,
)

import argparse
from torch_geometric.transforms import Compose
from gridfm_graphkit.tasks.base_task import BaseTask


class NestedNamespace(argparse.Namespace):
    """
    A namespace object that supports nested structures, allowing for
    easy access and manipulation of hierarchical configurations.

    """

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                # Recursively convert dictionaries to NestedNamespace
                setattr(self, key, NestedNamespace(**value))
            elif isinstance(value, list):
                list_of_namespaces = []
                for element in value:
                    if isinstance(element, dict):
                        list_of_namespaces.append(NestedNamespace(**element))
                    else:
                        list_of_namespaces.append(element)
                setattr(self, key, list_of_namespaces)
            else:
                setattr(self, key, value)

    def to_dict(self):
        # Recursively convert NestedNamespace back to dictionary
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, NestedNamespace):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def flatten(self, parent_key="", sep="."):
        # Flatten the dictionary with dot-separated keys
        items = []
        for key, value in self.__dict__.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            if isinstance(value, NestedNamespace):
                items.extend(value.flatten(new_key, sep=sep).items())
            else:
                items.append((new_key, value))
        return dict(items)


def load_normalizer(args):
    """
    Load the appropriate data normalization methods

    Args:
        args (NestedNamespace): contains configs.

    Returns:
        tuple: Node and edge normalizers

    Raises:
        ValueError: If an unknown normalization method is specified.
    """
    method = args.data.normalization

    try:
        return NORMALIZERS_REGISTRY.create(
            method,
            args,
        )
    except KeyError:
        raise ValueError(f"Unknown transformation: {method}")


def get_loss_function(args):
    """
    Load the appropriate loss function

    Args:
        args (NestedNamespace): contains configs.

    Returns:
        nn.Module: Loss function

    Raises:
        ValueError: If an unknown loss function is specified.
    """
    loss_functions = []
    for loss_name, loss_args in zip(args.training.losses, args.training.loss_args):
        try:
            loss_functions.append(LOSS_REGISTRY.create(loss_name, loss_args, args))
        except KeyError:
            raise ValueError(f"Unknown loss: {loss_name}")

    warmup_epochs = getattr(args.training, "physics_warmup_epochs", 0)
    warmup_indices = [
        i
        for i, loss_name in enumerate(args.training.losses)
        if loss_name == "LayeredWeightedPhysics"
    ]

    return MixedLoss(
        loss_functions=loss_functions,
        weights=args.training.loss_weights,
        warmup_indices=warmup_indices,
        warmup_epochs=warmup_epochs,
    )


def load_model(args) -> torch.nn.Module:
    """
    Load the appropriate model

    Args:
        args (NestedNamespace): contains configs.

    Returns:
        nn.Module: The selected model initialized with the provided configurations.

    Raises:
        ValueError: If an unknown model type is specified.
    """
    model_type = args.model.type

    try:
        return MODELS_REGISTRY.create(model_type, args)
    except KeyError:
        raise ValueError(f"Unknown model type: {model_type}")


def get_task_transforms(args) -> Compose:
    """
    Load the task-specific transforms
    """

    task_transforms = args.task.task_name

    try:
        return TRANSFORM_REGISTRY.create(task_transforms, args)
    except KeyError:
        raise ValueError(f"Unknown task: {task_transforms}")


def get_task(args, data_normalizers) -> BaseTask:
    """
    Load the task module
    """
    task = args.task.task_name

    try:
        return TASK_REGISTRY.create(task, args, data_normalizers)
    except KeyError:
        raise ValueError(f"Unknown task: {task}")


def get_physics_decoder(args) -> torch.nn.Module:
    """
    Load the task module
    """
    task = args.task.task_name

    try:
        return PHYSICS_DECODER_REGISTRY.create(task)
    except KeyError:
        raise ValueError(f"No physics decoder associate to {task} task")
