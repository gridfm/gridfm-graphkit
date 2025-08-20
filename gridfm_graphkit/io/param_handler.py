from gridfm_graphkit.training.loss import (
    PBELoss,
    MaskedMSELoss,
    SCELoss,
    MixedLoss,
    MSELoss,
)
from gridfm_graphkit.io.registries import (
    MASKING_REGISTRY,
    NORMALIZERS_REGISTRY,
    MODELS_REGISTRY,
)

import argparse


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
            True,
            args,
        ), NORMALIZERS_REGISTRY.create(method, False, args)
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
    for loss_name in args.training.losses:
        if loss_name == "MSE":
            loss_functions.append(MSELoss())
        elif loss_name == "MaskedMSE":
            loss_functions.append(MaskedMSELoss())
        elif loss_name == "SCE":
            loss_functions.append(SCELoss())
        elif loss_name == "PBE":
            loss_functions.append(PBELoss())
        else:
            raise ValueError(f"Unknown loss function: {loss_name}")

    return MixedLoss(loss_functions=loss_functions, weights=args.training.loss_weights)


def load_model(args):
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


def get_transform(args):
    """
    Load the appropriate dataset transform from the registry.
    """
    mask_type = args.data.mask_type

    try:
        return MASKING_REGISTRY.create(mask_type, args)
    except KeyError:
        raise ValueError(f"Unknown transformation: {mask_type}")
