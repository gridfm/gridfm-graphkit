import yaml
import argparse
from gridFM.datasets.data_normalization import *
import itertools


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


def flatten_dict(d, parent_key="", sep="."):
    """
    Flatten a nested dictionary into a single-level dictionary with dot-separated keys.

    Args:
        d (dict): The dictionary to flatten.
        parent_key (str, optional): Prefix for the keys in the flattened dictionary.
        sep (str, optional): Separator for nested keys. Defaults to '.'.

    Returns:
        dict: A flattened version of the input dictionary.
    """
    items = []
    for key, value in d.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            items.extend(flatten_dict(value, new_key, sep=sep).items())
        else:
            items.append((new_key, value))
    return dict(items)


def unflatten_dict(d, sep="."):
    """
    Reconstruct a nested dictionary from a flattened dictionary with dot-separated keys.

    Args:
        d (dict): The flattened dictionary to unflatten.
        sep (str, optional): Separator used in the flattened keys. Defaults to '.'.

    Returns:
        dict: A nested dictionary reconstructed from the flattened input.
    """
    result = {}
    for key, value in d.items():
        parts = key.split(sep)
        target = result
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return result


def merge_dict(base, updates):
    """
    Recursively merge updates into a base dictionary, but only if the keys exist in the base.

    Args:
        base (dict): The original dictionary to be updated.
        updates (dict): The dictionary containing updates.

    Raises:
        KeyError: If a key in updates does not exist in base.
        TypeError: If a key in base is not a dictionary but updates attempt to provide nested values.
    """
    for key, value in updates.items():
        if key not in base:
            raise KeyError(f"Key '{key}' not found in base configuration.")

        if isinstance(value, dict):
            if not isinstance(base[key], dict):
                raise TypeError(
                    f"Default config expects  {type(base[key])}, but got a dict at key '{key}'"
                )
            # Recursively merge dictionaries
            merge_dict(base[key], value)
        else:
            # Update the existing key
            base[key] = value


def param_combination_gen(grid_config):
    """
    Generate all combinations of parameters from a nested dictionary

    Args:
        grid_config (dict): A nested dictionary where keys are parameter names
            and values are lists of possible values.

    Returns:
        list: A list of dictionaries representing all possible parameter combinations.
              Each dictionary corresponds to one combination.
    """

    # Flatten the grid config for combination generation
    flat_grid = flatten_dict(grid_config)

    # Separate keys and values for itertools.product
    keys, values = zip(*flat_grid.items())

    # Generate all combinations of parameters
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # Unflatten the combinations back into nested dictionaries
    nested_combinations = [unflatten_dict(comb) for comb in combinations]
    return nested_combinations


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

    if method == "minmax":
        return MinMaxNormalizer(), MinMaxNormalizer()
    elif method == "standard":
        return Standardizer(), Standardizer()
    elif method == "baseMVAnorm":
        return BaseMVANormalizer(baseMVA=args.data.baseMVA), IdentityNormalizer()
    elif method == "identity":
        return IdentityNormalizer(), IdentityNormalizer()
    else:
        raise ValueError(f"Unknown normalization method: {method}")
