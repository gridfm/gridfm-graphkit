import yaml
import argparse
from gridFM.datasets.data_normalization import *


class NestedNamespace(argparse.Namespace):
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


def parse_yaml(path: str):
    # Load YAML configuration and merge with command-line args
    with open(path, "r") as file:
        config_dict = yaml.safe_load(file)

    # Convert YAML config dictionary into an NestedNamespace
    config_args = NestedNamespace(**config_dict)
    return config_args

def load_normalizer(args):
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
        

