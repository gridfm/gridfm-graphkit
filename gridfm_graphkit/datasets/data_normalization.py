from gridfm_graphkit.datasets.globals import PD, QD, PG, QG, VA
import torch
from abc import ABC, abstractmethod


class Normalizer(ABC):
    """
    Abstract base class for all normalization strategies.
    """

    @abstractmethod
    def fit(self, data: torch.Tensor) -> dict:
        """
        Fit normalization parameters from data.

        Args:
            data: Input tensor.

        Returns:
            Dictionary of computed parameters.
        """

    @abstractmethod
    def fit_from_dict(self, params: dict):
        """
        Set parameters from a precomputed dictionary.

        Args:
            params: Dictionary of parameters.
        """

    @abstractmethod
    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Normalize the input data.

        Args:
            data: Input tensor.

        Returns:
            Normalized tensor.
        """

    @abstractmethod
    def inverse_transform(self, normalized_data: torch.Tensor) -> torch.Tensor:
        """
        Undo normalization.

        Args:
            normalized_data: Normalized tensor.

        Returns:
            Original tensor.
        """


class MinMaxNormalizer(Normalizer):
    """
    Scales each feature to the [0, 1] range.
    """

    def __init__(self):
        self.min_val = None
        self.max_val = None

    def to(self, device):
        self.min_val = self.min_val.to(device)
        self.max_val = self.max_val.to(device)

    def fit(self, data: torch.Tensor) -> dict:
        self.min_val, _ = data.min(axis=0)
        self.max_val, _ = data.max(axis=0)

        return {"min_value": self.min_val, "max_value": self.max_val}

    def fit_from_dict(self, params: dict):
        if self.min_val is None:
            self.min_val = params.get("min_value")
        if self.max_val is None:
            self.max_val = params.get("max_value")

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        if self.min_val is None or self.max_val is None:
            raise ValueError("fit must be called before transform.")

        diff = self.max_val - self.min_val
        diff[diff == 0] = 1  # Avoid division by zero for features with zero range
        return (data - self.min_val) / diff

    def inverse_transform(self, normalized_data: torch.Tensor) -> torch.Tensor:
        if self.min_val is None or self.max_val is None:
            raise ValueError("fit must be called before inverse_transform.")

        diff = self.max_val - self.min_val
        diff[diff == 0] = 1
        return (normalized_data * diff) + self.min_val


class Standardizer(Normalizer):
    """
    Standardizes each feature to zero mean and unit variance.
    """

    def __init__(self):
        self.mean = None
        self.std = None

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)

    def fit(self, data: torch.Tensor) -> dict:
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0)

        return {"mean_value": self.mean, "std_value": self.std}

    def fit_from_dict(self, params: dict):
        if self.mean is None:
            self.mean = params.get("mean_value")
        if self.std is None:
            self.std = params.get("std_value")

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise ValueError("fit must be called before transform.")

        std = self.std.clone()
        std[std == 0] = 1  # Avoid division by zero for features with zero std
        return (data - self.mean) / std

    def inverse_transform(self, normalized_data: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise ValueError("fit must be called before inverse_transform.")

        std = self.std.clone()
        std[std == 0] = 1
        return (normalized_data * std) + self.mean


class BaseMVANormalizer(Normalizer):
    """
    In power systems, a suitable normalization strategy must preserve the physical properties of
    the system. A known method is the conversion to the per-unit (p.u.) system, which expresses
    electrical quantities such as voltage, current, power, and impedance as fractions of predefined
    base values. These base values are usually chosen based on system parameters, such as rated
    voltage. The per-unit conversion ensures that power system equations remain scale-invariant,
    preserving fundamental physical relationships.
    """

    def __init__(self, node_data: bool, baseMVA_orig: float = 100.0):
        """
        Args:
            node_data: Whether data is node-level or edge-level (PD, QD, PG, QG, VA).
            baseMVA_orig: Original baseMVA (e.g. from MATPOWER).
        """
        self.node_data = node_data
        self.baseMVA_orig = baseMVA_orig
        self.baseMVA = None

    def to(self, device):
        pass

    def fit(self, data: torch.Tensor, baseMVA: float = None) -> dict:
        if self.node_data:
            self.baseMVA = data[:, [PD, QD, PG, QG]].max()
        else:
            self.baseMVA = baseMVA

        return {"baseMVA_orig": self.baseMVA_orig, "baseMVA": self.baseMVA}

    def fit_from_dict(self, params: dict):
        if self.baseMVA is None:
            self.baseMVA = params.get("baseMVA")
        if self.baseMVA_orig is None:
            self.baseMVA_orig = params.get("baseMVA_orig")

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        if self.baseMVA is None:
            raise ValueError("BaseMVA is not specified")

        if self.baseMVA == 0:
            raise ZeroDivisionError("BaseMVA is 0.")

        if self.node_data:
            data[:, PD] = data[:, PD] / self.baseMVA
            data[:, QD] = data[:, QD] / self.baseMVA
            data[:, PG] = data[:, PG] / self.baseMVA
            data[:, QG] = data[:, QG] / self.baseMVA
            data[:, VA] = data[:, VA] * torch.pi / 180.0
        else:
            data = data * self.baseMVA_orig / self.baseMVA

        return data

    def inverse_transform(self, normalized_data: torch.Tensor) -> torch.Tensor:
        if self.baseMVA is None:
            raise ValueError("fit must be called before inverse_transform.")

        if self.node_data:
            normalized_data[:, PD] = normalized_data[:, PD] * self.baseMVA
            normalized_data[:, QD] = normalized_data[:, QD] * self.baseMVA
            normalized_data[:, PG] = normalized_data[:, PG] * self.baseMVA
            normalized_data[:, QG] = normalized_data[:, QG] * self.baseMVA
            normalized_data[:, VA] = normalized_data[:, VA] * 180.0 / torch.pi
        else:
            normalized_data = normalized_data * self.baseMVA / self.baseMVA_orig

        return normalized_data


class IdentityNormalizer(Normalizer):
    """
    No normalization: returns data unchanged.
    """

    def fit(self, data: torch.Tensor) -> dict:
        return {}

    def fit_from_dict(self, params: dict):
        pass

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        return data

    def inverse_transform(self, normalized_data: torch.Tensor) -> torch.Tensor:
        return normalized_data
