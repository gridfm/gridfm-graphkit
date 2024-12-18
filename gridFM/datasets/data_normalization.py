import torch
import numpy as np
from abc import ABC, abstractmethod
from gridFM.datasets.globals import *
import math


class Normalizer(ABC):
    @abstractmethod
    def fit(self, data: np.ndarray) -> dict:
        """Calculate the parameters required for normalization."""
        pass

    @abstractmethod
    def fit_from_dict(self, params: dict):
        """Set normalization parameters from a dictionary if not already set."""
        pass

    @abstractmethod
    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """Apply the normalization to the data."""
        pass

    @abstractmethod
    def inverse_transform(self, normalized_data: torch.Tensor) -> torch.Tensor:
        """Reverse the normalization process."""
        pass


class MinMaxNormalizer(Normalizer):
    def __init__(self):
        self.min_val = None
        self.max_val = None

    def to(self, device):
        self.min_val = self.min_val.to(device)
        self.max_val = self.max_val.to(device)

    def fit(self, data: np.ndarray) -> dict:
        """
        Calculate min and max values for each feature from the data.

        Args:
            data (np.ndarray): Input data tensor.

        Returns:
            dict: Dictionary containing min and max values for each feature.
        """
        self.min_val = torch.tensor(data.min(axis=0), dtype=torch.float)
        self.max_val = torch.tensor(data.max(axis=0), dtype=torch.float)

        return {"min_value": self.min_val, "max_value": self.max_val}

    def fit_from_dict(self, params: dict):
        if self.min_val is None:
            self.min_val = params.get("min_value")
        if self.max_val is None:
            self.max_val = params.get("max_value")

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            data (torch.Tensor): Input tensor to be normalized.

        Returns:
            torch.Tensor: Min-Max normalized data.
        """
        if self.min_val is None or self.max_val is None:
            raise ValueError("fit must be called before transform.")

        diff = self.max_val - self.min_val
        diff[diff == 0] = 1  # Avoid division by zero for features with zero range
        return (data - self.min_val) / diff

    def inverse_transform(self, normalized_data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            normalized_data (np.ndarray): Normalized data.

        Returns:
            torch.Tensor: Denormalized data.
        """
        if self.min_val is None or self.max_val is None:
            raise ValueError("fit must be called before inverse_transform.")

        diff = self.max_val - self.min_val
        diff[diff == 0] = 1
        return (normalized_data * diff) + self.min_val


class Standardizer(Normalizer):
    def __init__(self):
        self.mean = None
        self.std = None

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)

    def fit(self, data: np.ndarray) -> dict:
        """
        Calculate mean and standard deviation for each feature from the data.

        Args:
            data (np.ndarray): Input data tensor.

        Returns:
            dict: Dictionary containing mean and standard deviation for each feature.
        """
        self.mean = torch.tensor(data.mean(axis=0), dtype=torch.float)
        self.std = torch.tensor(data.std(axis=0), dtype=torch.float)

        return {"mean_value": self.mean, "std_value": self.std}

    def fit_from_dict(self, params: dict):
        if self.mean is None:
            self.mean = params.get("mean_value")
        if self.std is None:
            self.std = params.get("std_value")

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            data (np.ndarray): Input tensor to be standardized.

        Returns:
            torch.Tensor: Standardized data.
        """
        if self.mean is None or self.std is None:
            raise ValueError("fit must be called before transform.")

        std = self.std.clone()
        std[std == 0] = 1  # Avoid division by zero for features with zero std
        return (data - self.mean) / std

    def inverse_transform(self, normalized_data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            normalized_data (np.ndarray): Standardized data.

        Returns:
            torch.Tensor: Denormalized data.
        """
        if self.mean is None or self.std is None:
            raise ValueError("fit must be called before inverse_transform.")

        std = self.std.clone()
        std[std == 0] = 1
        return (normalized_data * std) + self.mean


class BaseMVANormalizer(Normalizer):
    def __init__(self, node_data: bool, baseMVA_orig: float = 100.0):
        # baseMVA_orig is the one provided by the MATPOWER casefiles
        self.node_data = node_data
        self.baseMVA_orig = baseMVA_orig
        self.baseMVA = None

    def to(self, device):
        pass

    def fit(self, data: np.ndarray, baseMVA: float = None) -> dict:
        """
        No need to compute baseMVA

        Args:
            data (np.ndarray): Input data tensor.

        Returns:
            dict: Dictionary containing baseMVA value.
        """

        if self.node_data:
            self.baseMVA = torch.tensor(data[:, [PD, QD, PG, QG]].max(), dtype=torch.float)
        else:
            self.baseMVA = baseMVA


        return {"baseMVA_orig": self.baseMVA_orig, "baseMVA": self.baseMVA}

    def fit_from_dict(self, params: dict):
        if self.baseMVA is None:
            self.baseMVA = params.get("baseMVA")
        if self.baseMVA_orig is None:
            self.baseMVA_orig = params.get("baseMVA_orig")

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            data (torch.Tensor): Input tensor to be normalized.

        Returns:
            torch.Tensor: Data divided by BaseMVA.
        """

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
        """
        Args:
            normalized_data (torch.Tensor): Normalized data.

        Returns:
            torch.Tensor: Denormalized data.
        """
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
    def fit(self, data: np.ndarray) -> dict:
        """
        No parameters to compute for IdentityNormalizer.

        Args:
            data (np.ndarray): Input data tensor.

        Returns:
            dict: An empty dictionary, as no parameters are computed.
        """
        return {}

    def fit_from_dict(self, params: dict):
        # No parameters to set for IdentityNormalizer
        pass

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            data (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The input data unchanged.
        """
        return data

    def inverse_transform(self, normalized_data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            normalized_data (torch.Tensor): Input data (already unchanged).

        Returns:
            torch.Tensor: The input data unchanged.
        """
        return normalized_data
