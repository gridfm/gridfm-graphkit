"""vLLM serving integration for gridfm-graphkit.

This subpackage lets a trained GridFM model be served through vLLM's ``/pooling``
endpoint. It provides two vLLM plugins, discovered via entry points declared in
``pyproject.toml`` and only active when the optional ``vllm`` extra is installed:

- a **general plugin** registering the ``GridFMGNS`` architecture
  (:class:`gridfm_graphkit.vllm.model.GridFMForPooling`), and
- an **IO-processor plugin** (``gridfm_pf_reconstruction``) that turns a
  power-grid case into graph tensors and reads back per-node embeddings and
  reconstructed PowerFlow quantities.

Only lightweight, vLLM-free helpers are re-exported here so that importing
``gridfm_graphkit.vllm`` never requires vLLM. The model and IO-processor modules
import vLLM at module load and should only be imported when it is installed.
"""

from __future__ import annotations

from gridfm_graphkit.vllm.types import (
    GridCase,
    GridFMRequest,
    GridFMResponse,
)
from gridfm_graphkit.vllm.utils import SUPPORTED_VLLM, check_vllm_version

__all__ = [
    "GridCase",
    "GridFMRequest",
    "GridFMResponse",
    "SUPPORTED_VLLM",
    "check_vllm_version",
]
