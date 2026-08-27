"""Rebuild a GridFM inference bundle from a serialized config.

A served GridFM model carries its full training config plus normalizer
statistics inside the HuggingFace ``config.json`` (under ``pretrained_cfg``).
This module turns that plain-dict payload back into the live objects the
serving path needs — the model, the normalizer, and the task masking
transform — reusing the existing gridfm-graphkit builders so there is exactly
one construction path shared between training and inference.

No vLLM import here on purpose: the same bundle is built offline (tests,
export verification) and inside the vLLM worker.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch_geometric.transforms import Compose

from gridfm_graphkit.datasets.normalizers import Normalizer
from gridfm_graphkit.io.param_handler import (
    NestedNamespace,
    get_task_transforms,
    load_model,
    load_normalizer,
)

# Keys inside the HF config's ``pretrained_cfg`` block.
CONFIG_KEY = "gridfm_config"
NORMALIZER_STATS_KEY = "normalizer_stats"


@dataclass
class InferenceBundle:
    """The live objects needed to serve one GridFM task."""

    args: NestedNamespace
    model: torch.nn.Module
    normalizer: Normalizer
    transforms: Compose


def build_args(gridfm_config: dict[str, Any]) -> NestedNamespace:
    """Turn a plain config dict into the nested namespace the builders expect."""
    return NestedNamespace(**gridfm_config)


def build_inference_bundle(pretrained_cfg: dict[str, Any]) -> InferenceBundle:
    """Construct model + normalizer + task transforms from a config payload.

    Args:
        pretrained_cfg: The ``pretrained_cfg`` dict from the HF ``config.json``.
            Must contain :data:`CONFIG_KEY` (the full gridfm config) and
            :data:`NORMALIZER_STATS_KEY` (baseMVA / baseMVA_orig / vn_kv_max).

    Returns:
        An :class:`InferenceBundle`. Model weights are *not* loaded here — vLLM
        loads them separately via ``load_weights``; offline callers load them
        themselves.
    """
    if CONFIG_KEY not in pretrained_cfg:
        raise ValueError(
            f"pretrained_cfg is missing the '{CONFIG_KEY}' block required to "
            f"rebuild the GridFM model",
        )
    args = build_args(pretrained_cfg[CONFIG_KEY])

    model = load_model(args)
    normalizer = load_normalizer(args)

    stats = pretrained_cfg.get(NORMALIZER_STATS_KEY)
    if stats is None:
        raise ValueError(
            f"pretrained_cfg is missing the '{NORMALIZER_STATS_KEY}' block "
            f"required to denormalize model inputs/outputs",
        )
    normalizer.fit_from_dict(
        {k: torch.as_tensor(v, dtype=torch.float) for k, v in stats.items()},
    )

    transforms = get_task_transforms(args)

    return InferenceBundle(
        args=args,
        model=model,
        normalizer=normalizer,
        transforms=transforms,
    )
