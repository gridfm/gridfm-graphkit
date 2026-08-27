"""Export a trained GridFM model to a HuggingFace/vLLM-loadable directory.

vLLM loads a model from a directory containing a ``config.json`` (describing the
architecture and carrying the model's own config) and a weights file. This
helper writes such a directory for a trained GridFM PowerFlow model so that
``vllm serve <dir> --runner pooling --io-processor-plugin gridfm_pf_reconstruction``
works, and the same directory can later be pushed to the Hub.

Pure torch + json + safetensors; no vLLM import.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from gridfm_graphkit.vllm.config import CONFIG_KEY, NORMALIZER_STATS_KEY
from gridfm_graphkit.vllm.plugins.general import ARCHITECTURE

CONFIG_FILENAME = "config.json"
WEIGHTS_FILENAME = "model.safetensors"


def build_hf_config(
    gridfm_config: dict[str, Any],
    normalizer_stats: dict[str, float],
) -> dict[str, Any]:
    """Assemble the ``config.json`` payload for a served GridFM model.

    Args:
        gridfm_config: The full gridfm training config (as a plain dict), enough
            for :func:`gridfm_graphkit.vllm.config.build_inference_bundle` to
            rebuild the model, normalizer and task transforms.
        normalizer_stats: ``baseMVA``, ``baseMVA_orig`` and ``vn_kv_max``.

    Returns:
        A JSON-serializable dict with ``architectures``, ``num_classes`` and a
        ``pretrained_cfg`` block. Deliberately carries **no** ``model_type``: vLLM
        parses the config through HuggingFace ``AutoConfig``, which rejects an
        unknown ``model_type`` even under ``--trust-remote-code``. Without a
        ``model_type`` but *with* a ``pretrained_cfg`` key, ``AutoConfig`` routes
        the dict to ``TimmWrapperConfig`` (the same path TerraTorch's served
        config takes), and vLLM then routes to the plugin-registered
        ``GridFMGNS`` architecture. ``num_classes: 0`` is required by
        ``TimmWrapperConfig.from_dict`` — without it ``num_labels`` resolves to
        ``None`` and the config raises during construction.
    """
    return {
        "architectures": [ARCHITECTURE],
        "num_classes": 0,
        "pretrained_cfg": {
            CONFIG_KEY: gridfm_config,
            NORMALIZER_STATS_KEY: {k: float(v) for k, v in normalizer_stats.items()},
        },
    }


def export_model(
    output_dir: str | Path,
    state_dict: dict[str, torch.Tensor],
    gridfm_config: dict[str, Any],
    normalizer_stats: dict[str, float],
) -> Path:
    """Write a vLLM-loadable model directory.

    Args:
        output_dir: Destination directory (created if missing).
        state_dict: The trained ``GNS_heterogeneous`` state dict.
        gridfm_config: Full gridfm config as a plain dict.
        normalizer_stats: baseMVA / baseMVA_orig / vn_kv_max.

    Returns:
        The path to ``output_dir``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = build_hf_config(gridfm_config, normalizer_stats)
    (output_dir / CONFIG_FILENAME).write_text(json.dumps(config, indent=2))

    # Contiguous CPU tensors for safetensors.
    cpu_state = {k: v.detach().cpu().contiguous() for k, v in state_dict.items()}
    try:
        from safetensors.torch import save_file

        save_file(cpu_state, str(output_dir / WEIGHTS_FILENAME))
    except ImportError:
        # Fall back to a torch checkpoint; the model's load_weights accepts a
        # ("state_dict", OrderedDict) stream as well.
        torch.save({"state_dict": cpu_state}, str(output_dir / "pytorch_model.bin"))

    return output_dir
