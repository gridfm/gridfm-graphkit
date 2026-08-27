"""Small helpers for the GridFM vLLM integration."""

from __future__ import annotations

from packaging import version

# The interface this integration targets. vLLM's IO-processor and pooling-model
# APIs shift between minor releases (renderer constructor arg, multimodal input
# nesting, pooling model mixins), so the plugin is written against this line and
# refuses to load silently against an untested one.
SUPPORTED_VLLM = ">=0.26,<0.27"


def check_vllm_version(target_version: str, comparison: str) -> bool:
    """Compare the installed vLLM version against ``target_version``.

    Args:
        target_version: A version string, e.g. ``"0.26.0"``.
        comparison: One of ``==``, ``!=``, ``<``, ``<=``, ``>``, ``>=``.

    Returns:
        The boolean result of the comparison.

    Raises:
        ImportError: If vLLM is not installed.
        ValueError: If ``comparison`` is not a recognized operator.
    """
    try:
        from vllm import __version__ as vllm_version
    except ImportError as exc:  # pragma: no cover - exercised only without vLLM
        raise ImportError(
            "vLLM is not installed. Install gridfm-graphkit with the 'vllm' "
            "extra (pip install 'gridfm-graphkit[vllm]') to use the serving "
            "integration.",
        ) from exc

    current = version.parse(vllm_version)
    target = version.parse(target_version)
    ops = {
        "==": current == target,
        "!=": current != target,
        "<": current < target,
        "<=": current <= target,
        ">": current > target,
        ">=": current >= target,
    }
    if comparison not in ops:
        raise ValueError(f"Invalid comparison operator: {comparison}")
    return ops[comparison]
