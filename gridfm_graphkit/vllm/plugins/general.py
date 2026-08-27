"""General vLLM plugin: register the GridFM pooling architecture.

Wired to the ``vllm.general_plugins`` entry-point group. vLLM calls
:func:`register_gridfm` once at startup; it registers the ``GridFMGNS``
architecture so a model whose ``config.json`` declares
``architectures: ["GridFMGNS"]`` resolves to
:class:`gridfm_graphkit.vllm.model.GridFMForPooling`.
"""

from __future__ import annotations

ARCHITECTURE = "GridFMGNS"
MODEL_CLASS_PATH = "gridfm_graphkit.vllm.model:GridFMForPooling"


def register_gridfm() -> None:
    """Register the GridFM pooling model with vLLM's model registry.

    Idempotent: skips registration if the architecture is already present, so
    repeated plugin loads (or a build that ships the arch natively) do not
    raise.
    """
    from vllm import ModelRegistry
    from vllm.logger import init_logger

    logger = init_logger(__name__)

    if ARCHITECTURE in ModelRegistry.get_supported_archs():
        logger.debug("%s already registered with vLLM; skipping.", ARCHITECTURE)
        return

    ModelRegistry.register_model(ARCHITECTURE, MODEL_CLASS_PATH)
    logger.info("Registered GridFM architecture %s with vLLM.", ARCHITECTURE)
