"""PowerFlow-reconstruction IO-processor plugin registration.

Wired to the ``vllm.io_processor_plugins`` entry-point group. Per the vLLM
plugin contract, the registration function returns the *class path string* of
an ``IOProcessor`` subclass; vLLM imports and constructs it later with
``(vllm_config, renderer)``. Returning a string (rather than importing the
class here) keeps entry-point discovery cheap and avoids importing vLLM until
the processor is actually built.

The path is fully dot-separated: vLLM resolves it with
:func:`vllm.utils.import_utils.resolve_obj_by_qualname`, which does
``qualname.rsplit(".", 1)`` — a ``module:Class`` colon form (as used by the
model-registry entry points) would leave ``io_processor:GridFMPFIOProcessor``
as a single attribute name and fail to import.
"""

from __future__ import annotations

IO_PROCESSOR_CLASS_PATH = (
    "gridfm_graphkit.vllm.plugins.pf_reconstruction.io_processor.GridFMPFIOProcessor"
)


def register_pf_reconstruction() -> str:
    """Return the class path of the PowerFlow-reconstruction IO processor."""
    return IO_PROCESSOR_CLASS_PATH
