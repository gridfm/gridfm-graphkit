"""IO processor for GridFM PowerFlow reconstruction over vLLM's /pooling API.

Turns a JSON power-grid case into the normalized, masked graph tensors the
model consumes, and turns the model's packed output back into denormalized
per-node predictions and embeddings. The heavy lifting — graph construction,
normalization, task masking — is delegated to the existing gridfm-graphkit
pipeline so serving and training share one code path.

Imports vLLM at load time; only import when the ``vllm`` extra is installed.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import Any, Optional

import pandas as pd
import torch
from vllm.config import VllmConfig
from vllm.entrypoints.pooling.pooling.protocol import (
    IOProcessorRequest,
    IOProcessorResponse,
)
from vllm.inputs import PromptType
from vllm.outputs import PoolingRequestOutput
from vllm.plugins.io_processors.interface import (
    IOProcessor,
    IOProcessorInput,
    IOProcessorOutput,
)

from gridfm_graphkit.datasets.graph_builder import build_hetero_data
from gridfm_graphkit.vllm import graph_codec
from gridfm_graphkit.vllm.config import build_inference_bundle
from gridfm_graphkit.vllm.types import GridFMRequest, GridFMResponse
from gridfm_graphkit.vllm.utils import check_vllm_version

logger = logging.getLogger(__name__)

# vLLM added the renderer constructor argument after 0.16.0, and changed the
# multimodal input nesting after 0.14.0. We target the 0.26 line.
_RENDERER_ARG = check_vllm_version("0.16.0", ">")
_NESTED_MM_DATA = check_vllm_version("0.14.0", ">")

if _RENDERER_ARG:
    from vllm.renderers import BaseRenderer


class GridFMPFIOProcessor(IOProcessor):
    """Pre/post-processing for PowerFlow reconstruction and embeddings."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        renderer: "Optional[BaseRenderer]" = None,
    ) -> None:
        if renderer is None:
            super().__init__(vllm_config)
        else:
            super().__init__(vllm_config, renderer)

        pretrained_cfg = vllm_config.model_config.hf_config.to_dict()["pretrained_cfg"]
        bundle = build_inference_bundle(pretrained_cfg)
        # Model weights are loaded separately in the vLLM worker; here we only
        # need the normalizer (for de/normalization) and the task transforms
        # (for building mask_dict).
        self.normalizer = bundle.normalizer
        self.transforms = bundle.transforms
        self._requests: dict[str, GridFMRequest] = {}

    # --- request parsing --------------------------------------------------

    def parse_request(self, request: Any) -> IOProcessorInput:
        return self.parse_data(request)

    def parse_data(self, data: Any) -> IOProcessorInput:
        if isinstance(data, dict):
            return GridFMRequest(**data)
        if isinstance(data, IOProcessorRequest):
            if not hasattr(data, "data"):
                raise ValueError("missing 'data' field in IOProcessorRequest")
            request_data = data.data
            if isinstance(request_data, dict):
                return GridFMRequest(**request_data)
            raise ValueError("Unable to parse the request data")
        raise ValueError("Unable to parse request")

    def output_to_response(
        self,
        plugin_output: IOProcessorOutput,
    ) -> IOProcessorResponse:
        return IOProcessorResponse(
            request_id=plugin_output.request_id,
            data=plugin_output,
        )

    # --- pre-processing ---------------------------------------------------

    def pre_process(
        self,
        prompt: IOProcessorInput,
        request_id: str | None = None,
        **kwargs,
    ) -> PromptType | Sequence[PromptType]:
        return asyncio.run(self.pre_process_async(prompt, request_id, **kwargs))

    async def pre_process_async(
        self,
        prompt: IOProcessorInput,
        request_id: str | None = None,
        **kwargs,
    ) -> PromptType | Sequence[PromptType]:
        request: GridFMRequest = prompt
        case = request.case

        bus_df = pd.DataFrame(case.bus)
        gen_df = pd.DataFrame(case.gen)
        branch_df = pd.DataFrame(case.branch)

        # Build → normalize → task-mask, reusing the training pipeline.
        data = build_hetero_data(bus_df, gen_df, branch_df)
        self.normalizer.transform(data)
        data = self.transforms(data)

        fields = graph_codec.encode_hetero_data(data)

        multi_modal_data: dict[str, Any] = dict(fields)
        if _NESTED_MM_DATA:
            multi_modal_data = {"image": multi_modal_data}

        if not request_id:
            request_id = "offline"
        self._requests[request_id] = request

        return {"prompt_token_ids": [1], "multi_modal_data": multi_modal_data}

    # --- post-processing --------------------------------------------------

    def post_process(
        self,
        model_output: Sequence[PoolingRequestOutput],
        request_id: str | None = None,
        **kwargs,
    ) -> IOProcessorOutput:
        if not request_id:
            request_id = "offline"
        request = self._requests.pop(request_id, None)
        return_embeddings = request.return_embeddings if request else True
        return_predictions = request.return_predictions if request else True

        outputs = list(model_output)
        if len(outputs) != 1:
            raise ValueError(
                f"GridFM PowerFlow reconstruction expects exactly one pooling "
                f"output, got {len(outputs)}",
            )

        packed = outputs[0].outputs.data
        unpacked = graph_codec.unpack_outputs(torch.as_tensor(packed))

        bus_pred = unpacked["bus_pred"]
        gen_pred = unpacked["gen_pred"]

        response = GridFMResponse(
            num_buses=int(bus_pred.shape[0]),
            num_gens=int(gen_pred.shape[0]),
            request_id=request_id,
        )

        if return_predictions:
            # Denormalize in place (Vm/Va left as-is; power quantities scaled).
            pred_dict = {"bus": bus_pred.clone(), "gen": gen_pred.clone()}
            self.normalizer.inverse_output(pred_dict, batch=None)
            response.bus_predictions = pred_dict["bus"].tolist()
            response.gen_predictions = pred_dict["gen"].tolist()

        if return_embeddings:
            response.bus_embeddings = unpacked["bus_emb"].tolist()
            response.gen_embeddings = unpacked["gen_emb"].tolist()

        return response
