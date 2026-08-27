"""vLLM pooling-model wrapper around GridFM's heterogeneous GNN.

This adapts :class:`gridfm_graphkit.models.gnn_heterogeneous_gns.GNS_heterogeneous`
to vLLM's attention-free, multimodal pooling-model interface, closely following
vLLM's own ``Terratorch`` wrapper (``vllm/model_executor/models/terratorch.py``,
Apache-2.0). The graph arrives as a bag of named tensors under a single
multimodal modality; the model rebuilds the ``HeteroData``, runs the GNN with
``return_embeddings=True``, and returns one packed tensor that vLLM's identity
pooler passes straight through to the IO processor.

This module imports vLLM at load time and must only be imported when the
optional ``vllm`` extra is installed.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
from transformers import BatchFeature

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs import ModalityData, MultiModalDataDict, MultiModalInput, mm_input
from vllm.logger import init_logger
from vllm.model_executor.layers.pooler import IdentityPooler
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    ImageItem,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
    PlaceholderRange,
)
from vllm.multimodal.parse import (
    DictEmbeddingItems,
    ModalityDataItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    ProcessorInputs,
    PromptUpdate,
    TimingContext,
)
from vllm.sequence import IntermediateTensors

from vllm.model_executor.models.interfaces import (
    IsAttentionFree,
    MultiModalEmbeddings,
    SupportsMultiModal,
)
from vllm.model_executor.models.interfaces_base import attn_type

from gridfm_graphkit.vllm import graph_codec
from gridfm_graphkit.vllm.config import build_inference_bundle

logger = init_logger(__name__)

# The multimodal modality bucket. vLLM's multimodal machinery is organized by
# modality name; "image" is used generically here (as vLLM's own Terratorch
# wrapper does) to route our graph tensors through the multimodal path.
_MODALITY = "image"


def _pretrained_cfg(vllm_config: VllmConfig) -> dict[str, Any]:
    return vllm_config.model_config.hf_config.to_dict()["pretrained_cfg"]


def _mm_fields_config(*, is_shared: bool) -> Mapping[str, MultiModalFieldConfig]:
    """Field config for every graph tensor, all under the ``image`` modality.

    ``is_shared`` mirrors vLLM's Terratorch wrapper: shared (unbatched) during
    dummy-data parsing, batched during the real ``apply`` pass.
    """
    fields: dict[str, MultiModalFieldConfig] = {}
    for name in graph_codec.GRAPH_FIELDS:
        fields[name] = (
            MultiModalFieldConfig.shared(_MODALITY, batch_size=1)
            if is_shared
            else MultiModalFieldConfig.batched(_MODALITY)
        )
    return fields


def _dummy_graph_fields(
    *, bus_feat: int, gen_feat: int, edge_feat: int
) -> dict[str, torch.Tensor]:
    """A minimal, well-typed graph for vLLM memory profiling.

    Values are arbitrary; only dtype, rank and feature width matter. Three buses
    and two generators — deliberately **more than one generator**: the GNN's
    physics decoder does ``gen_temp.squeeze()`` before a ``scatter_add`` over
    generators, which collapses a single-generator tensor to a 0-d scalar and
    breaks the scatter. Real cases always have several generators; the dummy
    must too.

    Feature widths must equal the GNN's declared input dims — the fields fed to
    :meth:`GridFMForPooling.forward` are already the *post-transform* tensors
    (the IO processor applies normalization + task masking before encoding), so
    the widths here are ``input_bus_dim`` / ``input_gen_dim`` / ``edge_dim``
    from the model config, not the raw column layouts in ``graph_builder``.
    """
    n_bus, n_gen = 3, 2

    # Two branches (0-1, 1-2), each encoded in both directions.
    n_branch = 4

    bus_x = torch.ones(n_bus, bus_feat, dtype=torch.float)
    gen_x = torch.ones(n_gen, gen_feat, dtype=torch.float)
    bus_bus_edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    bus_bus_edge_attr = torch.ones(n_branch, edge_feat, dtype=torch.float)
    # Generator g attaches to bus g (g in 0..n_gen-1).
    gen_bus_edge_index = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    bus_gen_edge_index = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)

    # Masks: bus/gen/branch are 2D [rows, feat]; PQ/PV/REF are 1D [n_bus].
    return {
        graph_codec.FIELD_BUS_X: bus_x,
        graph_codec.FIELD_GEN_X: gen_x,
        graph_codec.FIELD_BUS_BUS_EDGE_INDEX: bus_bus_edge_index,
        graph_codec.FIELD_BUS_BUS_EDGE_ATTR: bus_bus_edge_attr,
        graph_codec.FIELD_GEN_BUS_EDGE_INDEX: gen_bus_edge_index,
        graph_codec.FIELD_BUS_GEN_EDGE_INDEX: bus_gen_edge_index,
        graph_codec.FIELD_MASK_BUS: torch.zeros(n_bus, bus_feat, dtype=torch.bool),
        graph_codec.FIELD_MASK_GEN: torch.zeros(n_gen, gen_feat, dtype=torch.bool),
        graph_codec.FIELD_MASK_BRANCH: torch.zeros(
            n_branch, edge_feat, dtype=torch.bool
        ),
        # One REF bus, one PV bus (with a generator), the rest PQ — a valid
        # single-slack topology.
        graph_codec.FIELD_MASK_PQ: torch.tensor(
            [i > 1 for i in range(n_bus)], dtype=torch.bool
        ),
        graph_codec.FIELD_MASK_PV: torch.tensor(
            [i == 1 for i in range(n_bus)], dtype=torch.bool
        ),
        graph_codec.FIELD_MASK_REF: torch.tensor(
            [i == 0 for i in range(n_bus)], dtype=torch.bool
        ),
    }


class GridFMMultiModalDataParser(MultiModalDataParser):
    """Route our dict-of-named-tensors graph through the multimodal path.

    vLLM's default parser only understands a handful of built-in image
    representations (PIL, ndarray, embedding tensor). Our graph arrives as a
    ``dict[str, Tensor]`` of named fields, so — mirroring vLLM's own Terratorch
    parser — we intercept the ``image`` bucket: when it holds a dict we wrap it
    in :class:`DictEmbeddingItems`, which turns the named tensors into
    per-modality multimodal kwargs. Without this the dummy-profiling run yields
    an empty modality set (``KeyError: "Modality 'image' not found"``).
    """

    def _parse_image_data(
        self,
        data: dict[str, torch.Tensor] | ModalityData[ImageItem],
    ) -> ModalityDataItems[Any, Any] | None:
        if isinstance(data, dict):
            return DictEmbeddingItems(
                data,
                modality=_MODALITY,
                required_fields=set(graph_codec.GRAPH_FIELDS),
                fields_factory=lambda _data: _mm_fields_config(is_shared=True),
            )
        return super()._parse_image_data(data)

    def parse_mm_data(self, mm_data: MultiModalDataDict) -> MultiModalDataItems:
        if _MODALITY not in mm_data:
            mm_data = {_MODALITY: mm_data}
        return super().parse_mm_data(mm_data)


class GridFMProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {_MODALITY: 1}

    def get_data_parser(self) -> MultiModalDataParser:
        return GridFMMultiModalDataParser()

    def get_input_dims(self) -> tuple[int, int, int]:
        """``(input_bus_dim, input_gen_dim, edge_dim)`` from the model config.

        These are the *post-transform* feature widths the GNN's encoders expect
        — exactly what :meth:`GridFMForPooling.forward` receives — so the dummy
        profiling graph must match them.
        """
        from gridfm_graphkit.vllm.config import CONFIG_KEY

        model_cfg = self.get_hf_config().to_dict()["pretrained_cfg"][CONFIG_KEY][
            "model"
        ]
        return (
            int(model_cfg["input_bus_dim"]),
            int(model_cfg["input_gen_dim"]),
            int(model_cfg["edge_dim"]),
        )


class GridFMDummyInputsBuilder(BaseDummyInputsBuilder[GridFMProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return ""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        if mm_options:
            logger.warning(
                "Configurable multimodal profiling options are not supported "
                "for GridFM and are ignored.",
            )
        bus_feat, gen_feat, edge_feat = self.info.get_input_dims()
        return {
            _MODALITY: _dummy_graph_fields(
                bus_feat=bus_feat, gen_feat=gen_feat, edge_feat=edge_feat
            )
        }


class GridFMMultiModalProcessor(BaseMultiModalProcessor[GridFMProcessingInfo]):
    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
        *,
        is_shared: bool = True,
    ) -> Mapping[str, MultiModalFieldConfig]:
        return _mm_fields_config(is_shared=is_shared)

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        return []

    def apply(
        self,
        inputs: ProcessorInputs,
        timing_ctx: TimingContext,
    ) -> MultiModalInput:
        mm_items = inputs.mm_data_items

        with timing_ctx.record("apply_hf_processor"):
            _, passthrough_data = self._get_hf_mm_data(mm_items)
            mm_processed_data = BatchFeature(
                {
                    k: torch.as_tensor(v).unsqueeze(0)
                    for k, v in passthrough_data.items()
                },
                tensor_type="pt",
            )

        mm_kwargs = MultiModalKwargsItems.from_hf_inputs(
            mm_processed_data,
            self._get_mm_fields_config(
                mm_processed_data,
                inputs.hf_processor_mm_kwargs,
                is_shared=False,
            ),
        )

        with timing_ctx.record("get_mm_hashes"):
            mm_hashes = inputs.get_mm_hashes(self.info.model_id)

        mm_placeholders = {_MODALITY: [PlaceholderRange(offset=0, length=0)]}

        return mm_input(
            prompt_token_ids=[1],
            mm_kwargs=mm_kwargs,
            mm_hashes=mm_hashes,
            mm_placeholders=mm_placeholders,
        )


@attn_type("attention_free")
@MULTIMODAL_REGISTRY.register_processor(
    GridFMMultiModalProcessor,
    info=GridFMProcessingInfo,
    dummy_inputs=GridFMDummyInputsBuilder,
)
class GridFMForPooling(nn.Module, IsAttentionFree, SupportsMultiModal):
    """Serve GridFM's heterogeneous GNN as a vLLM pooling model."""

    supports_multimodal_raw_input_only = True
    is_pooling_model = True

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith(_MODALITY):
            return None
        raise ValueError(f"Only the '{_MODALITY}' modality is supported")

    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        pretrained_cfg = _pretrained_cfg(vllm_config)
        bundle = build_inference_bundle(pretrained_cfg)

        # The wrapped GNN. Named ``model`` so checkpoint keys map to
        # ``model.<param>`` for AutoWeightsLoader (see load_weights).
        self.model = bundle.model
        self.model.eval()

        # Identity pooler: whatever forward returns becomes the pooled output.
        self.pooler = IdentityPooler()

    def get_language_model(self) -> nn.Module:
        # vLLM (>=0.25) calls ``get_language_model()`` on every model during
        # load. Multimodal models are expected to expose a language-model
        # submodule; this GNN has none, so — mirroring vLLM's own Terratorch
        # wrapper — we report ourselves as the "language model". Combined with
        # ``embed_input_ids`` (which maps the single sentinel token to a
        # zero-width embedding) this satisfies the interface without a real LM.
        return self

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # No real tokens are used; the mandatory single prompt token maps to a
        # zero-width embedding (mirrors vLLM's Terratorch wrapper).
        return torch.empty((input_ids.shape[0], 0))

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        # vLLM's multimodal collation prepends an "items" dimension to every
        # field: one graph per prompt token. Real requests carry a single graph
        # (leading dim 1); vLLM's warmup ``_dummy_run`` replicates the dummy
        # graph across ``max_num_reqs`` items. vLLM then treats our output as
        # ``[num_tokens, hidden]`` — it slices ``hidden_states[:num_tokens]`` in
        # ``_pool`` and indexes ``hidden_states[logit_indices]`` (up to
        # ``num_tokens - 1``) during warmup. So we must return one packed row per
        # item, keeping the leading dim equal to the token/item count and all
        # graph data in the trailing dimension (mirroring vLLM's Terratorch
        # wrapper). Collapsing to a single graph would truncate real output and
        # blow the warmup index out of bounds.
        fields = {
            name: kwargs[name] for name in graph_codec.GRAPH_FIELDS if name in kwargs
        }
        n_items = next(iter(fields.values())).shape[0]

        packed_rows: list[torch.Tensor] = []
        for i in range(n_items):
            item_fields = {name: tensor[i] for name, tensor in fields.items()}
            data = graph_codec.decode_hetero_data(item_fields)
            with torch.no_grad():
                predictions, embeddings = self.model(data, return_embeddings=True)
            packed_rows.append(
                graph_codec.pack_outputs(
                    bus_pred=predictions["bus"],
                    gen_pred=predictions["gen"],
                    bus_emb=embeddings["bus"],
                    gen_emb=embeddings["gen"],
                )
            )

        return torch.stack(packed_rows, dim=0)

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        """Load a GridFM checkpoint into the wrapped GNN.

        Accepts either a flat stream of ``(name, tensor)`` pairs (safetensors
        export) or a single ``("state_dict", OrderedDict)`` entry (raw torch
        checkpoint). Names are prefixed with ``model.`` to match the wrapped
        submodule.
        """
        params_list: list[tuple[str, torch.Tensor]] = []
        for key, value in weights:
            if isinstance(value, (dict, OrderedDict)):
                if key == "state_dict":
                    for name, weight in value.items():
                        params_list.append((f"model.{name}", weight))
                    break
            elif isinstance(value, torch.Tensor):
                name = key if key.startswith("model.") else f"model.{key}"
                params_list.append((name, value))

        loader = AutoWeightsLoader(self)
        return loader.load_weights(params_list)
