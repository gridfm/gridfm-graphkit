"""Offline end-to-end test of the GridFM serving core (no vLLM dependency).

Exercises the whole serving path except the vLLM wrapper layer: rebuild the
model/normalizer/transforms from a config payload, build a graph from raw
tables, normalize + mask it, round-trip it through the codec, run the GNN with
``return_embeddings=True``, pack/unpack the output, and denormalize. Model
weights are random (no published checkpoint), so only shapes and wiring are
asserted — which is exactly what the codec and config layers are responsible
for.
"""

import os

import pandas as pd
import torch
import yaml

from gridfm_graphkit.datasets.graph_builder import build_hetero_data
from gridfm_graphkit.vllm import graph_codec
from gridfm_graphkit.vllm.config import (
    CONFIG_KEY,
    NORMALIZER_STATS_KEY,
    build_inference_bundle,
)
from gridfm_graphkit.vllm.export import build_hf_config

_CONFIG = "tests/config/datamodule_test_base_config.yaml"
_RAW = "tests/data/case14_ieee/raw"
_STATS = "tests/data/case14_ieee/processed/data_stats_HeteroDataMVANormalizer.pt"


def _pretrained_cfg() -> dict:
    with open(_CONFIG) as f:
        gridfm_config = yaml.safe_load(f)
    stats = torch.load(_STATS, weights_only=True)
    return {
        CONFIG_KEY: gridfm_config,
        NORMALIZER_STATS_KEY: {k: float(v) for k, v in stats.items()},
    }


def _case_frames(scenario: int = 0):
    """Raw scenario tables with per-bus gen Q-limits merged in.

    Mirrors the aggregation the on-disk dataset performs in ``process`` so the
    bus table carries every column ``build_hetero_data`` expects.
    """
    bus = pd.read_parquet(os.path.join(_RAW, "bus_data.parquet"))
    gen = pd.read_parquet(os.path.join(_RAW, "gen_data.parquet"))
    branch = pd.read_parquet(os.path.join(_RAW, "branch_data.parquet"))

    agg_gen = (
        gen.groupby(["scenario", "bus"])[["min_q_mvar", "max_q_mvar"]]
        .sum()
        .reset_index()
    )
    bus = bus.merge(agg_gen, on=["scenario", "bus"], how="left").fillna(0)

    return (
        bus[bus["scenario"] == scenario].reset_index(drop=True),
        gen[gen["scenario"] == scenario].reset_index(drop=True),
        branch[branch["scenario"] == scenario].reset_index(drop=True),
    )


def test_build_inference_bundle_constructs_objects(generate_processed_test_data):
    bundle = build_inference_bundle(_pretrained_cfg())
    assert bundle.model is not None
    assert bundle.normalizer.baseMVA is not None
    # Transforms is a torch_geometric Compose (callable on HeteroData).
    assert callable(bundle.transforms)


def test_offline_serving_core_round_trip(generate_processed_test_data):
    pretrained_cfg = _pretrained_cfg()
    bundle = build_inference_bundle(pretrained_cfg)

    bus_df, gen_df, branch_df = _case_frames(0)
    n_bus = len(bus_df)
    n_gen = len(gen_df)

    # Pre-process: build → normalize → task-mask (as the IO processor does).
    data = build_hetero_data(bus_df, gen_df, branch_df)
    bundle.normalizer.transform(data)
    data = bundle.transforms(data)
    assert "bus" in data.mask_dict and "gen" in data.mask_dict

    # Codec: encode → (wire) → decode.
    fields = graph_codec.encode_hetero_data(data)
    rebuilt = graph_codec.decode_hetero_data(fields)

    # Model forward with embeddings (random weights: shapes only).
    bundle.model.eval()
    with torch.no_grad():
        predictions, embeddings = bundle.model(rebuilt, return_embeddings=True)

    packed = graph_codec.pack_outputs(
        predictions["bus"],
        predictions["gen"],
        embeddings["bus"],
        embeddings["gen"],
    )
    out = graph_codec.unpack_outputs(packed)

    assert out["bus_pred"].shape[0] == n_bus
    assert out["gen_pred"].shape[0] == n_gen
    assert out["bus_emb"].shape[0] == n_bus
    assert out["gen_emb"].shape[0] == n_gen
    assert out["bus_emb"].shape[1] == out["gen_emb"].shape[1]

    # Denormalization runs and preserves shapes.
    pred_dict = {"bus": out["bus_pred"].clone(), "gen": out["gen_pred"].clone()}
    bundle.normalizer.inverse_output(pred_dict, batch=None)
    assert pred_dict["bus"].shape == out["bus_pred"].shape
    assert pred_dict["gen"].shape == out["gen_pred"].shape


def test_build_hf_config_shape():
    cfg = build_hf_config(
        gridfm_config={"model": {"type": "GNS_heterogeneous"}},
        normalizer_stats={"baseMVA": 100.0, "baseMVA_orig": 100.0, "vn_kv_max": 345.0},
    )
    assert cfg["architectures"] == ["GridFMGNS"]
    # No model_type: vLLM routes by architecture (HF AutoConfig rejects unknown types).
    assert "model_type" not in cfg
    # Required by TimmWrapperConfig.from_dict (the AutoConfig path for a
    # model_type-less config carrying a `pretrained_cfg` key).
    assert cfg["num_classes"] == 0
    assert cfg["pretrained_cfg"][CONFIG_KEY]["model"]["type"] == "GNS_heterogeneous"
    assert cfg["pretrained_cfg"][NORMALIZER_STATS_KEY]["baseMVA"] == 100.0
