import glob

import pytest
import yaml

from gridfm_graphkit.io.config_version import (
    CONFIG_VERSION,
    CONVERSION_AUTO_COPY,
    CONVERSION_AUTO_INLINE,
    CONVERSION_NO,
    _convert_v0_to_v1,
    detect_version,
    resolve_config,
    upgrade_config,
)


# A representative v0 (pre-versioning) optimizer config.
V0_CONFIG = {
    "optimizer": {
        "beta1": 0.9,
        "beta2": 0.999,
        "learning_rate": 0.0005,
        "lr_decay": 0.7,
        "lr_patience": 5,
    },
    "seed": 0,
}

# The v1 form the converter is expected to produce (matches PR #92 schema).
V1_OPTIMIZER = {
    "type": "AdamW",
    "learning_rate": 0.0005,
    "optimizer_params": {"betas": [0.9, 0.999]},
    "scheduler_type": "ReduceLROnPlateau",
    "scheduler_params": {"mode": "min", "factor": 0.7, "patience": 5},
}


# ---- detect_version ----


def test_detect_version_missing_is_v0():
    assert detect_version({"optimizer": {}}) == 0


def test_detect_version_explicit():
    assert detect_version({"version": 1, "optimizer": {}}) == 1


def test_detect_version_non_int_raises():
    with pytest.raises(ValueError):
        detect_version({"version": "1"})


# ---- conversion ----


def test_convert_v0_to_v1_optimizer_block():
    converted = _convert_v0_to_v1(V0_CONFIG)
    assert converted["version"] == 1
    assert converted["optimizer"] == V1_OPTIMIZER


def test_convert_v0_to_v1_does_not_mutate_input():
    original = {
        "optimizer": {
            "beta1": 0.9,
            "beta2": 0.999,
            "learning_rate": 0.0005,
            "lr_decay": 0.7,
            "lr_patience": 5,
        },
    }
    snapshot = yaml.safe_dump(original)
    _convert_v0_to_v1(original)
    assert yaml.safe_dump(original) == snapshot


def test_convert_preserves_unknown_optimizer_keys():
    cfg = {"optimizer": {"learning_rate": 1e-3, "custom_flag": True}}
    converted = _convert_v0_to_v1(cfg)
    assert converted["optimizer"]["custom_flag"] is True


def test_convert_missing_optimizer_raises():
    with pytest.raises(ValueError):
        _convert_v0_to_v1({"seed": 0})


def test_upgrade_config_reaches_current_version():
    upgraded = upgrade_config(V0_CONFIG)
    assert detect_version(upgraded) == CONFIG_VERSION


# ---- resolve_config ----


def test_resolve_current_version_passthrough():
    cfg = {"version": CONFIG_VERSION, "optimizer": V1_OPTIMIZER}
    assert resolve_config(cfg, "c.yaml", CONVERSION_NO) is cfg


def test_resolve_old_version_no_mode_raises_with_help():
    with pytest.raises(ValueError) as exc:
        resolve_config(V0_CONFIG, "old.yaml", CONVERSION_NO)
    msg = str(exc.value)
    assert "old.yaml" in msg
    assert CONVERSION_AUTO_INLINE in msg
    assert CONVERSION_AUTO_COPY in msg


def test_resolve_auto_inline_migrates_without_writing(tmp_path):
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(yaml.safe_dump(V0_CONFIG))
    before = cfg_file.read_text()

    result = resolve_config(V0_CONFIG, str(cfg_file), CONVERSION_AUTO_INLINE)

    assert detect_version(result) == CONFIG_VERSION
    assert result["optimizer"] == V1_OPTIMIZER
    # file on disk untouched
    assert cfg_file.read_text() == before


def test_resolve_auto_copy_writes_file(tmp_path):
    cfg_file = tmp_path / "cfg.yaml"
    out_file = tmp_path / "converted.yaml"

    result = resolve_config(
        V0_CONFIG,
        str(cfg_file),
        CONVERSION_AUTO_COPY,
        converted_config_path=str(out_file),
    )

    assert out_file.exists()
    written = yaml.safe_load(out_file.read_text())
    assert written == result
    assert written["optimizer"] == V1_OPTIMIZER
    assert written["version"] == CONFIG_VERSION


def test_resolve_auto_copy_default_path(tmp_path):
    cfg_file = tmp_path / "cfg.yaml"
    resolve_config(V0_CONFIG, str(cfg_file), CONVERSION_AUTO_COPY)
    expected = tmp_path / f"cfg.v{CONFIG_VERSION}.yaml"
    assert expected.exists()


def test_resolve_future_version_raises():
    cfg = {"version": CONFIG_VERSION + 1}
    with pytest.raises(ValueError):
        resolve_config(cfg, "future.yaml", CONVERSION_NO)


def test_resolve_unknown_mode_raises():
    with pytest.raises(ValueError):
        resolve_config(V0_CONFIG, "c.yaml", "bogus_mode")


# ---- shipped configs ----


@pytest.mark.parametrize(
    "yaml_path",
    glob.glob("examples/config/*.yaml") + glob.glob("tests/config/*.yaml"),
)
def test_shipped_configs_are_current_version(yaml_path):
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    assert detect_version(cfg) == CONFIG_VERSION
    # resolve is a no-op passthrough for current-version configs
    assert resolve_config(cfg, yaml_path, CONVERSION_NO) is cfg
