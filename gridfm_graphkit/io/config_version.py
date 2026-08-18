"""Config-file versioning and migration.

gridfm-graphkit config files carry a top-level integer ``version`` key. The
current schema version is :data:`CONFIG_VERSION`. Configs that predate the
introduction of versioning have no ``version`` key and are treated as
version ``0``.

Schema history
--------------

* **v0** — the original ``optimizer`` block, with a hard-coded ``AdamW``
  optimizer and ``ReduceLROnPlateau`` scheduler::

      optimizer:
        beta1: 0.9
        beta2: 0.999
        learning_rate: 0.0005
        lr_decay: 0.7
        lr_patience: 5

* **v1** — configurable optimizer/scheduler (PR #92). The optimizer type and
  its parameters, and the scheduler type and its parameters, are named
  explicitly::

      optimizer:
        type: AdamW
        learning_rate: 0.0005
        optimizer_params:
          betas: [0.9, 0.999]
        scheduler_type: ReduceLROnPlateau
        scheduler_params:
          mode: min
          factor: 0.7      # previously lr_decay
          patience: 5      # previously lr_patience

**This branch of gridfm-graphkit requires config version 1.** A v0 config must
be migrated before it can be used for training/evaluation — either in memory
via ``--config_conversion auto_inline`` or written to a new file via
``--config_conversion auto_copy`` (see :func:`resolve_config`).
"""

from __future__ import annotations

import copy
import os

import yaml

# The config-schema version this code understands and produces.
CONFIG_VERSION = 1

# Historical defaults hard-coded in the v0 optimizer/scheduler path, used to
# fill gaps when a v0 config omits an (in practice always-present) key.
_V0_DEFAULT_BETA1 = 0.9
_V0_DEFAULT_BETA2 = 0.999

# Valid values for the --config_conversion CLI option.
CONVERSION_NO = "no"
CONVERSION_AUTO_INLINE = "auto_inline"
CONVERSION_AUTO_COPY = "auto_copy"
CONVERSION_MODES = (CONVERSION_NO, CONVERSION_AUTO_INLINE, CONVERSION_AUTO_COPY)


def detect_version(config: dict) -> int:
    """Return the schema version of a loaded config dict.

    A config with no top-level ``version`` key predates versioning and is
    treated as v0.
    """
    version = config.get("version", 0)
    if not isinstance(version, int):
        raise ValueError(
            f"Config 'version' must be an integer, got {version!r} "
            f"({type(version).__name__}).",
        )
    return version


def _convert_v0_to_v1(config: dict) -> dict:
    """Migrate a v0 config dict to v1 (returns a new dict, input untouched).

    Reproduces the exact behaviour v0 hard-coded: an ``AdamW`` optimizer with
    ``betas=(beta1, beta2)`` and a ``ReduceLROnPlateau`` scheduler with
    ``mode="min"``, ``factor=lr_decay``, ``patience=lr_patience``.
    """
    new = copy.deepcopy(config)
    optimizer = new.get("optimizer")
    if not isinstance(optimizer, dict):
        raise ValueError(
            "Cannot convert config to v1: missing or malformed 'optimizer' block.",
        )

    beta1 = optimizer.pop("beta1", _V0_DEFAULT_BETA1)
    beta2 = optimizer.pop("beta2", _V0_DEFAULT_BETA2)
    lr_decay = optimizer.pop("lr_decay", None)
    lr_patience = optimizer.pop("lr_patience", None)

    # v0 always used AdamW; carry over the learning_rate unchanged.
    migrated: dict = {"type": "AdamW"}
    if "learning_rate" in optimizer:
        migrated["learning_rate"] = optimizer.pop("learning_rate")
    migrated["optimizer_params"] = {"betas": [beta1, beta2]}

    # v0 always used ReduceLROnPlateau(mode="min").
    scheduler_params: dict = {"mode": "min"}
    if lr_decay is not None:
        scheduler_params["factor"] = lr_decay
    if lr_patience is not None:
        scheduler_params["patience"] = lr_patience
    migrated["scheduler_type"] = "ReduceLROnPlateau"
    migrated["scheduler_params"] = scheduler_params

    # Preserve any unrecognised optimizer keys rather than silently dropping.
    migrated.update(optimizer)

    new["optimizer"] = migrated
    new["version"] = 1
    return new


# Ordered chain of converters: index i migrates version i -> i + 1.
_CONVERTERS = {0: _convert_v0_to_v1}


def upgrade_config(config: dict, from_version: int | None = None) -> dict:
    """Migrate ``config`` up to :data:`CONFIG_VERSION`, applying each step in turn.

    Returns a new dict; the input is not mutated.
    """
    version = detect_version(config) if from_version is None else from_version
    result = config
    while version < CONFIG_VERSION:
        converter = _CONVERTERS.get(version)
        if converter is None:
            raise ValueError(
                f"No migration path from config version {version} to {version + 1}.",
            )
        result = converter(result)
        version += 1
    return result


def _migration_help(config_path: str, version: int) -> str:
    """Build the actionable error message shown when a stale config is rejected."""
    stem, ext = os.path.splitext(config_path)
    suggested = f"{stem}.v{CONFIG_VERSION}{ext or '.yaml'}"
    return (
        f"Config '{config_path}' is version {version}, but this version of "
        f"gridfm-graphkit requires config version {CONFIG_VERSION}.\n"
        "The optimizer/scheduler schema changed (see PR #92).\n"
        "Choose how to migrate via --config_conversion:\n"
        f"  --config_conversion {CONVERSION_AUTO_INLINE}\n"
        "      Convert in memory and run now; the config file is left unchanged.\n"
        f"  --config_conversion {CONVERSION_AUTO_COPY} "
        "[--converted_config_path PATH]\n"
        "      Write an upgraded copy and run from it "
        f"(default PATH: '{suggested}').\n"
        "Or update the 'optimizer' block to the v"
        f"{CONFIG_VERSION} schema by hand and add 'version: {CONFIG_VERSION}'."
    )


def resolve_config(
    config: dict,
    config_path: str,
    conversion_mode: str = CONVERSION_NO,
    converted_config_path: str | None = None,
) -> dict:
    """Return a config at :data:`CONFIG_VERSION`, migrating per ``conversion_mode``.

    Args:
        config: the raw config dict loaded from YAML.
        config_path: path the config was loaded from (used for messages and to
            derive the default ``auto_copy`` output filename).
        conversion_mode: one of :data:`CONVERSION_MODES`.

            * ``"no"`` (default) — reject an out-of-date config with an
              actionable error explaining the options.
            * ``"auto_inline"`` — migrate in memory and proceed; the file on
              disk is not modified.
            * ``"auto_copy"`` — migrate, write the result to
              ``converted_config_path`` (or a derived default), and proceed.
        converted_config_path: destination for ``auto_copy``; if omitted a
            sibling ``<name>.v<CONFIG_VERSION>.yaml`` file is used.

    Raises:
        ValueError: if the config is newer than :data:`CONFIG_VERSION`, if it is
            out of date and ``conversion_mode`` is ``"no"``, or if
            ``conversion_mode`` is not a recognised value.
    """
    if conversion_mode not in CONVERSION_MODES:
        raise ValueError(
            f"Unknown config_conversion mode '{conversion_mode}'. "
            f"Must be one of {list(CONVERSION_MODES)}.",
        )

    version = detect_version(config)

    if version == CONFIG_VERSION:
        return config

    if version > CONFIG_VERSION:
        raise ValueError(
            f"Config '{config_path}' is version {version}, which is newer than "
            f"the version {CONFIG_VERSION} supported by this gridfm-graphkit "
            "install. Upgrade gridfm-graphkit to use this config.",
        )

    # version < CONFIG_VERSION -> migration required.
    if conversion_mode == CONVERSION_NO:
        raise ValueError(_migration_help(config_path, version))

    upgraded = upgrade_config(config, from_version=version)

    if conversion_mode == CONVERSION_AUTO_COPY:
        if converted_config_path is None:
            stem, ext = os.path.splitext(config_path)
            converted_config_path = f"{stem}.v{CONFIG_VERSION}{ext or '.yaml'}"
        with open(converted_config_path, "w") as f:
            yaml.safe_dump(upgraded, f, sort_keys=False, default_flow_style=None)
        print(
            f"[config] Converted v{version} config -> v{CONFIG_VERSION}, "
            f"written to '{converted_config_path}'.",
        )
    else:  # CONVERSION_AUTO_INLINE
        print(
            f"[config] Converted v{version} config -> v{CONFIG_VERSION} in memory "
            "(file on disk unchanged).",
        )

    return upgraded
