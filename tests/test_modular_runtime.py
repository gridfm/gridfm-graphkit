"""Unit tests for the two additive modularity seams added in PR #149:

* ``get_task_transforms`` / ``_build_transform`` — the config-declared
  ``data.transforms`` override (and its fallback to the task-name composition).
* ``_load_entrypoint_plugins`` — entry-point auto-discovery of plugins.

The tests register throwaway transforms into the real ``TRANSFORM_REGISTRY``
and remove them afterwards, so they exercise the production code paths without
touching on-disk data fixtures.
"""

import warnings
from unittest import mock

import pytest
from torch_geometric.transforms import Compose

from gridfm_graphkit.cli import _load_entrypoint_plugins
from gridfm_graphkit.io.param_handler import (
    NestedNamespace,
    _build_transform,
    get_task_transforms,
)
from gridfm_graphkit.io.registries import TRANSFORM_REGISTRY


# --------------------------------------------------------------------------- #
# Fixtures: throwaway transforms covering the three constructor shapes that
# `_build_transform` must bridge.
# --------------------------------------------------------------------------- #
class _ArgsTransform:
    """Transform whose constructor takes the config namespace: ``cls(args)``."""

    def __init__(self, args):
        self.args = args

    def __call__(self, data):
        return data


class _KwargsTransform:
    """Transform configured purely by keyword args: ``cls(**kwargs)``."""

    def __init__(self, flag=False):
        self.flag = flag

    def __call__(self, data):
        return data


class _NoArgTransform:
    """Transform taking nothing: ``cls()``."""

    def __init__(self):
        self.built = True

    def __call__(self, data):
        return data


@pytest.fixture
def registered_transforms():
    """Register the throwaway transforms by name; unregister on teardown."""
    names = {
        "ArgsTransform": _ArgsTransform,
        "KwargsTransform": _KwargsTransform,
        "NoArgTransform": _NoArgTransform,
    }
    for name, cls in names.items():
        TRANSFORM_REGISTRY.register(name)(cls)
    try:
        yield names
    finally:
        for name in names:
            TRANSFORM_REGISTRY._registry.pop(name, None)


def _args_with_transforms(transforms):
    """Build a NestedNamespace with the given ``data.transforms`` value."""
    return NestedNamespace(data={"transforms": transforms}, task={})


# --------------------------------------------------------------------------- #
# _build_transform: the (args,**kw) -> (**kw) -> (args) -> () fallback.
# --------------------------------------------------------------------------- #
def test_build_transform_args_shape(registered_transforms):
    args = NestedNamespace(data={})
    built = _build_transform("ArgsTransform", args, {})
    assert isinstance(built, _ArgsTransform)
    assert built.args is args


def test_build_transform_kwargs_shape(registered_transforms):
    # A kwargs-only constructor: the (args, **kw) attempt raises TypeError
    # (unexpected positional), so the fallback must reach (**kw).
    built = _build_transform("KwargsTransform", object(), {"flag": True})
    assert isinstance(built, _KwargsTransform)
    assert built.flag is True


def test_build_transform_no_arg_shape(registered_transforms):
    built = _build_transform("NoArgTransform", object(), {})
    assert isinstance(built, _NoArgTransform)
    assert built.built is True


def test_build_transform_unknown_name_raises_keyerror(registered_transforms):
    # _build_transform resolves via TRANSFORM_REGISTRY.get, which raises KeyError
    # for an unknown name (get_task_transforms turns that into a ValueError).
    with pytest.raises(KeyError):
        _build_transform("DoesNotExist", object(), {})


# --------------------------------------------------------------------------- #
# get_task_transforms: the declared-override path, entry forms, errors.
# --------------------------------------------------------------------------- #
def test_declared_list_bare_string_entries(registered_transforms):
    args = _args_with_transforms(["NoArgTransform", "NoArgTransform"])
    composed = get_task_transforms(args)
    assert isinstance(composed, Compose)
    assert len(composed.transforms) == 2
    assert all(isinstance(t, _NoArgTransform) for t in composed.transforms)


def test_declared_list_raw_dict_entry_passes_kwargs(registered_transforms):
    # The raw-dict branch. NestedNamespace converts dict list-entries to
    # NestedNamespace when built from YAML, so to hit the `isinstance(entry, dict)`
    # branch we set the list to hold a plain dict after construction.
    args = NestedNamespace(data={}, task={})
    args.data.transforms = [{"name": "KwargsTransform", "flag": True}]
    composed = get_task_transforms(args)
    assert len(composed.transforms) == 1
    assert isinstance(composed.transforms[0], _KwargsTransform)
    assert composed.transforms[0].flag is True


def test_declared_list_nested_namespace_entry(registered_transforms):
    # The YAML loader wraps nested mapping entries as NestedNamespace, not dict —
    # get_task_transforms must handle that form too.
    entry = NestedNamespace(name="KwargsTransform", flag=True)
    args = NestedNamespace(data=NestedNamespace(transforms=[entry]), task={})
    composed = get_task_transforms(args)
    assert len(composed.transforms) == 1
    assert composed.transforms[0].flag is True


def test_declared_list_mixed_entry_forms_preserve_order(registered_transforms):
    args = _args_with_transforms(
        [
            "ArgsTransform",
            {"name": "KwargsTransform", "flag": True},
            "NoArgTransform",
        ],
    )
    composed = get_task_transforms(args)
    kinds = [type(t) for t in composed.transforms]
    assert kinds == [_ArgsTransform, _KwargsTransform, _NoArgTransform]


def test_declared_list_unknown_transform_raises_valueerror(registered_transforms):
    args = _args_with_transforms(["NoArgTransform", "NotRegistered"])
    with pytest.raises(ValueError, match="Unknown transform 'NotRegistered'"):
        get_task_transforms(args)


def test_declared_list_invalid_entry_type_raises_valueerror(registered_transforms):
    args = _args_with_transforms([123])
    with pytest.raises(ValueError, match="Invalid data.transforms entry"):
        get_task_transforms(args)


def test_absent_transforms_falls_back_to_task_name(registered_transforms):
    # Regression: with no data.transforms, the single composition registered
    # under task.task_name is used (historical behaviour).
    sentinel = Compose([])
    args = NestedNamespace(data={}, task={"task_name": "PowerFlow"})
    with mock.patch.object(
        TRANSFORM_REGISTRY,
        "create",
        return_value=sentinel,
    ) as create:
        result = get_task_transforms(args)
    create.assert_called_once_with("PowerFlow", args)
    assert result is sentinel


def test_empty_transforms_list_falls_back_to_task_name(registered_transforms):
    # An empty list is falsy -> same fallback as absent.
    args = NestedNamespace(data={"transforms": []}, task={"task_name": "PowerFlow"})
    with mock.patch.object(TRANSFORM_REGISTRY, "create") as create:
        get_task_transforms(args)
    create.assert_called_once_with("PowerFlow", args)


# --------------------------------------------------------------------------- #
# _load_entrypoint_plugins: callable vs module load, broken-plugin isolation.
# --------------------------------------------------------------------------- #
class _FakeEntryPoint:
    def __init__(self, name, loaded):
        self.name = name
        self._loaded = loaded

    def load(self):
        return self._loaded


def test_entrypoint_callable_is_invoked():
    called = {"n": 0}

    def register():
        called["n"] += 1

    ep = _FakeEntryPoint("good", register)
    with mock.patch(
        "importlib.metadata.entry_points",
        return_value=[ep],
    ):
        _load_entrypoint_plugins()
    assert called["n"] == 1


def test_entrypoint_module_is_not_called():
    # A module target is non-callable: load() importing it is enough; the loader
    # must not try to call it.
    module = mock.MagicMock(spec=[])  # spec=[] -> not callable
    ep = _FakeEntryPoint("mod", module)
    with mock.patch("importlib.metadata.entry_points", return_value=[ep]):
        _load_entrypoint_plugins()  # must not raise


def test_broken_plugin_warns_but_does_not_abort():
    def boom():
        raise RuntimeError("plugin is broken")

    good_called = {"n": 0}

    def good():
        good_called["n"] += 1

    broken_ep = _FakeEntryPoint("broken", boom)
    good_ep = _FakeEntryPoint("good", good)
    with mock.patch(
        "importlib.metadata.entry_points",
        return_value=[broken_ep, good_ep],
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _load_entrypoint_plugins()  # must not raise despite the broken plugin
    # The broken plugin warned...
    assert any("broken" in str(w.message) for w in caught)
    # ...and the following good plugin still ran (one bad plugin can't abort the CLI).
    assert good_called["n"] == 1


def test_no_entrypoints_is_a_noop():
    with mock.patch("importlib.metadata.entry_points", return_value=[]):
        _load_entrypoint_plugins()  # must not raise
