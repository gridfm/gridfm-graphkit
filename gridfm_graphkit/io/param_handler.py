import torch
from gridfm_graphkit.training.loss import MixedLoss
from gridfm_graphkit.io.registries import (
    NORMALIZERS_REGISTRY,
    MODELS_REGISTRY,
    LOSS_REGISTRY,
    TRANSFORM_REGISTRY,
    TASK_REGISTRY,
    PHYSICS_DECODER_REGISTRY,
)

import argparse
from torch_geometric.transforms import Compose
from gridfm_graphkit.tasks.base_task import BaseTask


class NestedNamespace(argparse.Namespace):
    """
    A namespace object that supports nested structures, allowing for
    easy access and manipulation of hierarchical configurations.

    """

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                # Recursively convert dictionaries to NestedNamespace
                setattr(self, key, NestedNamespace(**value))
            elif isinstance(value, list):
                list_of_namespaces = []
                for element in value:
                    if isinstance(element, dict):
                        list_of_namespaces.append(NestedNamespace(**element))
                    else:
                        list_of_namespaces.append(element)
                setattr(self, key, list_of_namespaces)
            else:
                setattr(self, key, value)

    def to_dict(self):
        # Recursively convert NestedNamespace back to dictionary
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, NestedNamespace):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def flatten(self, parent_key="", sep="."):
        # Flatten the dictionary with dot-separated keys
        items = []
        for key, value in self.__dict__.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            if isinstance(value, NestedNamespace):
                items.extend(value.flatten(new_key, sep=sep).items())
            else:
                items.append((new_key, value))
        return dict(items)


def load_normalizer(args):
    """
    Load the appropriate data normalization methods

    Args:
        args (NestedNamespace): contains configs.

    Returns:
        tuple: Node and edge normalizers

    Raises:
        ValueError: If an unknown normalization method is specified.
    """
    method = args.data.normalization

    try:
        return NORMALIZERS_REGISTRY.create(
            method,
            args,
        )
    except KeyError:
        raise ValueError(f"Unknown transformation: {method}")


def get_loss_function(args):
    """
    Load the appropriate loss function

    Args:
        args (NestedNamespace): contains configs.

    Returns:
        nn.Module: Loss function

    Raises:
        ValueError: If an unknown loss function is specified.
    """
    loss_functions = []
    for loss_name, loss_args in zip(args.training.losses, args.training.loss_args):
        try:
            loss_functions.append(LOSS_REGISTRY.create(loss_name, loss_args, args))
        except KeyError:
            raise ValueError(f"Unknown loss: {loss_name}")

    warmup_epochs = getattr(args.training, "physics_warmup_epochs", 0)
    warmup_indices = [
        i
        for i, loss_name in enumerate(args.training.losses)
        if loss_name == "LayeredWeightedPhysics"
    ]

    return MixedLoss(
        loss_functions=loss_functions,
        weights=args.training.loss_weights,
        warmup_indices=warmup_indices,
        warmup_epochs=warmup_epochs,
    )


def load_model(args) -> torch.nn.Module:
    """
    Load the appropriate model

    Args:
        args (NestedNamespace): contains configs.

    Returns:
        nn.Module: The selected model initialized with the provided configurations.

    Raises:
        ValueError: If an unknown model type is specified.
    """
    model_type = args.model.type

    try:
        return MODELS_REGISTRY.create(model_type, args)
    except KeyError:
        raise ValueError(f"Unknown model type: {model_type}")


def _build_transform(name, args, entry_kwargs):
    """Instantiate one registered transform for the config-declared list path.

    Transform constructors are not uniform: some take ``(args)``, some take
    specific keyword arguments (e.g. ``mask_ratio``), and some take no
    arguments. To let a YAML list mix all of them, resolve the registered
    class/factory and call it with the convention:

      1. ``cls(args, **entry_kwargs)`` — transforms that read the config namespace
      2. ``cls(**entry_kwargs)``       — transforms configured purely by kwargs
      3. ``cls(args)`` / ``cls()``     — fallbacks for the two simple shapes

    ``entry_kwargs`` are the extra keys on a list entry, e.g.
    ``{name: AddPFHeteroMask, mask_ref_vm: true}`` -> ``{"mask_ref_vm": True}``.

    .. note::
        The four-way call convention is tolerant but silent: an unexpected or
        misspelled kwarg falls through to the next attempt rather than being
        reported. A future enhancement could introspect each constructor via
        ``inspect.signature`` and raise a precise error listing the accepted
        kwargs (e.g. "AddPFHeteroMask accepts: mask_ref_vm"). That is worth doing
        only if it stays backward-compatible with every existing config — the
        transforms that take the whole ``args`` namespace must keep working — so
        it is deferred rather than done here.
    """
    cls = TRANSFORM_REGISTRY.get(name)
    attempts = (
        lambda: cls(args, **entry_kwargs),
        lambda: cls(**entry_kwargs),
        lambda: cls(args),
        lambda: cls(),
    )
    last_err = None
    for attempt in attempts:
        try:
            return attempt()
        except TypeError as e:  # signature mismatch -> try the next convention
            last_err = e
    raise TypeError(
        f"Could not instantiate transform '{name}' with any supported "
        f"constructor convention: {last_err}",
    )


def get_task_transforms(args) -> Compose:
    """
    Load the task transforms.

    Two modes, checked in order:

    * **Config-declared override.** If ``data.transforms`` is present it is a
      list of transforms to compose, in order. Each entry is either a bare
      registry name (``"RemoveInactiveBranches"``) or a mapping with a ``name``
      key plus per-transform kwargs (``{name: AddPFHeteroMask, mask_ref_vm: true}``).
      This lets a config assemble an arbitrary pipeline from any registered
      transform — built-in or plugin-provided — without touching code.
    * **Default (task-name) composition.** If ``data.transforms`` is absent,
      fall back to the historical behaviour: look up the single composition
      registered under ``task.task_name`` (``PowerFlow`` / ``OptimalPowerFlow`` /
      ``StateEstimation`` / any plugin-registered name). This keeps every
      existing config working unchanged.

    .. warning::
        The override only changes *how the transform pipeline is assembled* — it
        does **not** decouple the task from the data schema. The task module
        selected by ``task.task_name`` still assumes the batch carries whatever
        its default composition would have produced (e.g. OPF/PF expect the mask
        fields added by ``Add*HeteroMask`` + ``ApplyMasking``). If your list
        omits a step the task relies on, data will load fine but the task will
        fail downstream. A safe override reproduces the default pipeline and
        adds/swaps steps around it, rather than dropping the ones the task needs.
    """
    declared = getattr(args.data, "transforms", None)

    if declared:
        transforms = []
        for entry in declared:
            if isinstance(entry, str):
                name, entry_kwargs = entry, {}
            elif isinstance(entry, NestedNamespace):
                entry_dict = entry.to_dict()
                name = entry_dict.pop("name")
                entry_kwargs = entry_dict
            elif isinstance(entry, dict):
                entry_kwargs = dict(entry)
                name = entry_kwargs.pop("name")
            else:
                raise ValueError(
                    f"Invalid data.transforms entry {entry!r}: expected a name "
                    "string or a mapping with a 'name' key.",
                )
            try:
                transforms.append(_build_transform(name, args, entry_kwargs))
            except KeyError:
                raise ValueError(
                    f"Unknown transform '{name}' in data.transforms. Register it "
                    "(built-in or via a plugin) before referencing it.",
                )
        return Compose(transforms)

    task_transforms = args.task.task_name
    try:
        return TRANSFORM_REGISTRY.create(task_transforms, args)
    except KeyError:
        raise ValueError(f"Unknown task: {task_transforms}")


def get_task(args, data_normalizers) -> BaseTask:
    """
    Load the task module
    """
    task = args.task.task_name

    try:
        return TASK_REGISTRY.create(task, args, data_normalizers)
    except KeyError:
        raise ValueError(f"Unknown task: {task}")


def get_physics_decoder(args) -> torch.nn.Module:
    """
    Load the task module
    """
    task = args.task.task_name

    try:
        return PHYSICS_DECODER_REGISTRY.create(task)
    except KeyError:
        raise ValueError(f"No physics decoder associate to {task} task")
