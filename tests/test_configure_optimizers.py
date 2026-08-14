import pytest
import torch
from unittest.mock import MagicMock

from gridfm_graphkit.io.param_handler import NestedNamespace
from gridfm_graphkit.tasks.base_task import BaseTask


def make_task(optimizer_config: dict, callbacks_config: dict = {}):
    """Build a minimal BaseTask instance with a fake model and given optimizer config.

    Bypasses __init__ to avoid requiring a full args/normalizer setup — we only
    need self.args and self.model to exercise configure_optimizers().
    """
    args = NestedNamespace(
        optimizer=optimizer_config,
        callbacks=callbacks_config,
    )
    task = BaseTask.__new__(BaseTask)
    task.args = args
    # Fake model with a single parameter so the optimizer has something to hold
    task.model = MagicMock()
    task.model.parameters.return_value = iter(
        [torch.nn.Parameter(torch.zeros(1))],
    )
    return task


def test_default_optimizer_when_type_omitted():
    """Omitting 'type' should silently default to AdamW without raising."""
    task = make_task({"learning_rate": 0.001})
    result = task.configure_optimizers()
    assert isinstance(result["optimizer"], torch.optim.AdamW)


def test_explicit_adamw_type():
    """Explicitly setting type: AdamW should produce an AdamW optimizer."""
    task = make_task({"type": "AdamW", "learning_rate": 0.001})
    result = task.configure_optimizers()
    assert isinstance(result["optimizer"], torch.optim.AdamW)


def test_custom_optimizer_type():
    """Setting type: SGD should produce an SGD optimizer."""
    task = make_task({"type": "SGD", "learning_rate": 0.01})
    result = task.configure_optimizers()
    assert isinstance(result["optimizer"], torch.optim.SGD)


def test_optimizer_params_applied():
    """optimizer_params should be unpacked and passed to the optimizer."""
    task = make_task(
        {
            "type": "AdamW",
            "learning_rate": 0.001,
            "optimizer_params": {"weight_decay": 0.05},
        },
    )
    result = task.configure_optimizers()
    assert result["optimizer"].param_groups[0]["weight_decay"] == pytest.approx(0.05)


def test_default_optimizer_params_when_omitted():
    """Omitting optimizer_params should not raise — defaults to empty dict."""
    task = make_task({"learning_rate": 0.001})
    result = task.configure_optimizers()
    assert "optimizer" in result


def test_no_scheduler_when_omitted():
    """Omitting scheduler_type should return optimizer only with no lr_scheduler key."""
    task = make_task({"learning_rate": 0.001})
    result = task.configure_optimizers()
    assert "lr_scheduler" not in result


def test_scheduler_configured_when_present():
    """Specifying scheduler_type should return both optimizer and lr_scheduler."""
    task = make_task(
        {
            "learning_rate": 0.001,
            "scheduler_type": "ReduceLROnPlateau",
            "scheduler_params": {"mode": "min", "factor": 0.7, "patience": 5},
        },
    )
    result = task.configure_optimizers()
    assert "lr_scheduler" in result
    assert isinstance(
        result["lr_scheduler"]["scheduler"],
        torch.optim.lr_scheduler.ReduceLROnPlateau,
    )


def test_invalid_optimizer_type_raises_value_error():
    """An unrecognised optimizer type should raise ValueError with a clear message."""
    task = make_task({"type": "NonExistentOptimizer", "learning_rate": 0.001})
    with pytest.raises(ValueError, match="Unknown optimizer type"):
        task.configure_optimizers()


def test_invalid_scheduler_type_raises_value_error():
    """An unrecognised scheduler type should raise ValueError with a clear message."""
    task = make_task(
        {
            "learning_rate": 0.001,
            "scheduler_type": "NonExistentScheduler",
        },
    )
    with pytest.raises(ValueError, match="Unknown scheduler type"):
        task.configure_optimizers()
