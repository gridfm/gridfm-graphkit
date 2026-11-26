import os

import pytest

from gridfm_graphkit import __main__ as cli_main


@pytest.fixture
def no_ibv_devinfo(monkeypatch):
    """Simulate a host (or container) where the ibv_devinfo binary is absent."""

    def _missing(*args, **kwargs):
        raise FileNotFoundError("ibv_devinfo")

    monkeypatch.setattr(cli_main.subprocess, "run", _missing)


def test_fix_infiniband_warns_when_ibv_devinfo_missing(
    monkeypatch,
    no_ibv_devinfo,
) -> None:
    monkeypatch.delenv("NCCL_IB_HCA", raising=False)

    with pytest.warns(UserWarning, match="ibv_devinfo not found"):
        cli_main.fix_infiniband()

    assert "NCCL_IB_HCA" not in os.environ


def test_fix_infiniband_keeps_existing_nccl_ib_hca_when_ibv_devinfo_missing(
    monkeypatch,
    no_ibv_devinfo,
) -> None:
    monkeypatch.setenv("NCCL_IB_HCA", "mlx5_0")

    with pytest.warns(UserWarning, match="ibv_devinfo not found"):
        cli_main.fix_infiniband()

    assert os.environ["NCCL_IB_HCA"] == "mlx5_0"
