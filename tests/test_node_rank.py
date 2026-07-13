"""Unit tests for NODE_RANK computation from LSF metadata in set_env().

set_env() derives the distributed-training NODE_RANK by locating the current
host (HOSTNAME) inside the LSF host list (LSB_MCPU_HOSTS). LSF host lists and
HOSTNAME may disagree on FQDN vs short form, so the matching falls back from
full name to short name. These tests exercise that fallback without a cluster.
"""

import os

import pytest

from gridfm_graphkit.__main__ import set_env


@pytest.fixture(autouse=True)
def restore_distributed_env():
    """Save and restore distributed-training env vars set by set_env().

    set_env() writes NODE_RANK, MASTER_ADDR, MASTER_PORT, NCCL_SOCKET_IFNAME,
    and NCCL_IB_CUDA_SUPPORT directly into os.environ.  Without this fixture
    those values leak into subsequent test files (e.g. test_pipeline.py) and
    cause Lightning to block waiting for a non-existent distributed master.
    """
    keys = [
        "NODE_RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
        "NCCL_SOCKET_IFNAME",
        "NCCL_IB_CUDA_SUPPORT",
    ]
    original = {key: os.environ.get(key) for key in keys}
    yield
    for key, value in original.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _apply_lsf_env(monkeypatch, mcpu_hosts, hostname):
    """Set the minimal LSF env vars set_env() consumes, isolated per test."""
    monkeypatch.setenv("LSB_MCPU_HOSTS", mcpu_hosts)
    monkeypatch.setenv("LSB_JOBID", "123456")
    monkeypatch.setenv("HOSTNAME", hostname)
    # NODE_RANK must be computed fresh, not inherited from the environment.
    monkeypatch.delenv("NODE_RANK", raising=False)


def test_node_rank_exact_short_match(monkeypatch):
    """HOSTNAME and host list both short: index in the list wins."""
    _apply_lsf_env(monkeypatch, "node0 4 node1 4 node2 4", "node1")
    set_env()
    assert os_environ_node_rank() == "1"


def test_node_rank_fqdn_hostname_short_list(monkeypatch):
    """HOSTNAME is FQDN, list is short: falls back to short-name match."""
    _apply_lsf_env(
        monkeypatch,
        "p3-r06-n4 4 p1-r13-n2 4 p6-r24-n1 4",
        "p1-r13-n2.bluevela.rmf.ibm.com",
    )
    set_env()
    assert os_environ_node_rank() == "1"


def test_node_rank_short_hostname_fqdn_list(monkeypatch):
    """HOSTNAME is short, list is FQDN: short-vs-short comparison still matches."""
    _apply_lsf_env(
        monkeypatch,
        "a.example.com 4 b.example.com 4 c.example.com 4",
        "c",
    )
    set_env()
    assert os_environ_node_rank() == "2"


def test_node_rank_master_is_rank_zero(monkeypatch):
    """The first host in the list is the master and must get rank 0."""
    _apply_lsf_env(monkeypatch, "master 4 worker 4", "master.dc.ibm.com")
    set_env()
    assert os_environ_node_rank() == "0"


def test_node_rank_unresolvable_raises(monkeypatch):
    """A host absent from the list (even by short name) is a hard error."""
    _apply_lsf_env(monkeypatch, "node0 4 node1 4", "stranger.dc.ibm.com")
    with pytest.raises(RuntimeError, match="Unable to compute NODE_RANK"):
        set_env()


def os_environ_node_rank():
    import os

    return os.environ.get("NODE_RANK")
