"""Effective-resistance positional encoding used by HH-MPNN.

For every bus, the encoding is five statistics of its effective resistance to
all other buses, computed on the in-service bus graph weighted by branch
susceptance (Arowolo & Cremer, "Towards Generalization of Graph Neural Networks
for AC Optimal Power Flow", arXiv:2510.06860).

The encoding depends only on the in-service topology and the branch
susceptances, not on loads or generation. :class:`EffectiveResistancePE`
therefore caches it under a hash of exactly those inputs: a dataset with one
fixed grid costs a single computation, a dataset with topology perturbations
costs one computation per distinct topology.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections import OrderedDict

import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform

from gridfm_graphkit.datasets.globals import B_ON, TAP, YFT_TF_I

BRANCH = ("bus", "connects", "bus")

# Name of the bus attribute holding the encoding, and its width.
ER_PE_ATTR = "pestat_ER"
ER_PE_DIM = 5


def connected_components(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Label the connected components of an undirected graph.

    Args:
        edge_index: ``[2, E]`` edges; direction is ignored.
        num_nodes: Number of nodes.

    Returns:
        ``[num_nodes]`` long tensor. Two nodes share a label if and only if
        they are connected; the label is the smallest node id of the component.
    """
    labels = torch.arange(num_nodes, device=edge_index.device)
    row, col = edge_index
    while True:
        # Each sweep gives every node the smallest label among its neighbours.
        new = labels.scatter_reduce(0, col, labels[row], reduce="amin")
        new = new.scatter_reduce(0, row, new[col], reduce="amin")
        if torch.equal(new, labels):
            return labels
        labels = new


@torch.no_grad()
def effective_resistance_pe(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Row statistics of the effective-resistance matrix of a weighted graph.

    The effective resistance between nodes ``i`` and ``j`` is
    ``R_ij = L+_ii + L+_jj - 2 L+_ij``, with ``L+`` the pseudoinverse of the
    weighted graph Laplacian. Edges with non-positive weight are ignored and
    parallel edges add up.

    Statistics are taken over the other nodes of the same connected component,
    because the effective resistance between disconnected nodes is infinite. A
    node that is alone in its component gets zeros.

    Cost is cubic in the size of the largest component and quadratic in memory.

    Args:
        edge_index: ``[2, E]`` edges. Every branch must appear in both
            directions (the graphkit convention); the two directions are
            averaged into one symmetric weight.
        edge_weight: ``[E]`` edge weights.
        num_nodes: Number of nodes.

    Returns:
        ``[num_nodes, 5]`` float tensor with columns
        ``[mean, median, std, max, min]`` (population standard deviation).
    """
    keep = edge_weight > 0
    edge_index = edge_index[:, keep]
    weight = edge_weight[keep].double()

    adjacency = torch.zeros(num_nodes, num_nodes, dtype=torch.double)
    adjacency.index_put_((edge_index[0], edge_index[1]), weight, accumulate=True)
    adjacency = (adjacency + adjacency.T) / 2
    laplacian = torch.diag(adjacency.sum(dim=1)) - adjacency

    labels = connected_components(edge_index, num_nodes)
    pe = torch.zeros(num_nodes, ER_PE_DIM, dtype=torch.double)
    for label in labels.unique():
        nodes = (labels == label).nonzero().squeeze(1)
        size = nodes.numel()
        if size == 1:
            continue
        # For a connected graph, L+ = (L + 11^T / n)^-1 - 11^T / n.
        block = laplacian[nodes][:, nodes]
        pinv = torch.linalg.inv(block + 1.0 / size) - 1.0 / size
        diag = pinv.diagonal()
        resistance = diag[:, None] + diag[None, :] - 2 * pinv

        off_diagonal = ~torch.eye(size, dtype=torch.bool)
        others = resistance[off_diagonal].view(size, size - 1).sort(dim=1).values
        # Mean of the two middle values, so even counts match numpy.median.
        median = (others[:, (size - 2) // 2] + others[:, (size - 1) // 2]) / 2
        pe[nodes] = torch.stack(
            [
                others.mean(dim=1),
                median,
                others.std(dim=1, correction=0),
                others[:, -1],
                others[:, 0],
            ],
            dim=1,
        )
    return pe.float()


class EffectiveResistancePE(BaseTransform):
    """Attach the effective-resistance positional encoding to the bus nodes.

    The result is stored as ``data["bus"].pestat_ER`` with shape
    ``[num_bus, 5]``. Branch weights are the series susceptance
    ``|Im(Yft)| * tap``, which equals ``x / (r^2 + x^2)`` for branches without
    phase shift. They are read from the edge attributes as they are at this
    point of the pipeline, so the encoding is on the same power base as the
    admittance features the model sees.

    Out-of-service branches are excluded whether this transform runs before or
    after ``RemoveInactiveBranches``.

    Results are cached in memory per process, keyed by a hash of the bus count,
    the in-service edges and their weights. With ``cache_dir`` they are also
    written to disk, so other dataloader workers, later epochs and later runs
    reuse them.

    Args:
        cache_dir: Optional directory for the on-disk cache.
        memory_cache_size: Maximum number of encodings kept in memory per
            process; the least recently used one is evicted first.

    Raises:
        ValueError: If an unknown keyword argument is passed.
    """

    def __init__(
        self,
        *,
        cache_dir: str | None = None,
        memory_cache_size: int = 1024,
        **unknown,
    ):
        super().__init__()
        if unknown:
            # Not a TypeError: the config-driven builder would swallow that and
            # silently fall back to the defaults.
            raise ValueError(
                f"EffectiveResistancePE got unknown arguments {sorted(unknown)}; "
                "accepted: cache_dir, memory_cache_size.",
            )
        self.cache_dir = cache_dir
        self.memory_cache_size = memory_cache_size
        self._memory_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)

    @staticmethod
    def _in_service_graph(data: HeteroData) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the edges and susceptance weights of the in-service branches."""
        edge_index = data[BRANCH].edge_index
        edge_attr = data[BRANCH].edge_attr
        tap = edge_attr[:, TAP]
        tap = torch.where(tap == 0, torch.ones_like(tap), tap)
        weight = edge_attr[:, YFT_TF_I].abs() * tap
        # The status column is still present before RemoveInactiveBranches.
        if edge_attr.size(1) > B_ON:
            weight = weight * (edge_attr[:, B_ON] == 1)
        keep = weight > 0
        return edge_index[:, keep], weight[keep]

    @staticmethod
    def _fingerprint(
        edge_index: torch.Tensor,
        weight: torch.Tensor,
        num_nodes: int,
    ) -> str:
        """Hash everything the encoding depends on."""
        digest = hashlib.sha256()
        digest.update(int(num_nodes).to_bytes(8, "little"))
        digest.update(edge_index.cpu().numpy().tobytes())
        # Quantise so float noise does not cause spurious cache misses.
        quantised = (weight.double() * 1e6).round().to(torch.int64)
        digest.update(quantised.cpu().numpy().tobytes())
        return digest.hexdigest()[:32]

    def _disk_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"pe_cache_{key}.pt")

    def _save_to_disk(self, key: str, pe: torch.Tensor) -> None:
        """Write atomically so concurrent workers never read a partial file."""
        fd, tmp_path = tempfile.mkstemp(
            dir=self.cache_dir,
            prefix=f".pe_cache_tmp_{os.getpid()}_",
            suffix=".pt",
        )
        try:
            os.close(fd)
            torch.save(pe, tmp_path)
            os.replace(tmp_path, self._disk_path(key))
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def forward(self, data: HeteroData) -> HeteroData:
        num_nodes = data["bus"].num_nodes
        edge_index, weight = self._in_service_graph(data)
        key = self._fingerprint(edge_index, weight, num_nodes)

        pe = self._memory_cache.get(key)
        if pe is None:
            if self.cache_dir is not None and os.path.exists(self._disk_path(key)):
                pe = torch.load(self._disk_path(key), weights_only=True)
            else:
                pe = effective_resistance_pe(edge_index, weight, num_nodes)
                if self.cache_dir is not None:
                    self._save_to_disk(key, pe)
            self._memory_cache[key] = pe
            if len(self._memory_cache) > self.memory_cache_size:
                self._memory_cache.popitem(last=False)
        else:
            self._memory_cache.move_to_end(key)

        # Clone so in-place changes to one sample cannot corrupt the cache.
        data["bus"][ER_PE_ATTR] = pe.clone()
        return data
