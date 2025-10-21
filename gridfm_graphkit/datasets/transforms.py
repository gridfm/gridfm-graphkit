from gridfm_graphkit.datasets.globals import PQ, PV, REF, PG, QG, VM, VA, G, B
from gridfm_graphkit.io.registries import MASKING_REGISTRY

import torch
from torch import Tensor
from torch_geometric.transforms import BaseTransform
from typing import Optional, Any
import torch_geometric.typing
from torch_geometric.data import Data
from torch_geometric.utils import (
    get_self_loop_attr,
    is_torch_sparse_tensor,
    to_edge_index,
    to_torch_coo_tensor,
    to_torch_csr_tensor,
)

import numpy as np
import os
import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()})
import gridfm_graphkit.models.algos as algos


class AddNormalizedRandomWalkPE(BaseTransform):
    r"""Adds the random walk positional encoding from the
    [Graph Neural Networks with Learnable Structural and Positional Representations](https://arxiv.org/abs/2110.07875)
    paper to the given graph. This is an adaptation from the original Pytorch Geometric implementation.

    Args:
        walk_length (int): The number of random walk steps.
        attr_name (str, optional): The attribute name of the data object to add
            positional encodings to. If set to :obj:`None`, will be
            concatenated to :obj:`data.x`.
            (default: :obj:`"random_walk_pe"`)
    """

    def __init__(
        self,
        walk_length: int,
        attr_name: Optional[str] = "random_walk_pe",
    ) -> None:
        self.walk_length = walk_length
        self.attr_name = attr_name

    def forward(self, data: Data) -> Data:
        if data.edge_index is None:
            raise ValueError("Expected data.edge_index to be not None")
        row, col = data.edge_index
        N = data.num_nodes
        if N is None:
            raise ValueError("Expected data.num_nodes to be not None")

        if N <= 2_000:  # Dense code path for faster computation:
            adj = torch.zeros((N, N), device=row.device)
            adj[row, col] = data.edge_weight
            loop_index = torch.arange(N, device=row.device)
        elif torch_geometric.typing.WITH_WINDOWS:
            adj = to_torch_coo_tensor(
                data.edge_index,
                data.edge_weight,
                size=data.size(),
            )
        else:
            adj = to_torch_csr_tensor(
                data.edge_index,
                data.edge_weight,
                size=data.size(),
            )

        row_sums = adj.sum(dim=1, keepdim=True)  # Sum along rows
        row_sums = row_sums.clamp(min=1e-8)  # Prevent division by zero

        adj = adj / row_sums  # Normalize each row to sum to 1

        def get_pe(out: Tensor) -> Tensor:
            if is_torch_sparse_tensor(out):
                return get_self_loop_attr(*to_edge_index(out), num_nodes=N)
            return out[loop_index, loop_index]

        out = adj
        pe_list = [get_pe(out)]
        for _ in range(self.walk_length - 1):
            out = out @ adj
            pe_list.append(get_pe(out))

        pe = torch.stack(pe_list, dim=-1)
        data[self.attr_name] = pe

        return data


def add_node_attr(data: Data, 
                    value: Any,
                    attr_name: str
                    ) -> Data:
    if attr_name is None:
        raise ValueError("Expected attr_name to be not None")
    else:
        data[attr_name] = value

    return data

def convert_to_single_emb(x, offset=512):
    """
    The edge feature embedding range is set to start at 512 to accomodate
    negative branch feature values in PF data.
    """
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + \
        torch.arange(offset, (feature_num + 1) * offset, offset, dtype=torch.long)
    x = x + feature_offset
    
    return x

def get_edge_encoding(edge_attr, N, edge_index, max_dist, path):
    if len(edge_attr.size()) == 1:
            edge_attr = edge_attr[:, None]
    attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
    attn_edge_type[edge_index[0, :], edge_index[1, :]
                    ] = convert_to_single_emb(edge_attr.long()) + 1
    if os.name == 'nt':
        edge_input = algos.gen_edge_input(
                    max_dist, 
                    path, 
                    attn_edge_type.numpy(),
                    localtype=np.int32
                    )
    else:
        edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
        
    return attn_edge_type, torch.from_numpy(edge_input).long()

def preprocess_item(data, max_hops):
    """
    Calculation of the Graphormer attention bias, and positional/structural 
    variables. From a Data-like object the shortest paths in number of hops 
    between nodes are calculated, being cut off at max_hops. In addition to the
    centrality (assume undirected graphs) and attention bias, these are the 
    inputs to the model structural and positional encodings.
    """
    edge_index = data.edge_index
    edge_attr = data.edge_attr
    N = data.num_nodes
    edge_adj = torch.sparse_coo_tensor(
                                    edge_index,
                                    torch.ones(edge_index.shape[1]).to(data.x.device),
                                    [N, N]
                                    )

    adj = edge_adj.to_dense().to(torch.int16)

    # get shortest paths in number of hops (shortest_path_result) and intermediate nodes
    # for those shortest paths (path)
    if os.name == 'nt':
        shortest_path_result, path = algos.floyd_warshall(
                        adj.numpy().astype(np.int32), 
                        max_hops, 
                        localtype=np.int32
                        )
    else:
        shortest_path_result, path = algos.floyd_warshall(
                        adj.numpy().astype(np.int32), 
                        max_hops
                        )

    spatial_pos = torch.from_numpy((shortest_path_result)).long().to(data.x.device)
    attn_bias = torch.zeros([N, N], dtype=torch.float).to(data.x.device) 

    if edge_attr is not None:
        attn_edge_type, edge_input = get_edge_encoding(edge_attr, N, edge_index, max_hops, path)
    else:
        edge_input = None
        attn_edge_type = None

    in_degree = adj.long().sum(dim=1).view(-1)
    out_degree = adj.long().sum(dim=0).view(-1)
    return attn_bias, spatial_pos, in_degree, out_degree, attn_edge_type, edge_input

def pad_1d_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)

def pad_2d_unsqueeze(x, padlen):
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:,:] = -1e9
        new_x[:xlen, :] = x 
        x = new_x
    return x.unsqueeze(0)

def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros(
            [padlen, padlen], dtype=x.dtype).fill_(float('-inf'))   
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0     
        x = new_x
    return x.unsqueeze(0)

def pad_edge_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros(
            (padlen, padlen) + x.size()[2:], dtype=x.dtype).fill_(int(0))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)

def pad_spatial_pos_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)


class AddGraphormerEncodings(BaseTransform):
    """Adds a positional encoding (node centrallity) to the given graph, as 
    well as the attention and edge biases, as described in: Do transformers 
    really perform badly for graph representation?, C. Ying et al., 2021.
    
    Args:
        max_node_num (int): The number of nodes in the largest graph considered.
        max_hops (int): The maximum path length between nodes to consider for
                        the edge encodings.
    """

    def __init__(
        self,
        max_node_num: int,
        max_hops: int,
    ) -> None:
        self.max_node_num = max_node_num
        self.max_hops = max_hops

    def forward(self, data: Data) -> Data:
        if data.edge_index is None:
            raise ValueError("Expected data.edge_index to be not None")

        N = data.num_nodes
        if N is None:
            raise ValueError("Expected data.num_nodes to be not None")

        attn_bias, spatial_pos, in_degree, out_degree, attn_edge_type, edge_input = \
                            preprocess_item(data, self.max_hops)
        
        attn_bias = pad_attn_bias_unsqueeze(attn_bias, self.max_node_num)
        spatial_pos = pad_spatial_pos_unsqueeze(spatial_pos, self.max_node_num)
        in_degree = pad_1d_unsqueeze(in_degree, self.max_node_num).squeeze()
        edge_input = pad_edge_bias_unsqueeze(edge_input, self.max_node_num)
        attn_edge_type = pad_edge_bias_unsqueeze(attn_edge_type, self.max_node_num)

        data = add_node_attr(data, attn_bias, attr_name='attn_bias')
        data = add_node_attr(data, spatial_pos, attr_name='spatial_pos')
        data = add_node_attr(data, in_degree, attr_name='in_degree')
        data = add_node_attr(data, edge_input, attr_name='edge_input')
        data = add_node_attr(data, attn_edge_type, attr_name='attn_edge_type')

        data.x = pad_2d_unsqueeze(data.x, self.max_node_num).squeeze()
        data.y = pad_2d_unsqueeze(data.y, self.max_node_num).squeeze()

        return data


class AddEdgeWeights(BaseTransform):
    """
    Computes and adds edge weight as the magnitude of complex admittance.

    The magnitude is computed from the G and B components in `data.edge_attr` and stored in `data.edge_weight`.
    """

    def forward(self, data):
        if not hasattr(data, "edge_attr"):
            raise AttributeError("Data must have 'edge_attr'.")

        # Extract real and imaginary parts of admittance
        real = data.edge_attr[:, G]
        imag = data.edge_attr[:, B]

        # Compute the magnitude of the complex admittance
        edge_weight = torch.sqrt(real**2 + imag**2)

        # Add the computed edge weights to the data object
        data.edge_weight = edge_weight

        return data


@MASKING_REGISTRY.register("none")
class AddIdentityMask(BaseTransform):
    """Creates an identity mask, and adds it as a `mask` attribute.

    The mask is generated such that every entry is False, so no masking is actually applied
    """

    def __init__(self, args):
        super().__init__()

    def forward(self, data):
        if not hasattr(data, "y"):
            raise AttributeError("Data must have ground truth 'y'.")

        # Generate an identity mask
        mask = torch.zeros_like(data.y, dtype=torch.bool)

        # Add the mask to the data object
        data.mask = mask

        return data


@MASKING_REGISTRY.register("rnd")
class AddRandomMask(BaseTransform):
    """Creates a random mask, and adds it as a `mask` attribute.

    The mask is generated such that each entry is `True` with probability
    `mask_ratio` and `False` otherwise.
    """

    def __init__(self, args):
        super().__init__()
        self.mask_dim = args.data.mask_dim
        self.mask_ratio = args.data.mask_ratio

    def forward(self, data):
        if not hasattr(data, "x"):
            raise AttributeError("Data must have node features 'x'.")

        # Generate a random mask
        mask = torch.rand(data.x.size(0), self.mask_dim) < self.mask_ratio

        # Add the mask to the data object
        data.mask = mask

        return data


@MASKING_REGISTRY.register("pf")
class AddPFMask(BaseTransform):
    """Creates a mask according to the power flow problem and assigns it as a `mask` attribute."""

    def __init__(self, args):
        super().__init__()

    def forward(self, data):
        # Ensure the data object has the required attributes
        if not hasattr(data, "y"):
            raise AttributeError("Data must have ground truth 'y'.")

        if not hasattr(data, "x"):
            raise AttributeError("Data must have node features 'x'.")

        # Generate masks for each type of node
        mask_PQ = data.x[:, PQ] == 1  # PQ buses
        mask_PV = data.x[:, PV] == 1  # PV buses
        mask_REF = data.x[:, REF] == 1  # Reference buses

        # Initialize the mask tensor with False values
        mask = torch.zeros_like(data.y, dtype=torch.bool)

        mask[mask_PQ, VM] = True  # Mask Vm for PQ buses
        mask[mask_PQ, VA] = True  # Mask Va for PQ buses

        mask[mask_PV, QG] = True  # Mask Qg for PV buses
        mask[mask_PV, VA] = True  # Mask Va for PV buses

        mask[mask_REF, PG] = True  # Mask Pg for REF buses
        mask[mask_REF, QG] = True  # Mask Qg for REF buses

        # Attach the mask to the data object
        data.mask = mask

        return data


@MASKING_REGISTRY.register("opf")
class AddOPFMask(BaseTransform):
    """Creates a mask according to the optimal power flow problem and assigns it as a `mask` attribute."""

    def __init__(self, args):
        super().__init__()

    def forward(self, data):
        # Ensure the data object has the required attributes
        if not hasattr(data, "y"):
            raise AttributeError("Data must have ground truth 'y'.")

        if not hasattr(data, "x"):
            raise AttributeError("Data must have node features 'x'.")

        # Generate masks for each type of node
        mask_PQ = data.x[:, PQ] == 1  # PQ buses
        mask_PV = data.x[:, PV] == 1  # PV buses
        mask_REF = data.x[:, REF] == 1  # Reference buses

        # Initialize the mask tensor with False values
        mask = torch.zeros_like(data.y, dtype=torch.bool)

        mask[mask_PQ, VM] = True  # Mask Vm for PQ
        mask[mask_PQ, VA] = True  # Mask Va for PQ

        mask[mask_PV, PG] = True  # Mask Pg for PV
        mask[mask_PV, QG] = True  # Mask Qg for PV
        mask[mask_PV, VM] = True  # Mask Vm for PV
        mask[mask_PV, VA] = True  # Mask Va for PV

        mask[mask_REF, PG] = True  # Mask Pg for REF
        mask[mask_REF, QG] = True  # Mask Qg for REF
        mask[mask_REF, VM] = True  # Mask Vm for REF
        mask[mask_REF, VA] = True  # Mask Va for REF

        # Attach the mask to the data object
        data.mask = mask

        return data
