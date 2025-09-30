import torch

import numpy as np

from torch_geometric.loader import NeighborSampler
from torch_geometric.utils import to_undirected



def process_samples(batch_size, n_id, edge_index, dataset):
    """
    transformation of sampled nodes to: 
    - node features of sampled set, 
    - y, 
    - edges tensor

    # TODO reconcile redundance of using edge_index and dataset
    # in the case where the full graph is used
    """

    # print(edge_index)
    # print('<------->')
    if edge_index.size(1) != 0:
        edge_index = to_undirected(edge_index)
    n_nodes = len(n_id)
    edge_sp_adj = torch.sparse.FloatTensor(edge_index,
                                            torch.ones(edge_index.shape[1]),
                                            [n_nodes, n_nodes])
    edge_adj = edge_sp_adj

    # print('<<---------------->>')
    # print(n_id)
    # print(dataset.x.size())
    # print(dataset.y.size())

    return [dataset.x[n_id], dataset.y[n_id], edge_adj]
    

# GMAE_graph positional encoding
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, items, settype=''):
        super(MyDataset, self).__init__()

        self.items = items
        self.type = settype


    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        
        if self.type=='csv':
            graphdata = torch.load(item[1])
            num_nodes = graphdata.num_nodes
        
            # padding and mask creation should happend here
            ns0 = 1 # batch size
            ns1 = torch.arange(num_nodes, dtype=torch.int32)   # node ids
            ns2 = graphdata.edge_index
            data_item = process_samples(
                        ns0, 
                        ns1, 
                        ns2,
                        graphdata) + [0]    # TODO completely remove the appended [0]
        else:
            data_item = item    # in memory dataset in use

        return preprocess_item(data_item)


def preprocess_item(item):
    """
    """
    x, y, adj, orig_id = item[0], item[1], item[2].to_dense(), item[3]
    N = x.size(0)

    # node adj matrix [N, N] bool
    adj = adj.bool()

    shortest_path_result, path = algos.floyd_warshall(adj.numpy())
    spatial_pos = torch.from_numpy((shortest_path_result)).long()
    attn_bias = torch.zeros([N, N], dtype=torch.float)

    in_degree = adj.long().sum(dim=1).view(-1)
    out_degree = adj.long().sum(dim=0).view(-1)
    return x, y, adj, attn_bias, spatial_pos, in_degree, out_degree, orig_id
