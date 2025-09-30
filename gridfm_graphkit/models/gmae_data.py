from collator import collator
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from functools import partial
import random
import torch
from wrapper import MyDataset, process_samples
from torch_geometric.utils import to_undirected

from torch_geometric.datasets import Planetoid, WikiCS, Amazon
from torch_geometric.loader import NeighborSampler
import torch_geometric.transforms as T
import hqdata


dataset = None


def get_dataset(dataset_name='Cora', nodefile='', edgefile=''):
    global dataset
    path = 'dataset/' + dataset_name
    if dataset is not None:
        return dataset

    elif dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
        return Planetoid(root=path, name=dataset_name, transform=T.NormalizeFeatures())
    elif dataset_name == 'WikiCS':
        return WikiCS(root=path, transform=T.NormalizeFeatures())
    elif dataset_name == 'Amazon-Computers':
        return Amazon(root=path, name='computers', transform=T.NormalizeFeatures())
    elif dataset_name == 'Amazon-Photo':
        return Amazon(root=path, name='photo', transform=T.NormalizeFeatures())
    elif dataset_name == 'hqdata':
        return hqdata.simple_batch(nodefile, edgefile)
    else:
        raise NotImplementedError

def read_csv(infile):
    """
    assume two columns: instances number, file location and name
    """

    lines = []
    with open(infile, 'r') as ff:
        for line in ff:
            lines.append([xx.strip() for xx in line.split(',')])

    return lines

class GraphDataModule(LightningDataModule):
    name = "Cora"

    def __init__(
        self,
        dataset_name: str = 'Cora',
        num_workers: int = 8,
        batch_size: int = 64,
        seed: int = 42,
        edgefile: str = '',
        nodefile: str = '',
        processedfile: str = '', # preprocessed dataset file in pt format
        n_val_sampler: int = 10,
        num_node_features: int = 25,
        test=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dataset_name = dataset_name
        if nodefile and edgefile:
            self.dataset = get_dataset(dataset_name, nodefile, edgefile)
        else: 
            self.dataset = read_csv(processedfile)
        self.num_node_features = num_node_features
        self.seed = seed
        self.n_val_sampler = n_val_sampler

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.dataset_full = ...
        self.dataset_train = ...
        self.dataset_val = ...
        self.dataset_test = ... # not currently in use
        self.train_frac = 0.8   # train-val split only
        self.istest = test


    def setup(self, stage: str = None):
        """
        automatically called, if prepare_data() is defined, then the latter
        is called first

        during testing this section is not needed
        """

        if self.istest:
            pass
        else:
            items = self.dataset    # for disk data the dataset is in items form
            self.dataset_full = MyDataset(
                                items, 
                                settype='csv', 
                                )

            # split the train and validation data
            train_set_size = int(self.train_frac*len(self.dataset_full))
            valid_set_size = len(self.dataset_full) - train_set_size
            seed = torch.Generator().manual_seed(self.seed)
            train_set, valid_set = random_split(
                                self.dataset_full, 
                                [train_set_size, valid_set_size], 
                                generator=seed
                                )
            print('**train and val dataset sizes**',len(train_set),len(valid_set))
            self.dataset_train = train_set
            self.dataset_val = valid_set


    def train_dataloader(self):
        loader = DataLoader(self.dataset_train, batch_size=self.batch_size,
                            shuffle=True,
                            num_workers=self.num_workers,
                            collate_fn=partial(collator),
                            )
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.dataset_val, batch_size=self.batch_size,
                            shuffle=False,
                            num_workers=self.num_workers,
                            collate_fn=partial(collator),
                            )
        return loader

    def eval_dataloader(self):
        """
        for downstream evaluation
        """
        # do not wish to shuffle for evaluation
        graphs_to_process = self.dataset.datalist

    
        items = []    # from in mem dataset 

        for graphdata in graphs_to_process:
            # padding and mask creation should happend here
            num_nodes = graphdata.num_nodes
            ns0 = 1 # batch size
            ns1 = torch.arange(num_nodes, dtype=torch.int32)   # node ids
            ns2 = graphdata.edge_index
            data_item = process_samples(
                        ns0, 
                        ns1, 
                        ns2,
                        graphdata) + [0]    # TODO completely remove the appended [0]
            items.append(data_item)

        self.dataset_eval = MyDataset(items)
        loader = DataLoader(self.dataset_eval, 
                            batch_size=self.batch_size*self.n_val_sampler,
                            shuffle=False,
                            num_workers=self.num_workers,
                            collate_fn=partial(collator),
                            )
        return loader
