import numpy as np
import torch as th

from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl.data import AmazonCoBuyPhotoDataset, AmazonCoBuyComputerDataset
from dgl.data import CoauthorCSDataset, CoauthorPhysicsDataset
from torch_geometric.datasets import WebKB, Actor
from dataset import WikipediaNetwork
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected
import networkx as nx
import dgl
import torch
import random
np.random.seed(1024)
torch.manual_seed(1024)
torch.cuda.manual_seed(1024)
random.seed(1024)

def read_hete_datasets(datasets):
    if datasets in ["cornell", "texas", "wisconsin"]:
        torch_dataset = WebKB(root=f'../../../datasets/', name=datasets, transform=T.NormalizeFeatures())
        data = torch_dataset[0]
    elif datasets in ['squirrel', 'chameleon']:
        torch_dataset = WikipediaNetwork(root=f'../../../datasets/', name=datasets, geom_gcn_preprocess=True)
        data = torch_dataset[0]
    elif datasets in ['crocodile']:
        torch_dataset = WikipediaNetwork(root=f'../../../datasets/', name=datasets, geom_gcn_preprocess=False)
        data = torch_dataset[0]
    elif datasets == 'film':
        torch_dataset = Actor(root=f'../../../datasets/film/', transform=T.NormalizeFeatures())
        data = torch_dataset[0]
    data.edge_index = to_undirected(data.edge_index)
    G = nx.from_edgelist(data.edge_index.transpose(0, 1).numpy().tolist())
    g = dgl.from_networkx(G)
    g.ndata['feat'] = data.x
    data.train_mask = data.train_mask.transpose(0, 1)
    data.val_mask = data.val_mask.transpose(0, 1)
    data.test_mask = data.test_mask.transpose(0, 1)

    num_class=torch_dataset.num_classes
  

    return g, data.x, data.y, num_class
def read_dgl(name):
    if name == 'cora':
        dataset = CoraGraphDataset()
    elif name == 'citeseer':
        dataset = CiteseerGraphDataset()
    elif name == 'pubmed':
        dataset = PubmedGraphDataset()
    elif name == 'photo':
        dataset = AmazonCoBuyPhotoDataset()
    elif name == 'computer':
        dataset = AmazonCoBuyComputerDataset()
    elif name == 'cs':
        dataset = CoauthorCSDataset()
    elif name == 'physics':
        dataset = CoauthorPhysicsDataset()

    graph = dataset[0]
    feat = graph.ndata.pop('feat')
    labels = graph.ndata.pop('label')
    num_class = dataset.num_classes
    return graph, feat, labels, num_class

def readdata(name):
    citegraph = ['cora', 'citeseer', 'pubmed']
    hete=["cornell", "texas", "wisconsin", 'chameleon','squirrel']

    if name in hete:
        graph, feat, labels, num_class=read_hete_datasets(name)
        return graph, feat, labels, num_class
    else:
        graph, feat, labels, num_class,_,_,_ = read20perclass(name)


        return graph, feat, labels, num_class

def read20perclass(datasets):
    path='../data'
    if datasets == 'cora':
        torch_dataset = CoraGraphDataset()
    elif datasets == 'citeseer':
        torch_dataset = CiteseerGraphDataset()
    elif datasets == 'pubmed':
        torch_dataset = PubmedGraphDataset()
    elif datasets == 'photo':
        torch_dataset = AmazonCoBuyPhotoDataset(raw_dir=path)
    elif datasets == 'computer':
        torch_dataset = AmazonCoBuyComputerDataset(raw_dir=path)
    elif datasets == 'cs':
        torch_dataset = CoauthorCSDataset(raw_dir=path)
    elif datasets == 'physics':
        torch_dataset = CoauthorPhysicsDataset()

    g = torch_dataset[0]
    g = g.remove_self_loop().add_self_loop()

    feat = g.ndata.pop('feat')
    labels = g.ndata.pop('label')
    N = g.number_of_nodes()
    num_class=torch_dataset.num_classes
    split_list = []
    if datasets not in ['cora','citeseer','pubmed']:
        train_idx = torch.tensor(np.load("../../../20CLASS/{}/train_mask.npy".format(datasets)))
        val_idx = torch.tensor(np.load("../../../20CLASS/{}/val_mask.npy".format(datasets)))
        test_idx = torch.tensor(np.load("../../../20CLASS/{}/test_mask.npy".format(datasets)))
        return g, feat, labels, num_class, train_idx, val_idx, test_idx
    else:
        train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']
        train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze()
        val_idx = torch.nonzero(val_mask, as_tuple=False).squeeze()
        test_idx = torch.nonzero(test_mask, as_tuple=False).squeeze()
        split_list.append({'train_idx': train_idx,
                           'valid_idx': val_idx,
                           'test_idx': test_idx})

        return g, feat, labels, num_class, train_idx, val_idx, test_idx
