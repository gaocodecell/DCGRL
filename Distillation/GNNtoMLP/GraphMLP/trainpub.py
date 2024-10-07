from __future__ import division
from __future__ import print_function
import random
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from utils import load_citation, accuracy, get_A_r
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout, Linear, LayerNorm
import scipy.sparse as sp
class Mlp(nn.Module):
    def __init__(self, input_dim, hid_dim, dropout):
        super(Mlp, self).__init__()
        self.fc1 = Linear(input_dim, hid_dim)
        self.fc2 = Linear(hid_dim, hid_dim)
        self.act_fn = torch.nn.functional.gelu
        self._init_weights()

        self.dropout = Dropout(dropout)
        self.layernorm = LayerNorm(hid_dim, eps=1e-6)

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.layernorm(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def get_feature_dis(x):
    """
    x :           batch_size x nhid
    x_dis(i,j):   item means the similarity between x(i) and x(j).
    """
    x_dis = x@x.T
    mask = torch.eye(x_dis.shape[0]).cuda()
    x_sum = torch.sum(x**2, 1).reshape(-1, 1)
    x_sum = torch.sqrt(x_sum).reshape(-1, 1)
    x_sum = x_sum @ x_sum.T
    x_dis = x_dis*(x_sum**(-1))
    x_dis = (1-mask) * x_dis
    return x_dis

class GMLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GMLP, self).__init__()
        self.nhid = nhid
        self.mlp = Mlp(nfeat, self.nhid, dropout)
        self.classifier = Linear(self.nhid, nclass)

    def forward(self, x):
        x = self.mlp(x)

        feature_cls = x
        Z = x

        if self.training:
            x_dis = get_feature_dis(Z)

        class_feature = self.classifier(feature_cls)
        class_logits = F.log_softmax(class_feature, dim=1)

        if self.training:
            return class_logits, x_dis
        else:
            return class_logits


##################################
def cooadj2Sparsetentor(coo_adj):
    ' there is a trans=formation '
    values = coo_adj.data
    indices = np.vstack((coo_adj.row, coo_adj.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo_adj.shape
    # torch_sparse_adj is torch.sparse matrix
    adj = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    print(adj.shape)
    adj=adj.to_dense()
    return adj


def sys_normalized_adjacency(adj):
   adj = sp.coo_matrix(adj)
   adj = adj + sp.eye(adj.shape[0])
   row_sum = np.array(adj.sum(1))*1.0

   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def samplNCE(x, adj, adj_neg ):
    f= lambda x:torch.sigmoid(x)
    x_dis=f(torch.mm(x, x.t()))

    pos=torch.sum(torch.mul(x_dis, adj))
    neg =torch.sum(torch.mul(x_dis, adj_neg))
    lnc =pos+neg

    return lnc

####################################
## get data
from normalization import fetch_normalization,row_normalize
from dataset_loader import DataLoader
from scipy.sparse import coo_matrix

# Settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--epochs', type=int, default=400,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.6,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='Photo',
                    help='dataset to be used')
parser.add_argument('--alpha', type=float, default=2.0,
                    help='To control the ratio of Ncontrast loss')
parser.add_argument('--batch_size', type=int, default=2048,
                    help='batch size')
parser.add_argument('--order', type=int, default=2,  help='to compute order-th power of adj')
parser.add_argument('--device', type=int, default=0,  help='device')
parser.add_argument('--tau', type=float, default=1.0,   help='temperature for Ncontrast loss')
parser.add_argument('--net', type=str, default='2023GraphMLP',
                    help='dataset to be used')

args = parser.parse_args()
def get_data(name):
    if name in ['cora','citeseer', 'pubmed']:
        adj, features, labels, idx_train, idx_val, idx_test = load_citation(name, 'AugNormAdj', True)

    else:
        dataset = DataLoader(name)
        data = dataset[0]
        features = row_normalize(data.x)  # numpy
        features = torch.FloatTensor(features)
        labels = data.y



        coo_adj = coo_matrix((np.ones(data.num_edges), (data.edge_index[0].numpy(), data.edge_index[1].numpy())),
                             shape=(data.num_nodes, data.num_nodes))
        I = coo_matrix((np.ones(data.num_nodes), (np.arange(0, data.num_nodes, 1), np.arange(0, data.num_nodes, 1))),
                       shape=(data.num_nodes, data.num_nodes))


        adj = sparse_mx_to_torch_sparse_tensor(sys_normalized_adjacency(coo_adj))
        posadj = cooadj2Sparsetentor(coo_adj)
        # lap_adj = torch.sub(torch.eye(data.num_nodes), adj)

        print(name)
        idx_train = torch.LongTensor(np.load("../20CLASS/{}/train_mask.npy".format(name)))
        idx_val = torch.LongTensor(np.load("../20CLASS/{}/val_mask.npy".format(name)))
        idx_test = torch.LongTensor(np.load("../20CLASS/{}/test_mask.npy".format(name)))
        print(idx_train)


    return adj.to_dense(), features, labels, idx_train, idx_val, idx_test

adj, features, labels, idx_train, idx_val, idx_test=get_data(args.dataset)
device = torch.device('cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu')
## Model and optimizer
model = GMLP(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout).to(device)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
adj_label=get_A_r(adj,2)


adj_label=adj_label.to(device)
features=features.to(device)
labels = labels.to(device)
idx_train = idx_train.to(device)
idx_val = idx_val.to(device)
idx_test = idx_test.to(device)

def Ncontrast(x_dis, adj_label, tau = 1):
    """
    compute the Ncontrast loss
    """
    x_dis = torch.exp( tau * x_dis)
    x_dis_sum = torch.sum(x_dis, 1)
    x_dis_sum_pos = torch.sum(x_dis*adj_label, 1)
    loss = -torch.log(x_dis_sum_pos * (x_dis_sum**(-1))+1e-8).mean()
    return loss

def get_batch(batch_size):
    """
    get a batch of feature & adjacency matrix
    """
    rand_indx = torch.tensor(np.random.choice(np.arange(adj_label.shape[0]), batch_size)).type(torch.long).cuda()
    rand_indx[0:len(idx_train)] = idx_train
    features_batch = features[rand_indx]
    adj_label_batch = adj_label[rand_indx,:][:,rand_indx]
    return features_batch, adj_label_batch

def train():
    features_batch, adj_label_batch = get_batch(batch_size=args.batch_size)
    model.train()
    optimizer.zero_grad()
    output, x_dis = model(features_batch)
    loss_train_class = F.nll_loss(output[idx_train], labels[idx_train])
    loss_Ncontrast = Ncontrast(x_dis, adj_label_batch, tau = args.tau)
    loss_train = loss_train_class + loss_Ncontrast * args.alpha
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
def trainnobatch():

    model.train()
    optimizer.zero_grad()
    output, x_dis = model(features)

    loss_train_class= F.nll_loss(output[idx_train], labels[idx_train])

    loss_Ncontrast = Ncontrast(x_dis, adj_label, tau=args.tau)
    loss_train = loss_train_class + loss_Ncontrast * args.alpha

    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()


def test():
    model.eval()
    output = model(features)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    return acc_test, acc_val

best_accu = 0
best_val_acc = 0
print('\n'+'training configs', args)
for epoch in tqdm(range(args.epochs)):
    train()
    tmp_test_acc, val_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc

print(test_acc)
filename = f'{args.net}.csv'
print(f"Saving results to {filename}")
with open(f"{filename}", 'a+') as write_obj:
    write_obj.write(
            "dataset:{}, hidden:{}, lr:{}, tau:{}, hop:{}, alpha:{}, meanmicro:{} ".format(args.dataset, args.hidden, args.lr, args.tau, args.order, args.alpha, test_acc))
    write_obj.write("\n")
