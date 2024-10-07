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

from introduction import *
from utils import load_citation, accuracy
import warnings
warnings.filterwarnings('ignore')

# Settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--epochs', type=int, default=400,   help='Number of epochs to train.')

parser.add_argument('--seed', type=int, default=12345,   help='Number of epochs to train.')

parser.add_argument('--lr', type=float, default=0.01,   help='learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.6,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='chameleon',
                    help='dataset to be used')
parser.add_argument('--belta', type=float, default=5e-4,  help='To control the ratio of Ncontrast loss')
parser.add_argument('--train_rate', type=float, default=0.1,  help='train_rate')
parser.add_argument('--val_rate', type=float, default=0.1,  help='val_rate')

parser.add_argument('--K', type=int, default=2,  help='layer')
parser.add_argument('--order', type=int, default=2,  help='to compute order-th power of adj')
parser.add_argument('--device', type=int, default=0,  help='device')
parser.add_argument('--tau', type=float, default=1.0,   help='temperature for Ncontrast loss')
parser.add_argument('--net', type=str, default='MLP_SampleCross',
                    help='dataset to be used')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
from dataset_loader import DataLoader

from scipy.sparse import coo_matrix
import numpy as np
import torch
import scipy.sparse as sp
import numpy as np

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = 1
    return mask


def random_planetoid_splits(data, num_classes, percls_trn=20, val_lb=500, seed=12134):
    index = [i for i in range(0, data.y.shape[0])]
    train_idx = []
    rnd_state = np.random.RandomState(seed)
    for c in range(num_classes):
        class_idx = np.where(data.y.cpu() == c)[0]
        if len(class_idx) < percls_trn:
            train_idx.extend(class_idx)
        else:
            train_idx.extend(rnd_state.choice(class_idx, percls_trn, replace=False))
    rest_index = [i for i in index if i not in train_idx]
    val_idx = rnd_state.choice(rest_index, val_lb, replace=False)
    test_idx = [i for i in rest_index if i not in val_idx]

    data.train_mask = index_to_mask(train_idx, size=data.num_nodes)
    data.val_mask = index_to_mask(val_idx, size=data.num_nodes)
    data.test_mask = index_to_mask(test_idx, size=data.num_nodes)

    return data

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
    # print(lnc)
    return lnc

####################################
## get data
from normalization import fetch_normalization,row_normalize
def get_data(args):
    if args.dataset in ['wis']:
        adj, features, labels, idx_train, idx_val, idx_test = load_citation(args.dataset, 'AugNormAdj', True)

    else:
        dataset=DataLoader(args.dataset)
        data=dataset[0]
        features=row_normalize(data.x)  #numpy
        features= torch.FloatTensor(features)
        labels =data.y

        percls_trn = int(round(args.train_rate * len(data.y) / dataset.num_classes))
        val_lb = int(round(args.val_rate * len(data.y)))

        coo_adj = coo_matrix((np.ones(data.num_edges), (data.edge_index[0].numpy(), data.edge_index[1].numpy())),
                             shape=(data.num_nodes, data.num_nodes))
        I = coo_matrix((np.ones(data.num_nodes), (np.arange(0, data.num_nodes, 1), np.arange(0, data.num_nodes, 1))),
                       shape=(data.num_nodes, data.num_nodes))

        adj =sparse_mx_to_torch_sparse_tensor(sys_normalized_adjacency(coo_adj))
        posadj=cooadj2Sparsetentor(coo_adj)
        lap_adj =torch.sub(torch.eye(data.num_nodes),adj)
        permute_masks = random_planetoid_splits
        data = permute_masks(data, dataset.num_classes, percls_trn, val_lb, args.seed)
        print(args.dataset)
        idx_train=[index for index, value in enumerate(data.train_mask) if value]
        idx_val = [index for index, value in enumerate(data.val_mask) if value]
        idx_test = [index for index, value in enumerate(data.test_mask) if value]
        idx_train= torch.LongTensor(idx_train)
        idx_val=torch.LongTensor(idx_val)
        idx_test=torch.LongTensor(idx_test)

        data.adj=adj.to_dense()
        adj_neg =load_adj_neg(data.num_nodes,2)
        data.adj_neg=adj_neg.to_dense()
    return dataset, data, labels, idx_train, idx_val, idx_test






def load_adj_neg(num_nodes, sample):

    col = np.random.randint(0, num_nodes, size=num_nodes * sample)
    row = np.repeat(range(num_nodes), sample)
    index = np.not_equal(col,row)
    col = col[index]
    row = row[index]
    new_col = np.concatenate((col,row),axis=0)
    new_row = np.concatenate((row,col),axis=0)

    data = np.ones(new_col.shape[0])
    adj_neg = sp.coo_matrix((data, (new_row, new_col)), shape=(num_nodes, num_nodes))

    adj_neg = sparse_mx_to_torch_sparse_tensor(sys_normalized_adjacency(adj_neg))

    return adj_neg



#dataset, lap_adj, features, labels, idx_train, idx_val, idx_test =get_data(args)
dataset, data,labels, idx_train, idx_val, idx_test = get_data(args)

gnn_name = args.net
if gnn_name == 'GCN':
    Net = GCN
elif gnn_name == 'SampleGra':
    Net = SampleGra
elif gnn_name == 'OBJ':
    Net = OBJ
else:
    Net = MLP
## Model and optimizer
device = torch.device('cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu')
data=data.to(device)

labels = labels.to(device)
idx_train = idx_train.to(device)
idx_val = idx_val.to(device)
idx_test = idx_test.to(device)
    # adj=adj.to_dense().to(device)
    # adj_neg=adj_neg.to_dense().to(device)


if args.net=='SampleGra':
    model = Net(dataset, data,args).to(device)
else:
    model = Net(dataset, args).to(device)
optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)

def train():

    model.train()
    optimizer.zero_grad()
    output, x = model(data)

    loss_train= F.nll_loss(output[idx_train], labels[idx_train])

    # loss_trace=torch.trace(torch.mm(torch.mm(x_dis.t(), lap_adj), x_dis))
    # loss_sample = samplNCE(x, adj, adj_neg)
    # loss_train = loss_train_class + loss_sample * args.belta
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()


def test():
    model.eval()
    output,dis = model(data)
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
#filename = f'GCN_traceCross.csv'
print(f"Saving results to {filename}")
with open(f"{filename}", 'a+') as write_obj:
    write_obj.write(
            "dataset:{}, hidden:{}, lr:{}, hop:{}, belta:{}, train_rate:{}, meanmicro:{} ".format(args.dataset, args.hidden, args.lr,args.order, args.belta, args.train_rate, test_acc))
    write_obj.write("\n")



