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

from models import *
from utils import load_citation, accuracy, get_A_r
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
parser.add_argument('--hidden', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.6,  help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='chameleon',  help='dataset to be used')
parser.add_argument('--belta', type=float, default=2.0,  help='To control the ratio of Ncontrast loss')
parser.add_argument('--train_rate', type=float, default=0.6,  help='train_rate')
parser.add_argument('--val_rate', type=float, default=0.2,  help='val_rate')


parser.add_argument('--order', type=int, default=2,  help='to compute order-th power of adj')
parser.add_argument('--device', type=int, default=0,  help='device')
parser.add_argument('--tau', type=float, default=0.5,   help='temperature for Ncontrast loss')



parser.add_argument('--net', type=str, default='GCNLOSS',  help='dataset to be used')
parser.add_argument('--batch_size', type=int, default=2000,  help='batch size')
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



####################################
## get data
from normalization import fetch_normalization,row_normalize
def get_data(args):
    if args.dataset in ['arxiv']:
        adj, features, labels, idx_train, idx_val, idx_test = load_citation(args.dataset, 'AugNormAdj', True)
        print(labels)
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
        #adj=cooadj2Sparsetentor(coo_adj)
        permute_masks = random_planetoid_splits
        data = permute_masks(data, dataset.num_classes, percls_trn, val_lb, args.seed)
        print(args.dataset)
        idx_train=[index for index, value in enumerate(data.train_mask) if value]
        idx_val = [index for index, value in enumerate(data.val_mask) if value]
        idx_test = [index for index, value in enumerate(data.test_mask) if value]
        idx_train= torch.LongTensor(idx_train)
        idx_val=torch.LongTensor(idx_val)
        idx_test=torch.LongTensor(idx_test)
    return dataset, adj, features, labels, idx_train, idx_val, idx_test





dataset, adj, features, labels, idx_train, idx_val, idx_test =get_data(args)
adj_label = get_A_r(adj, 0)
data=dataset[0]

## Model and optimizer
# model = GMLP(nfeat=features.shape[1],
#             nhid=args.hidden,
#             nclass=labels.max().item() + 1,
#             dropout=args.dropout)


model = GCN_LOSS(dataset,args)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


if args.tau>0:
    device = torch.device('cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu')
    model.to(device)
    data = data.to(device)
    labels = labels.to(device)
    idx_train = idx_train.to(device)

    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)
    adj_label =adj_label.to(device)

if args.dataset=='arxiv':
    mask = torch.eye(args.batch_size).to(device)
else:

    mask = torch.eye(features.shape[0]).to(device)
#三种形式
def Ncontrast(x_dis, adj_label, tau = 1):
    """
    compute the Ncontrast loss
    """
    x_dis = torch.exp(tau * x_dis)
    x_dis_sum = torch.sum(x_dis, 1)
    x_dis_sum_pos = torch.sum(x_dis*adj_label, 1)
    ####   loss = -torch.log(x_dis_sum_pos).mean()  #only the postive samples
    loss = -torch.log(x_dis_sum_pos * (x_dis_sum ** (-1)) + 1e-8).mean()
    loss = torch.log(x_dis_sum).mean()
    return loss

def get_batch(batch_size):
    """
    get a batch of feature & adjacency matrix
    """
    rand_indx = torch.tensor(np.random.choice(np.arange(adj_label.shape[0]), batch_size)).type(torch.long).cuda()
    rand_indx[0:len(idx_train)] = idx_train
    features_batch = features[rand_indx]
    adj_label_batch = adj_label[rand_indx,:][:,rand_indx]
    print('adj_label_batch')
    print(adj_label_batch.shape)
    return features_batch, adj_label_batch

def train():
    #features_batch, adj_label_batch = get_batch(batch_size=args.batch_size)
    features_batch, adj_label_batch =features, adj_label
    model.train()
    optimizer.zero_grad()
    output, x_dis = model(data,mask)
    x_dis=x_dis.to(device)

    loss_train_class = F.nll_loss(output[idx_train], labels[idx_train])
    loss_Ncontrast = Ncontrast(x_dis, adj_label_batch, tau = args.tau)
    loss_train = loss_train_class + loss_Ncontrast * args.belta
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    return 

def test():
    model.eval()
    output,_ = model(data,mask)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])

    acc_train = accuracy(output[idx_train], labels[idx_train])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    return  acc_test, acc_val, acc_train, loss_train, loss_test

best_accu = 0
best_val_acc = 0
print('\n'+'training configs', args)
for epoch in tqdm(range(args.epochs)):
    train()
    tmp_test_acc, val_acc, acc_train, loss_train, loss_test = test()
    # if epoch%20==0:
    #     filename = f'CMP/{args.dataset}_loss.csv'
    #     with open(f"{filename}", 'a+') as write_obj:
    #         write_obj.write("{},{}".format(loss_train,loss_test))
    #         write_obj.write("\n")
    #
    #     file= f'CMP/{args.dataset}_acc.csv'
    #     with open(f"{file}", 'a+') as write_obj:
    #         write_obj.write("{},{}".format(acc_train,tmp_test_acc))
    #         write_obj.write("\n")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc

print(test_acc)
filename = f'{args.net}_C.csv'
print(f"Saving results to {filename}")
with open(f"{filename}", 'a+') as write_obj:
    write_obj.write( "dataset:{}, hidden:{}, lr:{}, hop:{}, belta:{}, tau:{}, meanmicro:{} ".format(args.dataset, args.hidden, args.lr,args.order, args.belta, args.tau, test_acc))
    write_obj.write("\n")


# log_file = open(r"log.txt", encoding="utf-8",mode="a+")
# with log_file as file_to_be_write:
#     print('tau', 'order', \
#             'batch_size', 'hidden', \
#                 'alpha', 'lr', \
#                     'weight_decay', 'data', \
#                         'test_acc', file=file_to_be_write, sep=',')
#     print(args.tau, args.order, \
#          args.batch_size, args.hidden, \
#              args.alpha, args.lr, \
#                  args.weight_decay, args.data, \
#                      test_acc.item(), file=file_to_be_write, sep=',')


