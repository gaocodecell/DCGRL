import argparse
import os.path as osp
from models import LogReg,Model
from utils import readdata
import torch
import torch as th
import torch.nn as nn
import  numpy as np
import warnings
import random
import scipy.sparse as sp
import dgl
import os
import torch.nn.functional as F
warnings.filterwarnings('ignore')
np.random.seed(1024)
torch.manual_seed(1024)
torch.cuda.manual_seed(1024)
random.seed(1024)

parser = argparse.ArgumentParser(description='GCGRL')
parser.add_argument('--dataset', type=str, default='pubmed', help='Name of dataset.')
parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
parser.add_argument('--epochs', type=int, default=50, help='Training epochs.')
parser.add_argument('--lr1', type=float, default=5e-4, help='Learning rate of pretraining.')
parser.add_argument('--lr2', type=float, default=1e-2, help='Learning rate of linear evaluator.')
parser.add_argument('--wd1', type=float, default=1e-6, help='Weight decay of pretraining.')
parser.add_argument('--wd2', type=float, default=1e-5, help='Weight decay of linear evaluator.')
parser.add_argument('--n_layers', type=int, default=2, help='Number of GNN layers')
parser.add_argument('--tau', type=float, default=0.75, help='Temperature hyperparameter.')
parser.add_argument('--tau2', type=float, default=0.75, help='Temperature hyperparameter.')
parser.add_argument("--hid_dim", type=int, default=2048, help='Hidden layer dim.')
parser.add_argument("--out_dim", type=int, default=2048, help='Hidden layer dim.')
parser.add_argument('--moving_average_decay', type=float, default=0.99)
parser.add_argument('--num_MLP', type=int, default=1)
parser.add_argument('--run_times', type=int, default=10)
parser.add_argument('--lama', type=float, default=1, help='Temperature hyperparameter.')
parser.add_argument('--net', type=str, default='DCGRL', help='Name of model.')
parser.add_argument('--alpha', type=float, default=0.1, help='Temperature hyperparameter.')
parser.add_argument('--train_rate', type=float, default=0.1, help='Temperature hyperparameter.')
parser.add_argument('--val_rate', type=float, default=0.1, help='Temperature hyperparameter.')

args = parser.parse_args()


# check cuda
if args.gpu != -1 and th.cuda.is_available():
    args.device = 'cuda:{}'.format(args.gpu)
else:
    args.device = 'cpu'


def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x
def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()
#
def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

def add_self_loops(adj_matrix):
    # 
    np.fill_diagonal(adj_matrix, 1)
    return adj_matrix
import psutil

# Function to measure memory usage
def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / float(2 ** 20)  # in MB
    return mem

##############################################

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = 1
    return mask


def random_planetoid_splits(N,labels, num_classes, percls_trn=20, val_lb=500, seed=12134):
    index = [i for i in range(0, N)]
    train_idx = []
    rnd_state = np.random.RandomState(seed)
    for c in range(num_classes):
        class_idx = np.where(labels.cpu() == c)[0]
        if len(class_idx) < percls_trn:
            train_idx.extend(class_idx)
        else:
            train_idx.extend(rnd_state.choice(class_idx, percls_trn, replace=False))
    rest_index = [i for i in index if i not in train_idx]
    val_idx = rnd_state.choice(rest_index, val_lb, replace=False)
    test_idx = [i for i in rest_index if i not in val_idx]

    return train_idx, val_idx, test_idx
def chose_stu(name, graph, g, feat, model):
    small=['cornell','texas','wisconsin']
    if name in small:
        embeds = model.get_embedding(g, feat)
        return embeds
    else:
        embeds = model.get_embedding(graph, feat)
        return embeds


if __name__ == '__main__':
    print(args)
    # load hyperparameters
    dataname = args.dataset
    hid_dim = args.hid_dim
    out_dim = args.out_dim
    n_layers = args.n_layers
    tau = args.tau
    epochs = args.epochs

    device = args.device

    graph, feat, labels, num_class= readdata(dataname)

    print(labels)
    print(num_class)
    in_dim = feat.shape[1]
    m,n = feat.shape[0], feat.shape[1]
    graph = graph.to(device)
    model = Model(graph,in_dim, hid_dim, out_dim, n_layers, tau, args.moving_average_decay, args.alpha)
    model = model.to(device)

  
    g = dgl.DGLGraph()

   
    N = graph.number_of_nodes()
    g.add_nodes(N)
    #
    for i in range(N):
        g.add_edges(i, i)
    g = g.to(device)

    feat = feat.to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr1, weight_decay=args.wd1)
    graph = graph.remove_self_loop().add_self_loop()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        embeding1, target1, loss1 = model(graph, feat, args.tau)
        embeding2, target2, loss2 = model(g, feat, args.tau2)
        luu=F.mse_loss(embeding1,embeding2)
        lvv=F.mse_loss(target1,target2)
        lossanother=luu+lvv
        loss=(loss1+loss2)+args.lama*lossanother
        loss.backward()
        optimizer.step()
        model.update_moving_average()
        # print('Epoch={:03d}, loss={:.4f}'.format(epoch, loss.item()))
        if epoch % 5 == 0:
            print('Epoch={:03d}, conloss={:.4f}'.format(epoch, loss.item()))

    print("=== Evaluation ===")
    embeds=chose_stu(args.dataset, graph, g, feat, model)

    results = []
    accs=[]
    percls_trn = int(round(args.train_rate * N / num_class))
    val_lb = int(round(args.val_rate * N))
    seeds = [77, 194, 419, 47, 121, 401, 210, 164, 629, 242, 32121]
    for run in range(0, args.run_times):
        # randomly split dataset
        permute_masks = random_planetoid_splits

        train_idx, val_idx, test_idx= permute_masks(N,labels, num_class, percls_trn, val_lb, seeds[run])

        train_idx_tmp = train_idx
        val_idx_tmp = val_idx
        test_idx_tmp = test_idx
        train_embs = embeds[train_idx_tmp]
        val_embs = embeds[val_idx_tmp]
        test_embs = embeds[test_idx_tmp]
        label = labels.to(device)

        train_labels = label[train_idx_tmp]
        val_labels = label[val_idx_tmp]
        test_labels = label[test_idx_tmp]

        train_feat = feat[train_idx_tmp]
        val_feat = feat[val_idx_tmp]
        test_feat = feat[test_idx_tmp]

        ''' Linear Evaluation '''
        logreg = LogReg(train_embs.shape[1], num_class)
        opt = th.optim.Adam(logreg.parameters(), lr=args.lr2, weight_decay=args.wd2)
        logreg = logreg.to(device)
        loss_fn = nn.CrossEntropyLoss()

        best_val_acc = 0
        eval_acc = 0

        for epoch in range(500):
            logreg.train()
            opt.zero_grad()
            logits = logreg(train_embs)
            preds = th.argmax(logits, dim=1)
            train_acc = th.sum(preds == train_labels).float() / train_labels.shape[0]
            loss = loss_fn(logits, train_labels)
            loss.backward()
            opt.step()

            logreg.eval()
            #print(f"Memory usage after epoch {epoch + 1}: {get_memory_usage()} MB")
            with th.no_grad():
                val_logits = logreg(val_embs)
                test_logits = logreg(test_embs)

                val_preds = th.argmax(val_logits, dim=1)
                test_preds = th.argmax(test_logits, dim=1)

                val_acc = th.sum(val_preds == val_labels).float() / val_labels.shape[0]
                test_acc = th.sum(test_preds == test_labels).float() / test_labels.shape[0]

                if val_acc >= best_val_acc:
                    best_val_acc = val_acc
                    eval_acc = test_acc

                print('Epoch:{}, train_acc:{:.4f}, val_acc:{:4f}, test_acc:{:4f}'.format(epoch, train_acc, val_acc, test_acc))
        print(f'Validation Accuracy: {best_val_acc}, Test Accuracy: {eval_acc}')
        accs.append(eval_acc.detach().cpu().numpy())

    print(accs)
    meanmicro = sum(accs) / 10
    m1 = np.std(accs)
    print(args.dataset)
    print(meanmicro)
    filename = f'{args.net}.csv'
    print(f"Saving results to {filename}")
    with open(f"{filename}", 'a+') as write_obj:
        write_obj.write(
            "net:{}, dataset:{}, n_layers:{}, tau1:{}, tau2:{}, wd1:{}, wd2:{}, lama:{}, alpha:{},  acc_test:{}, std:{}".format(
       args.net, args.dataset, args.n_layers, args.tau, args.tau2, args.wd1,args.wd2,  args.lama, args.alpha, meanmicro,m1))
        write_obj.write("\n")
