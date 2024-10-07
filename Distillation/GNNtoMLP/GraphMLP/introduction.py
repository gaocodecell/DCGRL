import torch
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, ChebConv

from torch.nn import Parameter

from torch.nn import Linear
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm

import torch.nn as nn

import numpy as np



class MLP(torch.nn.Module):
    def __init__(self, dataset,args):
        super(MLP, self).__init__()

        self.lin1 = Linear(dataset.num_features, args.hidden)

        # self.lins1 = nn.ModuleList([nn.Linear(self.nhid, 64) for _ in range(args.K-2)])  # there需要一个
        self.lin2 = Linear(args.hidden, dataset.num_classes)
        self.dropout = args.dropout
        self.K = args.K


    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        # for i in range(self.K-2):
        #     x= torch.relu(self.lins1[i](x))
        x = self.lin2(x)

        return F.log_softmax(x, dim=1), x
class GCN(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, args.hidden)
        self.conv2 = GCNConv(args.hidden, dataset.num_classes)
        # self.gcs = nn.ModuleList([GCNConv(args.hidden, args.hidden) for _ in range(args.K - 2)])  # there需要一个
        self.dropout = args.dropout
        self.K = args.K

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        if self.dropout>0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        # for i in range(self.K - 2):
        #     x = F.relu(self.gcs[i](x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1), x


#####################SampleNCE  begin  ###################################

class ThetaConv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_index, negedge_index):
        super(ThetaConv, self).__init__(aggr='add',flow="source_to_target")  # "Add" aggregation (Step 5).


        self.lin1 = nn.Linear(in_channels, out_channels, bias=False)
        self.lin2 = nn.Linear(in_channels, out_channels, bias=False)
        self.floop = nn.Linear(in_channels, out_channels, bias=False)

        self.edge_index=edge_index
        self.row, self.col = self.edge_index


        self.negedge_index=negedge_index
        self.nrow, self.ncol = self.nrgedge_index
        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, h1):

        numnode=h1.shape[0]
        h3 = torch.relu(self.floop(h1))

        ss = torch.mul(h1[self.row], h1[self.col])#h1 = torch.relu(self.floop(h1))
        s = torch.sum(ss, dim=1)
        s = torch.sigmoid(s)
        pos = 1 - s
        # threshold = self.step

        # 将矩阵中的每个值减去0.5   .cpu()
        # tensor_minus_05 = matrix.detach() - 0.5
        # # 计算绝对值并判断是否小于阈值
        # abs_tensor = torch.abs(tensor_minus_05)
        # mask = abs_tensor < threshold
        #
        # # 将满足条件的元素赋值为
        # s[mask] =0
        # pos[mask] = 0

        x3 = self.propagate(self.edge_index, size=(numnode, numnode), x=h3, norm=pos, flow ='source_to_target')

        nss = torch.mul(h1[self.nrow], h1[self.ncol])  # h1 = torch.relu(self.floop(h1))
        ns = torch.sum(nss, dim=1)
        ns = torch.sigmoid(ns)

        x4 = self.propagate(self.negedge_index, size=(numnode, numnode), x=h3, norm=ns, flow='source_to_target')
        h3 = torch.add(h1, x3)
        h3 = torch.sub(h3, x4)   #这里时有影响的么

        return h3

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out

class SampleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SampleConv, self).__init__()


        self.lin1 = nn.Linear(in_channels, out_channels, bias=False)
        self.lin2 = nn.Linear(in_channels, out_channels, bias=False)
        self.floop = nn.Linear(in_channels, out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.floop.reset_parameters()

    def forward(self, h1, adj,negadj):
        h3 = self.lin1(h1)
        h4 = self.lin2(h1)
        h1 = self.floop(h1)

        s = torch.mm(h3, h4.t())
        s = torch.sigmoid(-1*s)
        neg = 1 - s

        posx = torch.mul(s, adj)
        negx = torch.mul(neg, negadj)
        h = torch.add(h1, torch.mm(posx, h1))
        h = torch.sub(h, torch.mm(negx, h1))

        return h
class SampleGra(nn.Module):
    def __init__(self, dataset, data, args):
        super(SampleGra, self).__init__()
        self.K = args.K
        self.dropout = args.dropout
        self.nfeat = dataset.num_features
        self.num_hidden=args.hidden

        # self.para = args.alpha
        # self.sample =args.sample
        # self.hop =args.hop
        # self.act = args.activation



        self.layers1 = nn.ModuleList([SampleConv(self.num_hidden, self.num_hidden) for _ in range(args.K)])
        self.lin1 = nn.Linear(dataset.num_features, self.num_hidden)


        self.out_att1 = nn.Linear(self.num_hidden, dataset.num_classes)
        self.out_att2 = nn.Linear(2*self.num_hidden, dataset.num_classes)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()

        self.out_att1.reset_parameters()
        self.out_att2.reset_parameters()

    def forward(self, data):
        posadj, negadj =data.adj, data.adj_neg
        x, edge_index = data.x, data.edge_index
        Q1 = self.lin1(x)

        Q = F.dropout(Q1, p=self.dropout, training=self.training)


        for i in range(self.K):

            hq3 = self.layers1[i](Q, posadj, negadj)
            Q3 = F.normalize(hq3, p=2, dim=1)
            Q = Q3

        h1 = self.out_att1(Q)



        return F.log_softmax(h1, 1),h1


#####################SampleNCE  begin  ###################################



#########################  OBJ begin   #########################

class ADJConv(nn.Module):
    def __init__(self, in_channels, out_channels, para):
        super(ADJConv, self).__init__()

        self.step = para
        self.lin1 = nn.Linear(in_channels, out_channels, bias=False)
        self.lin2 = nn.Linear(in_channels, out_channels, bias=False)
        self.floop = nn.Linear(in_channels, out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.floop.reset_parameters()

    def forward(self, h1, adj):
        h3 = self.lin1(h1)
        h4 = self.lin2(h1)
        h1 = self.floop(h1)

        s = torch.mm(h3, h4.t())
        s = torch.sigmoid(-1*s)
        neg = 1 - s

        posx = torch.mul(s, adj)
        negx = torch.mul(neg, adj)
        h = torch.add(h1, torch.mm(posx, h3))
        h = torch.sub(h, torch.mm(negx, h4))

        return h



class OBJ(nn.Module):
    def __init__(self, dataset, args):
        super(OBJ, self).__init__()
        self.K = args.K
        self.dropout = args.dropout
        self.nfeat = dataset.num_features
        self.num_hidden=args.hidden

        self.para=args.belta


        self.layers1 = nn.ModuleList([ADJConv(self.num_hidden, self.num_hidden, self.para) for _ in range(args.K)])
        self.lin1 = nn.Linear(dataset.num_features, self.num_hidden)


        self.out_att1 = nn.Linear(self.num_hidden, dataset.num_classes)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()

        self.out_att1.reset_parameters()


    def forward(self, data):
        x, adj = data.x, data.adj
        Q = self.lin1(x)
        if self.dropout>0:

            Q = F.dropout(Q, p=self.dropout, training=self.training)

        for i in range(self.K):

            hq3 = self.layers1[i](Q, adj)
            Q3 = F.normalize(hq3, p=2, dim=1)
            Q = Q3

        h1 = self.out_att1(Q)

        return F.log_softmax(h1, 1),h1


#########################  OBJ end #########################


#####################SGC  begin  ###################################
class sgc_prop(MessagePassing):
    '''
    propagation class for GPR_GNN
    '''

    def __init__(self, K, bias=True, **kwargs):
        super(sgc_prop, self).__init__(aggr='add', **kwargs)
        self.K = K

    def forward(self, x, edge_index, edge_weight=None):
        list_mat = []


        # D^(-0.5)AD^(-0.5)
        edge_index, norm = gcn_norm(edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)

        for i in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm, size=None)
        return x

class SGC(torch.nn.Module):
    def __init__(self, dataset, args):
        super(SGC, self).__init__()
        self.dropout = args.dropout
        self.K = args.K
        self.prop = sgc_prop(self.K)
        self.lin1 = Linear(dataset.num_features, dataset.num_classes)


    def reset_parameters(self):
        self.prop.reset_parameters()
        self.lin1.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.prop(x, edge_index)
        x = self.lin1(x)
        return F.log_softmax(x, dim=1)


#####################SGC  end  ###################################