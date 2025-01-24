##########################################################
import dgl.function as fn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn import GraphConv
import copy
from typing import Optional, Tuple, Union

class LogReg(nn.Module):
    def __init__(self, hid_dim, out_dim):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        ret = self.fc(x)
        return ret


class MLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, use_bn=True):
        super(MLP, self).__init__()

        self.layer1 = nn.Linear(nfeat, nhid, bias=True)
        self.layer2 = nn.Linear(nhid, nclass, bias=True)

        self.bn = nn.BatchNorm1d(nhid)
        self.use_bn = use_bn
        self.act_fn = nn.ReLU()

    def forward(self, _, x):
        x = self.layer1(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.act_fn(x)
        x = self.layer2(x)

        return x
class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(target_ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = target_ema_updater.update_average(old_weight, up_weight)

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

class Predictor(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers):
        super(Predictor, self).__init__()
        self.linears = torch.nn.ModuleList()
        self.linears.append(nn.Linear(input_dim, output_dim))
        for layer in range(num_layers - 1):
            self.linears.append(nn.Linear(output_dim, output_dim))
        self.num_layers = num_layers

    def forward(self, embedding):
        h = embedding
        for layer in range(self.num_layers - 1):
            h = F.relu(self.linears[layer](h))
        h = self.linears[self.num_layers - 1](h)
        return h

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers):
        super(MLP, self).__init__()
        self.linears = torch.nn.ModuleList()
        self.linears.append(nn.Linear(input_dim, output_dim))
        for layer in range(num_layers - 1):
            self.linears.append(nn.Linear(output_dim, output_dim))
        self.num_layers = num_layers

    def forward(self, embedding):
        h = embedding
        for layer in range(self.num_layers - 1):
            h = self.linears[layer](h)
        h = self.linears[self.num_layers - 1](h)
        return h

def udf_u_add_log_e(edges):
    return {'m': torch.log(edges.dst['neg_sim'] + edges.data['sim'])}


#################################################################################
##########################   Encoder   ######################################

class GCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, use_ln=False):
        super().__init__()

        self.n_layers = n_layers
        self.convs = nn.ModuleList()
        self.convs.append(GraphConv(in_dim, hid_dim, norm='both'))
        self.use_ln = use_ln
        self.lns = nn.ModuleList()
        if n_layers > 1:
            for i in range(n_layers - 2):
                self.convs.append(GraphConv(hid_dim, hid_dim, norm='both'))
            for i in range(n_layers - 1):
                self.lns.append(nn.BatchNorm1d(hid_dim))
                # self.lns.append(nn.LayerNorm(hid_dim))
            self.convs.append(GraphConv(hid_dim, out_dim, norm='both'))

    def forward(self, graph, x):
        for i in range(self.n_layers - 1):
            if not self.use_ln:
                x = F.relu(self.convs[i](graph, x))
            else:
                x = F.relu(self.lns[i](self.convs[i](graph, x)))

        x = self.convs[-1](graph, x)

        return x

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(target_ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = target_ema_updater.update_average(old_weight, up_weight)

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

class Predictor(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers):
        super(Predictor, self).__init__()
        self.linears = torch.nn.ModuleList()
        self.linears.append(nn.Linear(input_dim, output_dim))
        for layer in range(num_layers - 1):
            self.linears.append(nn.Linear(output_dim, output_dim))
        self.num_layers = num_layers

    def forward(self, embedding):
        h = embedding
        for layer in range(self.num_layers - 1):
            h = F.relu(self.linears[layer](h))
        h = self.linears[self.num_layers - 1](h)
        return h

def udf_u_add_log_e(edges):
    return {'m': torch.log(edges.dst['neg_sim'] + edges.data['sim'])}

from models import choose_model
class DualModel(nn.Module):

    def __init__(self, graph, in_dim, hid_dim, out_dim,args):
        super(DualModel, self).__init__()
        self.moving_average_decay1= args.moving_average_decay1
        self.moving_average_decay2= args.moving_average_decay2
        self.num_layers=args.num_layers

        self.encoder1 =choose_model('GraphSAGE',graph,in_dim, hid_dim, out_dim, 2)   #GraphSAGE    #GCN(in_dim, hid_dim, out_dim,  self.num_layers)


        self.encoder1_target = copy.deepcopy(self.encoder1)
        set_requires_grad(self.encoder1_target, False)
        self.target_ema_updater1 = EMA(self.moving_average_decay1)
        self.encoder2 =MLP(in_dim, out_dim, self.num_layers)
        self.encoder2_target = copy.deepcopy(self.encoder2)
        set_requires_grad(self.encoder2_target, False)
        self.target_ema_updater2 = EMA(self.moving_average_decay2)

        self.num_MLP = args.num_MLP
        self.out_dim = out_dim
        self.projector = Predictor(out_dim, out_dim, self.num_MLP)

    def pos_score(self, graph, v, u, tau):
        if self.num_MLP == 0:
            graph.ndata['q'] = F.normalize(v)
        else:
            graph.ndata['q'] = F.normalize(self.projector(v))
        graph.ndata['u'] = F.normalize(u, dim=-1)
        graph.apply_edges(fn.u_mul_v('u', 'q', 'sim'))
        graph.edata['sim'] = graph.edata['sim'].sum(1) / tau
        graph.update_all(fn.copy_e('sim', 'm'), fn.mean('m', 'pos'))
        pos_score = graph.ndata['pos']

        return pos_score, graph

    def neg_score(self, h, s, graph, tau):
        z = F.normalize(h, dim=-1)
        stu = F.normalize(s, dim=-1)

        graph.edata['sim'] = torch.exp(graph.edata['sim'])
        neg_sim_intra = torch.exp(torch.mm(z, z.t()) / tau)
        neg_score_intra = neg_sim_intra.sum(1)
        neg_sim_inter = torch.exp(torch.mm(z, stu.t()) / tau)
        neg_score_inter = neg_sim_inter.sum(1)
        graph.ndata['neg_sim'] = neg_score_intra + neg_score_inter

        graph.update_all(udf_u_add_log_e, fn.mean('m', 'neg'))
        neg_score = graph.ndata['neg']
        return neg_score

    def update_moving_average(self):
        assert self.encoder1_target is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater1, self.encoder1_target, self.encoder1)
        assert self.encoder2_target is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater2, self.encoder2_target, self.encoder2)

    def forward(self, graph, g, feat, tau1, tau2):
        v1 = self.encoder1(graph, feat)
        u1 = self.encoder1_target(graph, feat)
        pos_score1, _ = self.pos_score(g, v1, u1, tau1)
        neg_score1 = self.neg_score(v1, u1, g, tau1)
        loss1 = (- pos_score1 + neg_score1).mean()

        v2 = self.encoder2(feat)
        u2 = self.encoder2_target(feat)
        pos_score2, _ = self.pos_score(g, v2, u2, tau2)
        neg_score2 = self.neg_score(v2, u2, g, tau2)
        loss2 = (- pos_score2 + neg_score2).mean()
        loss=loss1 + loss2
        return v1, u1, v2, u2,loss

    def get_embedding(self, graph,g, feat, norm):
        h1 = self.encoder1(graph,feat)
        h2 = self.encoder2(feat)
        if norm==1:
            h1=F.normalize(h1, dim=-1)
            h2=F.normalize(h2, dim=-1)

        return h1.detach(),h2.detach()


