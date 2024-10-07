
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax, add_remaining_self_loops
from torch_geometric.nn import GCNConv, SAGEConv, ChebConv
from torch.nn import Linear, Parameter
from torch_geometric.utils import add_self_loops, degree
import numpy as np
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import get_laplacian
from scipy.special import comb
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, ChebConv
from torch_geometric.nn import MessagePassing, APPNP




class GPR_prop(MessagePassing):
    '''
    propagation class for GPR_GNN
    '''

    def __init__(self, K, alpha, Init, Gamma=None, bias=True, **kwargs):
        super(GPR_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            # SGC-like
            TEMP = 0.0*np.ones(K+1)
            TEMP[-1] = 1.0
        elif Init == 'PPR':
            # PPR-like
            TEMP = alpha*(1-alpha)**np.arange(K+1)
            TEMP[-1] = (1-alpha)**K
        elif Init == 'NPPR':
            # Negative PPR
            TEMP = (alpha)**np.arange(K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'Random':
            # Random
            bound = np.sqrt(3/(K+1))
            TEMP = np.random.uniform(-bound, bound, K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'WS':
            # Specify Gamma
            TEMP = Gamma

        self.temp = Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        for k in range(self.K+1):
            self.temp.data[k] = self.alpha*(1-self.alpha)**k
        self.temp.data[-1] = (1-self.alpha)**self.K

    def forward(self, x, edge_index, edge_weight=None):
        gamma = torch.relu(self.temp)
        edge_index, norm = gcn_norm(edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)

        #hidden = x*(self.temp[0])
        hidden = x * (gamma[0])
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            #gamma = self.temp[k+1]
            hidden = hidden + gamma[k+1]*x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)

class GPRGNN(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GPRGNN, self).__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)

        self.prop1 = GPR_prop(args.K, args.alpha, args.Init)

        self.Init = args.Init
        self.dprate = args.dprate
        self.dropout = args.dropout

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = F.dropout(x, p=self.dprate, training=self.training)
        x = self.prop1(x, edge_index)
        return F.log_softmax(x, dim=1)


###################   BernNet  begin ###########################

class Bern_prop(MessagePassing):
    def __init__(self, K, bias=True, **kwargs):
        super(Bern_prop, self).__init__(aggr='add', **kwargs)

        self.K = K
        self.temp = Parameter(torch.Tensor(self.K + 1))
        self.reset_parameters()

    def reset_parameters(self):
        self.temp.data.fill_(1)

    def forward(self, x, edge_index, edge_weight=None):
        TEMP = F.relu(self.temp)

        # L=I-D^(-0.5)AD^(-0.5)
        edge_index1, norm1 = get_laplacian(edge_index, edge_weight, normalization='sym', dtype=x.dtype,
                                           num_nodes=x.size(self.node_dim))
        # 2I-L
        edge_index2, norm2 = add_self_loops(edge_index1, -norm1, fill_value=2., num_nodes=x.size(self.node_dim))

        tmp = []
        tmp.append(x)
        for i in range(self.K):
            x = self.propagate(edge_index2, x=x, norm=norm2, size=None)
            tmp.append(x)

        out = (comb(self.K, 0) / (2 ** self.K)) * TEMP[0] * tmp[self.K]

        for i in range(self.K):
            x = tmp[self.K - i - 1]
            x = self.propagate(edge_index1, x=x, norm=norm1, size=None)
            for j in range(i):
                x = self.propagate(edge_index1, x=x, norm=norm1, size=None)

            out = out + (comb(self.K, i + 1) / (2 ** self.K)) * TEMP[i + 1] * x
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)

class BernNet(torch.nn.Module):
    def __init__(self,dataset, args):
        super(BernNet, self).__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)
        self.m = torch.nn.BatchNorm1d(dataset.num_classes)
        self.prop1 = Bern_prop(args.K)
        #args.nummodes
        self.dprate = args.dprate
        self.dropout = args.dropout

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)


        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)

###################   BernNet end  ###########################
####################################GCN begin####################################


class GCN(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, args.hidden)
        self.conv2 = GCNConv(args.hidden, dataset.num_classes)
        self.gcs = nn.ModuleList([GCNConv(args.hidden, args.hidden) for _ in range(args.K - 1)])  # there需要一个
        self.dropout = args.dropout
        self.K = args.K

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        for i in range(self.K - 1):
            x = F.relu(self.gcs[i](x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

####################################GCN end####################################


######################################Graphsage begin##############################################

class GraphSage(torch.nn.Module):
    def __init__(self,dataset, args):
        super(GraphSage, self).__init__()

        self.conv1 = SAGEConv(dataset.num_features, args.hidden)
        self.gcs = nn.ModuleList([SAGEConv(args.hidden, args.hidden) for _ in range(args.K - 2)])  # there需要一个
        self.conv2 = SAGEConv(args.hidden, dataset.num_classes)
        self.dropout = args.dropout
        self.K = args.K

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        for i in range(self.K - 2):
            x = F.relu(self.gcs[i](x, edge_index))
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

######################################Graphsage end ##############################################





####################################FAGCN   begin ####################################


class FALayer(MessagePassing):
    def __init__(self,num_hidden, dropout):
        super(FALayer, self).__init__(aggr='add')

        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(2 * num_hidden, 1)
        nn.init.xavier_normal_(self.gate.weight, gain=1.414)

    def forward(self, h, edge_index):

        self.row, self.col = edge_index
        self.norm_degree = degree(self.row, num_nodes=h.shape[0]).clamp(min=1)
        self.norm_degree = torch.pow(self.norm_degree, -0.5)

        h2 = torch.cat([h[self.row], h[self.col]], dim=1)
        g = torch.tanh(self.gate(h2)).squeeze()
        norm = g * self.norm_degree[self.row] * self.norm_degree[self.col]
        # print('norm')
        # print(norm.shape)   torch.Size([10556])
        norm = self.dropout(norm)
        return self.propagate(edge_index, size=(h.size(0), h.size(0)), x=h, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1,1) * x_j

    def update(self, aggr_out):
        return aggr_out



class FAGCN(nn.Module):
    def __init__(self, dataset, args):
        super(FAGCN, self).__init__()
        self.eps = args.eps
        self.layer_num = args.K
        self.dropout = args.dropout
        self.layers = nn.ModuleList()
        self.num_hidden=args.hidden
        for i in range(self.layer_num):
            self.layers.append(FALayer(self.num_hidden, self.dropout))
        self.t1 = nn.Linear(dataset.num_features, self.num_hidden)
        self.t2 = nn.Linear(self.num_hidden, dataset.num_classes)


        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.t1.weight, gain=1.414)
        nn.init.xavier_normal_(self.t2.weight, gain=1.414)

    def forward(self,  data):
        h, edge_index = data.x, data.edge_index
        if self.dropout==0.0:
            h = torch.relu(self.t1(h))
        else:
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = torch.relu(self.t1(h))
            h = F.dropout(h, p=self.dropout, training=self.training)
        raw = h
        for i in range(self.layer_num):
            h = self.layers[i](h, edge_index)
            h = self.eps * raw + h
        h = self.t2(h)

        return F.log_softmax(h, 1)

####################################FAGCN   end ####################################



class GCN_Net(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GCN_Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, args.hidden)
        self.conv2 = GCNConv(args.hidden, dataset.num_classes)
        self.gcs = nn.ModuleList([GCNConv(args.hidden, args.hidden) for _ in range(args.K - 2)])  # there需要一个
        self.dropout = args.dropout
        self.K = args.K

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        for i in range(self.K - 2):
            x = F.relu(self.gcs[i](x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class ChebNet(torch.nn.Module):
    def __init__(self, dataset, args):
        super(ChebNet, self).__init__()
        self.conv1 = ChebConv(dataset.num_features, 32, K=2)
        self.conv2 = ChebConv(32, dataset.num_classes, K=2)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GAT_Net(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GAT_Net, self).__init__()
        self.conv1 = GATConv(
            dataset.num_features,
            args.hidden,
            heads=args.heads,
            dropout=args.dropout)
        self.convs = nn.ModuleList([GATConv(args.hidden * args.heads, args.hidden * args.heads, heads=args.output_heads, concat=False,dropout=args.dropout) for _ in range(args.K-2)])
        self.conv2 = GATConv(
            args.hidden * args.heads,
            dataset.num_classes,
            heads=args.output_heads,
            concat=False,
            dropout=args.dropout)
        self.dropout = args.dropout
        self.K = args.K

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        for i in range(self.K - 2):
            x = F.elu(self.convs[i](x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class APPNP_Net(torch.nn.Module):
    def __init__(self, dataset, args):
        super(APPNP_Net, self).__init__()
        nfeat=dataset.num_features
        self.lin1 = Linear(nfeat, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)
        self.prop1 = APPNP(args.K, args.alpha)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)
        return F.log_softmax(x, dim=1)

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

######################SGC  end##################################
#####################################################################################################################################################


######################Mixhop  begin##################################
class Mixprop(MessagePassing):

    def __init__(self, K, bias=True, **kwargs):
        super(Mixprop, self).__init__(aggr='add', **kwargs)
        self.K = K



    def forward(self, x, edge_index):
        list_mat = []
        list_mat.append(torch.relu(x))
        # D^(-0.5)AD^(-0.5)
        edge_index, norm = gcn_norm(edge_index, edge_weight=None, num_nodes=x.size(0), dtype=x.dtype)

        for i in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm, size=None)
            list_mat.append(torch.relu(x))
        final_mat = torch.cat(list_mat, dim=1)
        return final_mat

class Mixhop(torch.nn.Module):
    def __init__(self, dataset, args):
        super(Mixhop, self).__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(3*args.hidden, dataset.num_classes)
        self.dropout = args.dropout
        self.K = args.K
        self.prop = Mixprop(self.K)



    def reset_parameters(self):
        self.prop.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.lin1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.prop(x, edge_index)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)

######################Mixhop  end##################################
from torch_geometric.nn import JumpingKnowledge
class JKNet(torch.nn.Module):
    def __init__(self, dataset, args):
        in_channels = dataset.num_features
        out_channels = dataset.num_classes

        super(JKNet, self).__init__()
        self.conv1 = GCNConv(in_channels, args.hidden)
        self.conv2 = GCNConv(args.hidden, args.hidden)
        self.lin1 = torch.nn.Linear(args.hidden, out_channels)
        # self.one_step = APPNP(K=1, alpha=0)
        # self.JK = JumpingKnowledge(mode='lstm',
        #                            channels=16,
        #                            num_layers=4
        #                            )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x1 = F.relu(self.conv1(x, edge_index))
        x1 = F.dropout(x1, p=0.5, training=self.training)

        x2 = F.relu(self.conv2(x1, edge_index))
        x2 = F.dropout(x2, p=0.5, training=self.training)

        x = x1+x2

        x = self.lin1(x)
        return F.log_softmax(x, dim=1)



import torch_sparse
from torch import FloatTensor

# feat_dim: int,
# hidden_dim: int,
# class_dim: int,
# k: int = 2,
# dropout: float = 0.5,
# use_relu: bool = True
class H2GCN(nn.Module):
    def __init__(self,dataset, args):
        super(H2GCN, self).__init__()
        self.dropout = args.dropout
        self.k = args.K
        feat_dim =dataset.num_features

        self.act = F.relu
        self.use_relu = True
        self.w_embed = nn.Parameter(
            torch.zeros(size=(feat_dim, args.hidden)),
            requires_grad=True
        )
        self.w_classify = nn.Parameter(torch.zeros(size=((2 ** (self.k + 1) - 1) * args.hidden, dataset.num_classes)),
            requires_grad=True
        )
        self.params = [self.w_embed, self.w_classify]
        self.initialized = False
        self.a1 = None
        self.a2 = None
        self.reset_parameter()

    def reset_parameter(self):
        nn.init.xavier_uniform_(self.w_embed)
        nn.init.xavier_uniform_(self.w_classify)

    @staticmethod
    def _indicator(sp_tensor: torch.sparse.Tensor) -> torch.sparse.Tensor:
        csp = sp_tensor.coalesce()
        return torch.sparse_coo_tensor(
            indices=csp.indices(),
            values=torch.where(csp.values() > 0, 1, 0),
            size=csp.size(),
            dtype=torch.float
        )

    @staticmethod
    def _spspmm(sp1: torch.sparse.Tensor, sp2: torch.sparse.Tensor) -> torch.sparse.Tensor:
        assert sp1.shape[1] == sp2.shape[0], 'Cannot multiply size %s with %s' % (sp1.shape, sp2.shape)
        sp1, sp2 = sp1.coalesce(), sp2.coalesce()
        index1, value1 = sp1.indices(), sp1.values()
        index2, value2 = sp2.indices(), sp2.values()
        m, n, k = sp1.shape[0], sp1.shape[1], sp2.shape[1]
        indices, values = torch_sparse.spspmm(index1, value1, index2, value2, m, n, k)
        return torch.sparse_coo_tensor(
            indices=indices,
            values=values,
            size=(m, k),
            dtype=torch.float
        )

    @classmethod
    def _adj_norm(cls, adj: torch.sparse.Tensor) -> torch.sparse.Tensor:
        n = adj.size(0)
        d_diag = torch.pow(torch.sparse.sum(adj, dim=1).values(), -0.5)
        d_diag = torch.where(torch.isinf(d_diag), torch.full_like(d_diag, 0), d_diag)
        d_tiled = torch.sparse_coo_tensor(
            indices=[list(range(n)), list(range(n))],
            values=d_diag,
            size=(n, n)
        )
        return cls._spspmm(cls._spspmm(d_tiled, adj), d_tiled)

    def _prepare_prop(self, adj):
        n = adj.size(0)
        device = adj.device
        self.initialized = True
        sp_eye = torch.sparse_coo_tensor(
            indices=[list(range(n)), list(range(n))],
            values=[1.0] * n,
            size=(n, n),
            dtype=torch.float
        ).to(device)
        # initialize A1, A2
        a1 = self._indicator(adj - sp_eye)
        a2 = self._indicator(self._spspmm(adj, adj) - adj - sp_eye)
        # norm A1 A2
        self.a1 = self._adj_norm(a1)
        self.a2 = self._adj_norm(a2)

    def forward(self, adj: torch.sparse.Tensor, x: FloatTensor) -> FloatTensor:
        if not self.initialized:
            self._prepare_prop(adj)
        # H2GCN propagation
        rs = [self.act(torch.mm(x, self.w_embed))]
        for i in range(self.k):
            r_last = rs[-1]
            r1 = torch.spmm(self.a1, r_last)
            r2 = torch.spmm(self.a2, r_last)
            rs.append(self.act(torch.cat([r1, r2], dim=1)))
        r_final = torch.cat(rs, dim=1)
        r_final = F.dropout(r_final, self.dropout, training=self.training)
        x=torch.mm(r_final, self.w_classify)
        return torch.softmax(x, dim=1)