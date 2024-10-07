import torch
from torch.nn.parameter import Parameter
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.nn import Linear
import scipy.sparse as sp
import torch.nn as nn
import torch.nn.functional as F



import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import MessagePassing




#########################  GBK begin   #########################

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




class EdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels, step, edge_index):
        super(EdgeConv, self).__init__(aggr='add',flow="source_to_target")  # "Add" aggregation (Step 5).

        self.step= step
        self.lin1 = nn.Linear(in_channels, out_channels, bias=False)
        self.lin2 = nn.Linear(in_channels, out_channels, bias=False)
        self.floop = nn.Linear(in_channels, out_channels, bias=False)

        self.edge_index=edge_index
        self.row, self.col = self.edge_index
        self.edge =torch.vstack([self.col, self.row])
        #self.NEG_index=NEG_index
        #self.rowN, self.colN = self.NEG_index
        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
    # def pospropagate(self,numnode,ss,  h):
    #     s = torch.sum(ss, dim=1)
    #     s = torch.sigmoid(-1 * s)
    #     h = self.propagate(self.edge_index, size=(numnode, numnode), x=h, norm=s, flow='source_to_target')
    #     return h
    #
    # def negpropagate(self, numnode, ss, h):
    #     s = torch.sum(ss, dim=1)
    #     s = torch.sigmoid(s)
    #     h= self.propagate(self.NEG_index, size=(numnode, numnode), x=h, norm=s, flow='source_to_target')
    #     return h
    def forward(self, h1):
        h3 =self.lin1(h1)
        h4 =self.lin2(h1)
        numnode=h1.shape[0]
        h1 = self.floop(h1)


        ss=torch.mul(h3[self.row], h4[self.col])
        s = torch.sum(ss, dim=1)
        s = torch.sigmoid(-1 * s)
        h = self.propagate(self.edge_index, size=(numnode, numnode), x=h3, norm=s, flow='source_to_target')
        h3=torch.add(h1, self.step*h)

        h = self.propagate(self.edge_index, size=(numnode, numnode), x=h4, norm=1-s, flow='source_to_target')
        x = torch.sub(h3, self.step*h4)


        return x

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out


class GBK(nn.Module):
    def __init__(self, dataset, data, args):
        super(GBK, self).__init__()
        self.K = args.K
        self.dropout = args.dropout
        self.nfeat = dataset.num_features
        self.num_hidden=args.hidden

        self.para = args.alpha
        self.chan = args.chan

        self.act = args.activation



        self.layers1 = nn.ModuleList([EdgeConv(self.num_hidden, self.num_hidden, self.para,  data.edge_index) for _ in range(args.K)])
        self.lin1 = nn.Linear(dataset.num_features, self.num_hidden)


        self.out_att1 = nn.Linear(self.num_hidden, dataset.num_classes)
        self.out_att2 = nn.Linear(2*self.num_hidden, dataset.num_classes)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()

        self.out_att1.reset_parameters()
        self.out_att2.reset_parameters()

    def forward(self, data, adj1, adj2):
        x, edge_index = data.x, data.edge_index
        if self.act== 'None':
            Q = self.lin1(x)
        else:
            self.actf = getattr(F, self.act)
            Q = self.actf(self.lin1(x))
        # Q = F.dropout(Q1, p=self.dropout, training=self.training)
        Q2 = Q3= Q


        for i in range(self.K):

            hq3 = self.layers1[i](Q)
            Q3 = F.normalize(hq3, p=2, dim=1)
            Q = Q3

        if self.chan ==1:
            h1 = self.out_att1(Q)
        elif self.chan ==2:
            h1 = self.out_att2(torch.cat([Q2, Q3],dim=1))


        return F.log_softmax(h1, 1)


#########################  GBK end #########################



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

    def forward(self, h1, h4, adj):
        h3 = self.lin1(h1)
        h4 = self.lin2(h1)

        h1 = self.floop(h1)

        s = torch.mm(h3, h4.t())
        s = torch.sigmoid(-1*s)
        neg = 1 - s

        posx = torch.mul(s, adj)
        negx = torch.mul(neg.t(), adj)

        h4=torch.add(h1, torch.mm(negx, h4))
        h = torch.add(h1, torch.mm(posx, h3))
        h = torch.sub(h, h4)

        return h, h4




class StepConv(MessagePassing):
    def __init__(self, in_channels, out_channels, step, edge_index, act):
        super(StepConv, self).__init__(aggr='add',flow="source_to_target")  # "Add" aggregation (Step 5).

        self.step= step
        self.act=act
        self.lin1 = nn.Linear(in_channels, out_channels, bias=False)
        self.lin2 = nn.Linear(in_channels, out_channels, bias=False)
        self.floop = nn.Linear(in_channels, out_channels, bias=False)

        self.classify = nn.Linear(in_channels, out_channels, bias=False)

        self.edge_index=edge_index
        self.row, self.col = self.edge_index
        self.edge =torch.vstack([self.col, self.row])
        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.classify.reset_parameters()

    def forward(self, h1):

        h3 =self.lin1(h1)
        h4 =self.lin2(h1)
        numnode=h1.shape[0]

        if self.act == 'None':
            h1 = self.floop(h1)
        else:
            self.actf = getattr(F, self.act)
            h1 = self.actf(self.floop(h1))

        ss=torch.mul(h3[self.row], h4[self.col])
        s = torch.sum(ss, dim=1)
        s = torch.sigmoid(-1 * s)
        h = self.propagate(self.edge_index, size=(numnode, numnode), x=h3, norm=s, flow='source_to_target')
        h3=torch.add(h1, h)

        h4 = self.propagate(self.edge_index, size=(numnode, numnode), x=h4, norm=1-s, flow='source_to_target')
        x = torch.sub(h3, h4)
        if self.step==0:

            C=x
        else:
            x=h1
            #step 2 +STEP3
            similarity =torch.sigmoid(torch.mm(x,x.t()))
            as1=torch.mul(torch.sign(similarity-0.5),adj)
            as2 = torch.mul(torch.sign(0.5-similarity), adj)

            y1 =x+ (self.step*torch.mul((1-similarity), as1))@x
            y2 =x- (self.step * torch.mul(similarity, as2)) @ x
            x =y1+y2
            C = self.classify(x)


        return C

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out


class CMPGNN(nn.Module):
    def __init__(self, dataset, data,args):
        super(CMPGNN, self).__init__()
        self.K = args.K
        self.dropout = args.dropout
        self.nfeat = dataset.num_features
        self.num_hidden=args.hidden
        self.para = args.step

        self.layerNorm=args.Norm
        self.act = args.activation



        self.layers1 = nn.ModuleList([StepConv(self.num_hidden, self.num_hidden, self.para,data.edge_index, self.act) for _ in range(args.K)])
        self.lin1 = nn.Linear(dataset.num_features, self.num_hidden)
        self.out_att1 = nn.Linear(self.num_hidden, dataset.num_classes)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.out_att1.reset_parameters()


    def forward(self, data):#, adj
        x, edge_index = data.x, data.edge_index

        Q = self.lin1(x)
        if self.dropout>0:
            Q = F.dropout(Q, p=self.dropout, training=self.training)

        for i in range(self.K):
            hq3 = self.layers1[i](Q)
            Q3 = F.normalize(hq3, p=2, dim=1)
            if  self.layerNorm==1:
                Q = Q3

        h1 = self.out_att1(Q)

        return F.log_softmax(h1, 1)


#########################  OBJ  end #########################