import torch.nn as nn
import sys
import torch.nn.functional as F
import math
import torch
import torch.optim as optim
import numpy as np
import scipy.sparse as sp
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from deeprobust.graph import utils
from copy import deepcopy
from torch_geometric.nn import GCNConv
from sklearn.metrics import f1_score
from scipy.sparse import lil_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

fwd_att = 3

class GraphConvolution(Module):
    """Simple GCN layer, similar to https://github.com/tkipf/pygcn
    """

    def __init__(self, in_features, out_features, with_bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if with_bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        """ Graph Convolutional Layer forward function
        """
        if input.data.is_sparse:
            support = torch.spmm(input, self.weight)
        else:
            support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    """ 2 Layer Graph Convolutional Network.

    Parameters
    ----------
    nfeat : int
        size of input feature dimension
    nhid : int
        number of hidden units
    nclass : int
        size of output dimension
    dropout : float
        dropout rate for GCN
    lr : float
        learning rate for GCN
    weight_decay : float
        weight decay coefficient (l2 normalization) for GCN.
        When `with_relu` is True, `weight_decay` will be set to 0.
    with_relu : bool
        whether to use relu activation function. If False, GCN will be linearized.
    with_bias: bool
        whether to include bias term in GCN weights.
    device: str
        'cpu' or 'cuda'.

    Examples
    --------
	We can first load dataset and then train GCN.

    >>> from deeprobust.graph.data import Dataset
    >>> from deeprobust.graph.defense import GCN
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    >>> gcn = GCN(nfeat=features.shape[1],
              nhid=16,
              nclass=labels.max().item() + 1,
              dropout=0.5, device='cpu')
    >>> gcn = gcn.to('cpu')
    >>> gcn.fit(features, adj, labels, idx_train) # train without earlystopping
    >>> gcn.fit(features, adj, labels, idx_train, idx_val, patience=30) # train with earlystopping
    >>> gcn.test(idx_test)
    """

    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4,
            with_relu=True, with_bias=True, device=None):

        super(GCN, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.dropout = dropout
        self.lr = lr
        
        # GCN  
        #self.gc1 = GraphConvolution(nfeat, nhid, bias=with_bias)
        #self.gc2 = GraphConvolution(nhid, nclass, bias=with_bias)
        self.gc1 = GCNConv(nfeat,nhid,bias=True)
        self.gc2 = GCNConv(nhid,nclass,bias=True)
        
        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
        self.with_relu = with_relu
        self.with_bias = with_bias
        self.output = None
        self.best_model = None
        self.best_output = None
        self.adj_norm = None
        self.features = None
    

    def double_forward(self, x, adj):
        x = x.to_dense()

        # grab final att first

        att, n_adj = self.gsl(x,adj) # applied gsl algorithm=
        edge_index = att._indices() # get edge index

        tx = self.gc1(x,edge_index,edge_weight = att._values()) #pass to layer 1
        tx = F.relu(tx)
        att, n_adj = self.gsl(tx,n_adj,call_time=1)
        att_d = att.to_dense()
        r,c = att_d.nonzero()[:,0],att_d.nonzero()[:,1] # get edge index r,c
        edge_index = torch.stack((r,c),dim=0) # combine to a array
        adj_values = att_d[r,c] # get values

        x = self.gc1(x,edge_index,edge_weight = adj_values) #pass to layer 1
        x = F.relu(x)
        x = F.dropout(x,self.dropout,training=self.training)
        x = self.gc2(x,edge_index,edge_weight = adj_values)

        return F.log_softmax(x,dim=1)

    def single_forward(self, x, adj):
        x = x.to_dense()

        # grab final att first

        att, n_adj = self.gsl(x,adj) # applied gsl algorithm=
        edge_index = att._indices() # get edge index

        x = self.gc1(x,edge_index,edge_weight = att._values()) #pass to layer 1
        x = F.relu(x)
        x = F.dropout(x,self.dropout,training=self.training)
        x = self.gc2(x,edge_index,edge_weight = att._values())

        return F.log_softmax(x,dim=1)

    def blind_gsl_forward(self, x):
        x = x.to_dense()

        n_adj = self.blind_gsl(x) # applied gsl algorithm
        edge_index = n_adj._indices() # get edge index
        x = self.gc1(x,edge_index,edge_weight = n_adj._values()) #pass to layer 1, n_adj is attention adjacency
        x = F.relu(x)

        
        n_adj = self.blind_gsl(x,call_time=1)

        att_d = n_adj.to_dense()
        r,c = att_d.nonzero()[:,0],att_d.nonzero()[:,1] # get edge index r,c
        edge_index = torch.stack((r,c),dim=0) # combine to a array
        adj_values = att_d[r,c] # get values

        

        x = F.dropout(x,self.dropout,training=self.training)
        x = self.gc2(x,edge_index,edge_weight = adj_values)

        return F.log_softmax(x,dim=1)


    def forward(self, x, adj):
        if fwd_att == 1:
            return self.single_forward(x, adj)
        if fwd_att == 2:
            return self.double_forward(x, adj)
        if fwd_att == 3:
            return self.blind_gsl_forward(x)

        x = x.to_dense()

        if self.attention: # use GSL
            att, n_adj = self.gsl(x,adj) # applied gsl algorithm
        else:
            att = adj
 
        edge_index = att._indices() # get edge index
        x = self.gc1(x,edge_index,edge_weight = att._values()) #pass to layer 1
        x = F.relu(x)

        if self.attention: # applied gsl algrithm
            att, n_adj = self.gsl(x,n_adj,call_time=1)

            att_d = att.to_dense()
            r,c = att_d.nonzero()[:,0],att_d.nonzero()[:,1] # get edge index r,c
            edge_index = torch.stack((r,c),dim=0) # combine to a array
            adj_values = att_d[r,c] # get values

        else:
            edge_index = adj._indices()
            adj_values = adj._values()

        x = F.dropout(x,self.dropout,training=self.training)
        x = self.gc2(x,edge_index,edge_weight = adj_values)

        return F.log_softmax(x,dim=1)
        '''
        if self.with_relu:
            x = F.relu(self.gc1(x, adj))
        else:
            x = self.gc1(x, adj)

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
        '''
    # ==================================================================

    def initialize(self):
        """Initialize parameters of GCN.
        """
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()


    # =============================== QUAN ==============================
    def gsl(self, features, adj, is_lil = False, call_time = 0):
        """
        features: features of nodes
        adj: adjacency matrix which include index and values
        is_lil: whether adj matrix is lil matrix or not
        i: i-th call this function
        """
        edge_index = None
        if is_lil:
            edge_index = adj.tocoo()
        else:
            edge_index = adj._indices()

        node_num = features.shape[0]

        r,c = edge_index[0].cpu().data.numpy()[:],edge_index[1].cpu().data.numpy()[:]
        
        features_cp = features.cpu().data.numpy()

        sim_matrix = cosine_similarity(X = features_cp, Y = features_cp) #calculate the sim between nodes
        sim = sim_matrix[r, c] # grab sim scores for existing esges in adj matrix


        # calculate the malicious score
        #m_score = abs(sim_matrix - adj.todense())
        m_score = abs(sim_matrix - utils.to_scipy(adj).tolil())


        # decide perturbation edge number
        ptb_edge_num = int(node_num*node_num*0.025) #node_num**2*0.05 
        ptb_edge_num = ptb_edge_num + ptb_edge_num%2

        #print("val of edges === {}\n".format(ptb_edge_num))
        #print("m_score == {}".format(m_score))
        #print(type(m_score))
        # function to get largest index
        # https://stackoverflow.com/questions/43386432/how-to-get-indexes-of-k-maximum-values-from-a-numpy-multidimensional-array

        def k_largest_index_argsort(a, k):
            idx = np.argsort(a.ravel())[:-k-1:-1]
            return np.column_stack(np.unravel_index(idx, a.shape))


        mal_index = k_largest_index_argsort(np.asarray(m_score), ptb_edge_num)
        #print(mal_index)
        trans_mal=[mal_index[:,0], mal_index[:,1]]
        #print(trans_mal)
        # build the new adjacency matrix
        n_adj = lil_matrix((node_num,node_num),dtype = np.float32) 
        #sys.exit()
        n_adj[r,c]=1
        n_adj[tuple(trans_mal)] = 1
        temp = lil_matrix((node_num,node_num),dtype = np.float32)
        #print(type(temp))
        #print("adj === ",type(adj))
        if type(adj) is torch.Tensor:
            adj = utils.to_scipy(adj).tolil()
        #print(type(adj))
        temp[tuple(trans_mal)] = adj[tuple(trans_mal)]
        n_adj = n_adj-temp
    
    
        
        # Build Attention matrix
        att = lil_matrix((node_num,node_num),dtype = np.float32) 
        sim[sim<0.3]=0
        att[r,c]=sim
        att[tuple(trans_mal)] = 1
        # The new weight is point-wise multiplied with modified Adjacency Matrix
        inf_weight = att.multiply(n_adj)
        #old_att = att - temp
        #assert ((inf_weight!=old_att).nnz==0)



        # the following approach is the attention aproach
        if inf_weight[0,0] == 1:
            inf_weight = inf_weight-sp.diags(inf_weight.diagonal(),offsets=0,format='lil')
        
        att_norm = normalize(inf_weight,axis=1,norm='l1')

        if att_norm[0,0] == 0:
            degree = (att_norm !=0).sum(1).A1
            lam = 1/(degree+1)
            self_weight = sp.diags(np.array(lam),offsets=0,format='lil')
            ret_att = att_norm + self_weight
        else:
            ret_att = att_norm

    #  Reformat n_adj to floattensor
        r1,c1 = n_adj.nonzero()
        r_adj = np.vstack((r1,c1))
        r_adj = torch.tensor(r_adj, dtype=torch.int64)
        v = n_adj[r1,c1]
        v = torch.tensor(np.ones(len(r1)),dtype = torch.int64)
        #print(r_adj,v)
        n_adj = torch.sparse.FloatTensor(r_adj,v,(node_num,node_num))
    
    #  Reformat att to floattensor
        row,col = ret_att.nonzero()
        att_adj = np.vstack((row,col))
        att_edge_w = ret_att[row,col]
        att_edge_w = np.exp(att_edge_w)
        att_edge_w = torch.tensor(np.array(att_edge_w)[0],dtype=torch.float32)
        att_adj = torch.tensor(att_adj,dtype=torch.int64)

        shape = (node_num,node_num)

        final_att = torch.sparse.FloatTensor(att_adj,att_edge_w,shape)

        return final_att.to(self.device), n_adj.to(self.device)

        
    # This is the graph structure learning alike approach
    # Assume we do not use Adjacency Matrix, we only have features
    # We reconstruct the graph only base on feature similarity
    def blind_gsl(self, features, is_lil = False, call_time = 0):
        """
        features: features of nodes
        adj: adjacency matrix which include index and values
        is_lil: whether adj matrix is lil matrix or not
        i: i-th call this function
        """
        

        node_num = features.shape[0]
        
        features_cp = features.cpu().data.numpy()

        sim_matrix = cosine_similarity(X = features_cp, Y = features_cp) #calculate the sim between nodes


        # decide perturbation edge number
        edge_num = int(node_num*node_num*0.025) #node_num**2*0.05 
        edge_num = edge_num + edge_num%2

        #print("val of edges === {}\n".format(ptb_edge_num))
        #print("m_score == {}".format(m_score))
        #print(type(m_score))
        # function to get largest index
        # https://stackoverflow.com/questions/43386432/how-to-get-indexes-of-k-maximum-values-from-a-numpy-multidimensional-array

        def k_largest_index_argsort(a, k):
            idx = np.argsort(a.ravel())[:-k-1:-1]
            return np.column_stack(np.unravel_index(idx, a.shape))


        edg_index = k_largest_index_argsort(np.asarray(sim_matrix), edge_num)
        #print(mal_index)
        trans_edg=[edg_index[:,0], edg_index[:,1]]
        #print(trans_mal)
        # build the new adjacency matrix
        n_adj = lil_matrix((node_num,node_num),dtype = np.float32) 
        n_adj[tuple(trans_edg)] = 1

        # TODO: We can use similarity as attention
        inf_weight = n_adj




        # the following approach is the attention aproach
        if inf_weight[0,0] == 1:
            inf_weight = inf_weight-sp.diags(inf_weight.diagonal(),offsets=0,format='lil')
        
        att_norm = normalize(inf_weight,axis=1,norm='l1')

        if att_norm[0,0] == 0:
            degree = (att_norm !=0).sum(1).A1
            lam = 1/(degree+1)
            self_weight = sp.diags(np.array(lam),offsets=0,format='lil')
            ret_att = att_norm + self_weight
        else:
            ret_att = att_norm

    
    
    #  Reformat att to floattensor
        row,col = ret_att.nonzero()
        att_adj = np.vstack((row,col))
        att_edge_w = ret_att[row,col]
        att_edge_w = np.exp(att_edge_w)
        att_edge_w = torch.tensor(np.array(att_edge_w)[0],dtype=torch.float32)
        att_adj = torch.tensor(att_adj,dtype=torch.int64)

        shape = (node_num,node_num)

        final_att = torch.sparse.FloatTensor(att_adj,att_edge_w,shape)

        return final_att.to(self.device)



    def fit(self, features, adj, labels, idx_train, idx_val=None, train_iters=200, attention=False, initialize=True, verbose=False, normalize=True, patience=500, **kwargs):
        """Train the gcn model, when idx_val is not None, pick the best model according to the validation loss.

        Parameters
        ----------
        features :
            node features
        adj :
            the adjacency matrix. The format could be torch.tensor or scipy matrix
        labels :
            node labels
        idx_train :
            node training indices
        idx_val :
            node validation indices. If not given (None), GCN training process will not adpot early stopping
        train_iters : int
            number of training epochs
        initialize : bool
            whether to initialize parameters before training
        verbose : bool
            whether to show verbose logs
        normalize : bool
            whether to normalize the input adjacency matrix.
        patience : int
            patience for early stopping, only valid when `idx_val` is given
        """
        self.attention = attention

        self.device = self.gc1.weight.device
        if initialize:
            self.initialize()

        if type(adj) is not torch.Tensor:
            features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        else:
            features = features.to(self.device)
            adj = adj.to(self.device)
            labels = labels.to(self.device)

        if normalize:
            if utils.is_sparse_tensor(adj):
                adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                adj_norm = utils.normalize_adj_tensor(adj)
        else:
            adj_norm = adj

        self.adj_norm = adj_norm
        self.features = features
        self.labels = labels

        if idx_val is None:
            self._train_without_val(labels, idx_train, train_iters, verbose)
        else:
            if patience < train_iters:
                self._train_with_early_stopping(labels, idx_train, idx_val, train_iters, patience, verbose)
            else:
                self._train_with_val(labels, idx_train, idx_val, train_iters, verbose)

    def _train_without_val(self, labels, idx_train, train_iters, verbose):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for i in range(train_iters):
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

        self.eval()
        output = self.forward(self.features, self.adj_norm)
        self.output = output

    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            output = self.forward(self.features, self.adj_norm)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)

    def _train_with_early_stopping(self, labels, idx_train, idx_val, train_iters, patience, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        early_stopping = patience
        best_loss_val = 100

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            output = self.forward(self.features, self.adj_norm)

            # def eval_class(output, labels):
            #     preds = output.max(1)[1].type_as(labels)
            #     return f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='micro') + \
            #         f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')

            # perf_sum = eval_class(output[idx_val], labels[idx_val])
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())
                patience = early_stopping
            else:
                patience -= 1
            if i > early_stopping and patience <= 0:
                break

        if verbose:
             print('=== early stopping at {0}, loss_val = {1} ==='.format(i, best_loss_val) )
        self.load_state_dict(weights)

    def test(self, idx_test):
        """Evaluate GCN performance on test set.

        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        output = self.predict()
        # output = self.output
        loss_test = F.nll_loss(output[idx_test], self.labels[idx_test])
        acc_test = utils.accuracy(output[idx_test], self.labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test, output


    def predict(self, features=None, adj=None):
        """By default, the inputs should be unnormalized adjacency

        Parameters
        ----------
        features :
            node features. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.
        adj :
            adjcency matrix. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.


        Returns
        -------
        torch.FloatTensor
            output (log probabilities) of GCN
        """

        self.eval()
        if features is None and adj is None:
            return self.forward(self.features, self.adj_norm)
        else:
            if type(adj) is not torch.Tensor:
                features, adj = utils.to_tensor(features, adj, device=self.device)

            self.features = features
            if utils.is_sparse_tensor(adj):
                self.adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                self.adj_norm = utils.normalize_adj_tensor(adj)
            return self.forward(self.features, self.adj_norm)



