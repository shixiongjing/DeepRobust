import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import Random
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
import argparse
from scipy.sparse.linalg import eigs
from scipy.sparse import csgraph,lil_matrix
import time
from scipy import spatial

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--dataset', type=str, default='citeseer', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.05,  help='pertubation rate')
parser.add_argument('--ptb_n', type=int, default=0,  help='pertubation number')


args = parser.parse_args()
args.cuda = torch.cuda.is_available()
print('cuda: %s' % args.cuda)
device = torch.device("cuda" if args.cuda else "cpu")

if args.seed == -1:
    np.random.seed(int(time.time()))
else:
    np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

data = Dataset(root='/tmp/', name=args.dataset)
adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
idx_unlabeled = np.union1d(idx_val, idx_test)



# Setup Attack Model
model = Random()

if args.ptb_n == 0:
    n_perturbations = int(args.ptb_rate * (adj.sum()//2))
else:
    n_perturbations = args.ptb_n
print('perturbation number: '+str(n_perturbations))
model.attack(adj, n_perturbations)
modified_adj = model.modified_adj

# adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False, sparse=True)
# adj = adj.to(device)
# features = features.to(device)
# labels = labels.to(device)

# modified_adj = normalize_adj(modified_adj)
# modified_adj = sparse_mx_to_torch_sparse_tensor(modified_adj)
# modified_adj = modified_adj.to(device)

def SpectralDistance(adj,m_adj):

    L_norm = csgraph.laplacian(adj)
    L_norm_m = csgraph.laplacian(m_adj)
    
    evals,evecs = np.linalg.eig(L_norm.todense())
    evals = evals.real
    
    m_evals, m_evecs = np.linalg.eig(L_norm_m.todense())
    m_evals = m_evals.real
    dif2 = sum(m_evals)-sum(evals)
    dif3 = np.linalg.norm(m_evals)-np.linalg.norm(evals)
    print(dif3)
    
    dif1 = (np.diag(evals)-np.diag(m_evals))
        
    S_Dis = np.linalg.norm(dif1)

    return S_Dis,dif2

def test(adj):
    ''' test on GCN '''
    # adj = normalize_adj_tensor(adj)
    gcn = GCN(nfeat=features.shape[1],
              nhid=16,
              nclass=labels.max().item() + 1,
              dropout=0.5, device=device)

    gcn = gcn.to(device)

    optimizer = optim.Adam(gcn.parameters(),
                           lr=0.01, weight_decay=5e-4)

    gcn.fit(features, adj, labels, idx_train) # train without model picking
    # gcn.fit(features, adj, labels, idx_train, idx_val) # train with validation model picking
    output = gcn.output
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

    return acc_test.item()


def main():
    # print('=== testing GCN on original(clean) graph ===')
    # test(adj)
    # print('=== testing GCN on perturbed graph ===')
    # test(modified_adj)

    S_Distance,eigv_dif = SpectralDistance(adj,modified_adj)
    print('n_ptb: {}, S_Dis: {}, Eigv_dif: {}'.format(n_perturbations, S_Distance, eigv_dif))
    with open(args.dataset+'_'+'Random_vs_Net.log','a+') as f:
        print('n_ptb: {}, S_Dis: {}, Eigv_dif: {}'.format(n_perturbations, S_Distance, eigv_dif), file=f)


if __name__ == '__main__':
    main()

