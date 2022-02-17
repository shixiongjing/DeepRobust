import torch
import argparse
import scipy 
import numpy as np
import pickle
from deeprobust.graph.targeted_attack import Nettack
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
from deeprobust.graph.defense import *
from sklearn.preprocessing import normalize
from tqdm import tqdm
from scipy.sparse.linalg import eigs
from scipy.sparse import csgraph,lil_matrix
from scipy import spatial

# arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type = str, default = "cora",choices = ["cora","citeseer"],help="dataset")
parser.add_argument("--defense", type = bool, default = False, help="defense or not") # with --defense flag, the value of flag is true
parser.add_argument("--model", type = str, default = "GCN", choices= ["GCN","GAT","GIN"])
parser.add_argument("--debug", type = bool, default = True, choices= [True,False])
parser.add_argument("--seed", type = int, default = 29, help="Random Seed" )
parser.add_argument("--direct", action = "store_false", help = "direct attack / influence attack") # with --direct flag, the val of flag is false


args = parser.parse_args()
args.cuda = torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.direct:
    influencers = 0
else:
    influencers = 5

if args.cuda:
    torch.cuda.manual_seed(args.cuda)

if args.debug:
    print('cuda :: {}\ndataset :: {}\nDefense Algo :: {}\nmodel :: {}\nDirect attack :: {}'.format(args.cuda, args.dataset, args.defense, args.model, args.direct))

#get data from deeprobust/Dataset
data = Dataset(root='/tmp/',name=args.dataset)

#adj matrix, features, labels
adj, features, labels = data.adj, data.features, data.labels

#train,test sets
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

adj=adj+adj.T
adj[adj>1] = 1

#setup surrogate model
surrogate=GCN(nfeat = features.shape[1], nclass = labels.max().item()+1, nhid = 16, dropout = 0, with_relu = False, 
        with_bias = False, device = device).to(device)

surrogate.fit(features, adj, labels, idx_train, idx_val, patience = 30, train_iters = 100)
"""
features:
    features of nodes
adj:
    adjacency matrix
labels:
    labels
patience:
    patience for early stopping (valid when val is given)
train_iters:
    epochs
"""


# setup attack model
target_node = 384 #1554
model = Nettack(surrogate, nnodes = adj.shape[0], attack_structure = True, attack_features = False, device = device).to(device)

#set defense
defense = args.defense


def main():
    degrees = adj.sum(0).A1
    print('index ::', np.where(degrees == max(degrees)))
    per_num = int(degrees[target_node])

    if args.debug:
        print('degrees (# of perturbations) :: {}'.format(per_num))

    model.attack(features, adj, labels, target_node, per_num, direct = args.direct, n_influencers = influencers)
    m_adj = model.modified_adj
    m_features = model.modified_features

    #S_D_Clean = SpectralDistance(adj,adj)
    #print(S_D_Clean)

    #S_D_Same = SpectralDistance(m_adj,m_adj)
    #print(S_D_Same)
    #print(adj.shape)
    S_Distance,eigv_dif = SpectralDistance(adj,m_adj)
    #print(S_Distance)
    #dif = m_adj-adj
    #for r,c in zip(*dif.nonzero()):
    #    print(r,c,dif[r,c])
    print(S_Distance)


def SpectralDistance(adj,m_adj):

    #I = lil_matrix(np.eye(adj.shape[0])) 
    L_norm = csgraph.laplacian(adj)
    L_norm_m = csgraph.laplacian(m_adj)
    
    evals,evecs = np.linalg.eig(L_norm.todense())
    evals = evals.real
    #print(evals)
    print(evecs.shape)
    
    m_evals, m_evecs = np.linalg.eig(L_norm_m.todense())
    m_evals = m_evals.real

    evec_dif = evecs - m_evecs

    print("Evec difference:")

    print(evec_dif)
    print("================")
    
    #dif = (evals-m_evals)
    dif2 = sum(m_evals)-sum(evals)
    dif3 = np.linalg.norm(m_evals)-np.linalg.norm(evals)
    #print(dif2)
    #np.set_printoptions(threshold=np.inf)
    #with open('Eigenvalus.log','a+') as f:
    #    print(dif2,file=f)
        #print(dif,file=f)
    

    #L_norm = csgraph.laplacian(np.diag(evals))
    #L_norm_m = csgraph.laplacian(np.diag(m_evals))
    
    #dif = (L_norm - L_norm_m)

    #dif = (np.diag(evals)-np.diag(evals))

    #print(np.linalg.norm(dif,axis=1))
    
    dif1 = (np.diag(evals)-np.diag(m_evals))
    

    """
    Dis here is the difference of eigenvalues:
    """
    #d = evals - m_evals
    #Dis = {dis:idx for idx,dis in enumerate(d) if dis>1}

    #print(Dis)
        
    S_Dis = np.linalg.norm(dif1)
    #print(S_Dis)

    #Dis = {d:idx for idx,d in enumerate(S_Dis) if d>=1}
    #Dis = sorted(Dis,reverse=True)
    #print(Dis)
    #print(len(Dis))
    #print(np.where(S_Dis == max(S_Dis)))
    #dif = evals-m_evals

    return S_Dis, dif2, evec_dif
"""
    print("=======test on clean adj===================")
    print("without defense :: ")
    test(adj, features, target_node,defense_al=False)
    print("with defense (with default setting):: ")
    test(adj, features, target_node, defense_al = defense)

    print("================ test on perturbed adj =================")
    print("without defense ::")
    test(m_adj, m_features, target_node,defense_al=False)
    print("with defense (with default setting)::")
    test(m_adj, m_features, target_node, defense_al = defense)


def test(adj, features, target, defense_al=False):

    target_model = globals()[args.model](nfeat = features.shape[1], nhid = 16, nclass = labels.max().item()+1, dropout = 0.5, device = device)
    target_model = target_model.to(device)

    target_model.fit(features, adj, labels, idx_train, idx_val=idx_val, attention = defense_al)

    target_model.eval()

    _, output = target_model.test(idx_test=idx_test)

    probs = torch.exp(output[[target_node]])[0]

    print('probs: {}'.format(probs.detach().cpu().numpy()))

    acc_test = accuracy(output[idx_test], labels[idx_test])

    print('Test set accuracy:',
            "accuracy = {:.4f}".format(acc_test.item()))


    return acc_test.item()
"""
def multi_evecs():
    cnt = 0
    degrees = adj.sum(0).A1
    node_list = select_nodes(num_target=10)
    print(node_list)
    a = []
    num = len(node_list)
    print('=== Attacking %s nodes respectively ===' % num)
    for target_node in tqdm(node_list):
        n_perturbations = int(degrees[target_node])
        if n_perturbations <1:  # at least one perturbation
            continue

        model = Nettack(surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=False, device=device)
        model = model.to(device)
        model.attack(features, adj, labels, target_node, n_perturbations, direct=args.direct, n_influencers = influencers, verbose=False)
        modified_adj = model.modified_adj
        modified_features = model.modified_features

        S_Dis, sum_eigv_dif, evec_dif = SpectralDistance(adj,modified_adj)
        a.append(evec_dif.flatten())
    

    mean = np.mean(a, axis=0)
    var = np.var(a, axis=0)
    print('Mean:{}, Var:{}'.format(mean, var))


def multi_test():
    cnt = 0
    degrees = adj.sum(0).A1
    node_list = select_nodes(num_target=10)
    print(node_list)

    num = len(node_list)
    print('=== Attacking %s nodes respectively ===' % num)
    num_tar = 0
    for target_node in tqdm(node_list):
        n_perturbations = int(degrees[target_node])
        if n_perturbations <1:  # at least one perturbation
            continue

        model = Nettack(surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=False, device=device)
        model = model.to(device)
        model.attack(features, adj, labels, target_node, n_perturbations, direct=args.direct, n_influencers = influencers, verbose=False)
        modified_adj = model.modified_adj
        modified_features = model.modified_features

        S_Dis, sum_eigv_dif = SpectralDistance(adj,modified_adj)
        print(target_node,'::',S_Dis)
        with open(args.dataset+'_'+args.model+'_SpectralDistance_sum.log','a+') as f:
            print('Target Node: {}, S_Dis: {}, Eigv_dif: {}'.format(target_node,S_Dis,sum_eigv_dif),file=f)

        """
        acc = single_test(modified_adj, modified_features, target_node)
        if acc == 0:
            cnt += 1
        num_tar += 1
        with open(args.dataset+"_"+args.model+"_gsl.log","a+") as f:
            print('classification rate : %s' % (1-cnt/num_tar), '# of targets:',num_tar,file=f)
        print('classification rate : %s' % (1-cnt/num_tar), '# of targets:', num_tar)
        """

def single_test(adj, features, target_node):
    'ALL the baselines'

    # """defense models"""
    # classifier = globals()[args.defensemodel](nnodes=adj.shape[0], nfeat=features.shape[1], nhid=16,
    #                                           nclass=labels.max().item() + 1, dropout=0.5, device=device)

    # ''' test on GCN (poisoning attack), model could be GCN, GAT, GIN'''
    classifier = globals()[args.model](nfeat=features.shape[1], nhid=16, nclass=labels.max().item() + 1, dropout=0.5, device=device)
    classifier = classifier.to(device)
    classifier.fit(features, adj, labels, idx_train,
                   idx_val=idx_val,
                   idx_test=idx_test,
                   verbose=False, attention=defense) #model_name=model_name
    classifier.eval()
    acc_overall, output =  classifier.test(idx_test, ) #model_name=model_name

    probs = torch.exp(output[[target_node]])
    acc_test, pred_y, true_y = accuracy_1(output[[target_node]], labels[target_node])
    with open(args.dataset+"_"+args.model+"_gsl.log","a+") as f:
        print('Defense: {}, target:{}, pred:{}, label: {}'.format(defense, target_node, pred_y.item(),true_y.item()),file=f)
    print('target:{}, pred:{}, label: {}'.format(target_node, pred_y.item(), true_y.item()))
    print('Pred probs', probs.data)
    return acc_test.item()

"""=======Basic Functions============="""
def select_nodes(num_target = 10):
    '''
    selecting nodes as reported in nettack paper:
    (i) the 10 nodes with highest margin of classification, i.e. they are clearly correctly classified,
    (ii) the 10 nodes with lowest margin (but still correctly classified) and
    (iii) 20 more nodes randomly
    '''
    gcn = globals()[args.model](nfeat=features.shape[1],
              nhid=16,
              nclass=labels.max().item() + 1,
              dropout=0.5, device=device)
    gcn = gcn.to(device)
    gcn.fit(features, adj, labels, idx_train, idx_test, verbose=True)
    gcn.eval()
    output = gcn.predict()
    degrees = adj.sum(0).A1

    margin_dict = {}
    for idx in tqdm(idx_test):
        margin = classification_margin(output[idx], labels[idx])
        acc, _, _ = accuracy_1(output[[idx]], labels[idx])
        if acc==0 or int(degrees[idx])<1: # only keep the correctly classified nodes
            continue
        """check the outliers:"""
        neighbours = list(adj.todense()[idx].nonzero()[1])
        y = [labels[i] for i in neighbours]
        node_y = labels[idx]
        aa = node_y==y
        outlier_score = 1- aa.sum()/len(aa)
        if outlier_score >=0.5:
            continue

        margin_dict[idx] = margin
    sorted_margins = sorted(margin_dict.items(), key=lambda x:x[1], reverse=True)
    high = [x for x, y in sorted_margins[: num_target]]
    low = [x for x, y in sorted_margins[-num_target: ]]
    other = [x for x, y in sorted_margins[num_target: -num_target]]
    other = np.random.choice(other, 2*num_target, replace=False).tolist()

    return other + high + low        
    
    
def accuracy_1(output,labels):
    try:
        num = len(labels)

    except:
        num = 1

    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor([labels])

    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()

    return correct/num, preds, labels



if __name__ == "__main__":
    #main()
    #multi_test()
    multi_evecs()









