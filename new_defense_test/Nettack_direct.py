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


# arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type = str, default = "cora",choices = ["cora","citeseer"],help="dataset")
parser.add_argument("--defense", type = bool, default = False,choices= [True,False],help="defense")
parser.add_argument("--model", type = str, default = "GCN", choices= ["GCN","GAT","GIN"])
parser.add_argument("--debug", type = bool, default = True, choices= [True,False])
parser.add_argument("--seed", type = int, default = 29, help="Random Seed" )
parser.add_argument("--direct", type = bool, default = True, choices = [True, False], help = "direct attack / influence attack")


args = parser.parse_args()
args.cuda = torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.cuda)

if args.debug:
    print('cuda :: {}\ndataset :: {}\nDefense Algo :: {}\nmodel :: {}'.format(args.cuda, args.dataset, args.defense, args.model))

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
target_node = 249
model = Nettack(surrogate, nnodes = adj.shape[0], attack_structure = True, attack_features = False, device = device).to(device)

#set defense
defense = args.defense


def main():
    degrees = adj.sum(0).A1
    per_num = int(degrees[target_node])

    if args.debug:
        print('degrees (# of perturbations) :: {}'.format(per_num))

    model.attack(features, adj, labels, target_node, per_num, direct = args.direct)
    m_adj = model.modified_adj
    m_features = model.modified_features

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
        model.attack(features, adj, labels, target_node, n_perturbations, direct=True, verbose=False)
        modified_adj = model.modified_adj
        modified_features = model.modified_features
        acc = single_test(modified_adj, modified_features, target_node)
        if acc == 0:
            cnt += 1
        num_tar += 1
        print('classification rate : %s' % (1-cnt/num_tar), '# of targets:', num_tar)


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
    multi_test()









