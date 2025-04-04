import os
import numpy as np
import pandas as pd
import csv
import json
import dgl
import torch
from ogb.linkproppred import DglLinkPropPredDataset
os.environ["DGLBACKEND"] = "pytorch"
import dgl.backend as F
from dgl.data.utils import save_graphs, load_graphs
import random
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
from utils import negative_sample
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
import argparse

def train_val_test_split(graph, split_ratio, neg_ratio):

    num_edges = graph.num_edges()
    num_nodes = graph.num_nodes()
    src, dst = graph.edges()
    src, dst = F.asnumpy(src), F.asnumpy(dst)
    n_train, n_val, n_test = (
        int(num_edges * split_ratio[0]),
        int(num_edges * split_ratio[1]),
        int(num_edges * split_ratio[2]),
    )

    idx = np.random.permutation(num_edges)
    train_pos_idx = idx[:n_train]
    val_pos_idx = idx[n_train : n_train + n_val]
    test_pos_idx = idx[n_train + n_val :]
    neg_src, neg_dst = negative_sample(
        dgl.to_bidirected(graph,copy_ndata=True), neg_ratio * (n_val + n_test)
    )
    neg_n_val, neg_n_test = (
        neg_ratio * n_val,
        neg_ratio * n_test,
    )
    neg_val_src, neg_val_dst = (
        neg_src[:neg_n_val],
        neg_dst[:neg_n_val],
    )
    neg_test_src, neg_test_dst = (
        neg_src[neg_n_val:],
        neg_dst[neg_n_val:],
    )
    _val_edges = (
        F.tensor(src[val_pos_idx]),
        F.tensor(dst[val_pos_idx]),
    ), (F.tensor(neg_val_src), F.tensor(neg_val_dst))
    _test_edges = (
        F.tensor(src[test_pos_idx]),
        F.tensor(dst[test_pos_idx]),
    ), (F.tensor(neg_test_src), F.tensor(neg_test_dst))

    _train_graph = dgl.graph((src[train_pos_idx], dst[train_pos_idx]), num_nodes=num_nodes)
    _train_graph = dgl.to_bidirected(_train_graph, copy_ndata=True)

    _train_graph.ndata["feat"] = graph.ndata["feat"]

    return _train_graph, _val_edges, _test_edges


def process_dgl(dataset, path):
    """Process the DGL datasest"""

    isExist = os.path.exists(path + dataset)
    if not isExist:
        os.makedirs(path + dataset)

    if dataset == 'citeseer':
        data = dgl.data.CiteseerGraphDataset(raw_dir=path)
    elif dataset == 'pubmed':
        data = dgl.data.PubmedGraphDataset(raw_dir=path)
    elif dataset == 'cora':
        data = dgl.data.CoraGraphDataset(raw_dir=path)
    elif dataset == 'chameleon':
        data = dgl.data.ChameleonDataset(raw_dir=path)
    elif dataset == 'actor':
        data = dgl.data.ActorDataset(raw_dir=path)
    elif dataset == 'coauthor-cs':
        data = dgl.data.CoauthorCSDataset(raw_dir=path)
    elif dataset == 'coauthor-physics':
        data = dgl.data.CoauthorPhysicsDataset(raw_dir=path)

    graph = data[0]
    graph = dgl.to_bidirected(graph,copy_ndata=True)
    features = graph.ndata['feat'].numpy().astype(float)

    src, dst = graph.edges()
    mask = src <= dst
    directed_graph = dgl.graph((src[mask], dst[mask]))
    directed_graph.ndata['feat'] = graph.ndata['feat']

    np.save(path + dataset + '/feats', features)

    _train_graph, _val_edges, _test_edges = train_val_test_split(directed_graph, [0.8, 0.1, 0.1], 3)


    edges = _train_graph.edges()
    src = edges[0].tolist()
    dst = edges[1].tolist()
    edge_list = list(zip(src, dst))

    with open(path + dataset + '/edge_list_linkpred.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(edge_list)

    torch.save(_val_edges, path + dataset + '/val_edges.pt')

    torch.save(_test_edges, path + dataset + '/test_edges.pt')

    save_graphs(path + dataset + '/train_graph.bin', [_train_graph])


def process_ogb(dataset, path):
    """Process the OBG datasest"""

    isExist = os.path.exists(path + dataset)
    if not isExist:
        os.makedirs(path + dataset)

    datasets = DglLinkPropPredDataset(name = dataset, root = path + dataset + '/')

    split_edge = datasets.get_edge_split()

    train_edge, valid_edge, test_edge = split_edge["train"], split_edge["valid"], split_edge["test"]

    graph = datasets[0]

    val_edges = (valid_edge['edge'].T[0], valid_edge['edge'].T[1]), (valid_edge['edge_neg'].T[0], valid_edge['edge_neg'].T[1])
    test_edges = (test_edge['edge'].T[0], test_edge['edge'].T[1]), (test_edge['edge_neg'].T[0], test_edge['edge_neg'].T[1])

    edges = graph.edges()

    src = edges[0].tolist()
    dst = edges[1].tolist()

    edge_lists = list(zip(src, dst))

    # Write to CSV
    with open(path + dataset + '/edge_list_linkpred.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(edge_lists)

    torch.save(val_edges, path + dataset + '/val_edges.pt')

    torch.save(test_edges, path + dataset + '/test_edges.pt')

    graph.ndata['feat'] = graph.ndata['feat'].float()

    save_graphs(path + dataset + '/train_graph.bin', [graph])

    features = graph.ndata['feat'].numpy().astype(float)

    np.save(path + dataset + '/feats.npy', features)


if __name__ == "__main__":
    print('====='*10)
    path = os.path.abspath(os.path.join(os.getcwd(),'..'))
    dataset_path = path + '/datasets/'

    parser = argparse.ArgumentParser(description='Preprocessing')
    parser.add_argument('--dataset', type=str, default='cora')
    args = parser.parse_args()
    print(args)

    dataset = args.dataset

    if dataset in ['citeseer', 'cora', 'actor', 'chameleon', 'pubmed', 'coauthor-cs', 'coauthor-physics']:
        process_dgl(dataset, dataset_path)
    elif dataset in ['ogbl-collab', 'ogbl-ppa']:
        process_ogb(dataset, dataset_path)
