import os
import torch
os.environ["DGLBACKEND"] = "pytorch"
import numpy as np
import pandas as pd
import csv
import json
import pickle
import dgl
from numpy.polynomial import polynomial
os.environ["DGLBACKEND"] = "pytorch"
import dgl.backend as F
from dgl.data.utils import save_graphs, load_graphs
import random
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

# Function of accuracy metrics
def linkpred_metrics(y_pred_pos, y_pred_neg, name):
    if name == 'mrr':
        y_pred_neg = y_pred_neg.sort(-1)[0]
        rank = len(y_pred_neg) + 1 - torch.searchsorted(y_pred_neg, y_pred_pos)
        reciprocal_rank = rank.squeeze(-1).reciprocal()
        return reciprocal_rank.mean().item()

    elif name in ['hits20','hits50', 'hits100']:
        import re
        pattern = r'\d+'
        matches = re.findall(pattern, name)
        K = [int(match) for match in matches][0]

        kth_score_in_negative_edges = torch.topk(y_pred_neg, K)[0][-1]
        hitsK = float(torch.sum(y_pred_pos >= kth_score_in_negative_edges)) / len(y_pred_pos)

        return hitsK

# Function to save accuracy to CSV file
def save_to_csv(index, val_accs, test_accs, filename):
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([index, val_accs, test_accs])


# Function to create a DGL graph
def create_dgl_graph(dataset, path):

    edge_list = pd.read_csv(path + dataset + '/edge_list_linkpred.csv', header = None, names = ['src','dst'], index_col=False)

    src = edge_list['src'].to_list()
    dst = edge_list['dst'].to_list()

    graph = dgl.graph((src, dst), num_nodes=max(max(src), max(dst)) + 1)

    return graph

# Function to create node set
def read_set(file_name):
    with open(file_name, "r") as file:
        lines = file.readlines()
    result_set = set()
    for line in lines:
        parts = line.split()
        num1 = int(parts[0])
        num2 = int(parts[1])

        result_set.add(num1)
        result_set.add(num2)
    return result_set


# Function to reindex the node ID using pandas
def pd_reindex_id(edge_list):
    """Re-index the node ID of a graph using pandas."""

    node_set = pd.unique(edge_list[['src', 'dst']].values.ravel())
    node_set.sort()
    new_node_set = np.arange(len(node_set))
    node_mapping = dict(zip(node_set, new_node_set))
    edge_list['src'] = edge_list['src'].map(node_mapping)
    edge_list['dst'] = edge_list['dst'].map(node_mapping)

    return edge_list, node_mapping


# Functions for negative sampling
def _calc_redundancy(k_hat, num_edges, num_pairs, r=3):
    p_m = num_edges / num_pairs
    p_k = 1 - p_m

    a = p_k**2
    b = -p_k * (2 * k_hat + r**2 * p_m)
    c = k_hat**2

    poly = polynomial.Polynomial([c, b, a])
    N = poly.roots()[-1]
    redundancy = N / k_hat - 1.0
    return redundancy

def negative_sample(g, num_samples):
    num_nodes = g.num_nodes()
    redundancy = _calc_redundancy(num_samples, g.num_edges(), num_nodes**2)
    sample_size = int(num_samples * (1 + redundancy))
    edges = np.random.randint(0, num_nodes, size=(2, sample_size))
    edges = np.unique(edges, axis=1)
    mask_self_loop = edges[0] == edges[1]
    has_edges = F.asnumpy(g.has_edges_between(edges[0], edges[1]))
    mask = ~(np.logical_or(mask_self_loop, has_edges))
    edges = edges[:, mask]
    if edges.shape[1] >= num_samples:
        edges = edges[:, :num_samples]
    return edges

# Function to convert a graph to a bidirected graph with reverse mapping
def to_bidirected_with_reverse_mapping(g):
    g = dgl.remove_self_loop(g)
    g_simple, mapping = dgl.to_simple(
        dgl.add_reverse_edges(g,copy_edata=True), return_counts="count", writeback_mapping=True, copy_edata=True
    )
    c = g_simple.edata["count"]
    num_edges = g.num_edges()
    mapping_offset = torch.zeros(
        g_simple.num_edges() + 1, dtype=g_simple.idtype
    )
    mapping_offset[1:] = c.cumsum(0)
    idx = mapping.argsort()
    idx_uniq = idx[mapping_offset[:-1]]
    reverse_idx = torch.where(
        idx_uniq >= num_edges, idx_uniq - num_edges, idx_uniq + num_edges
    )
    reverse_mapping = mapping[reverse_idx]
    src1, dst1 = g_simple.edges()
    src2, dst2 = g_simple.find_edges(reverse_mapping)
    assert torch.equal(src1, dst2)
    assert torch.equal(src2, dst1)
    return g_simple, reverse_mapping


def average_gradients(model):
    """ Gradient averaging. """
    size = float(torch.distributed.get_world_size())
    for param in model.parameters():
        torch.distributed.all_reduce(param.grad.data, op=torch.distributed.ReduceOp.SUM)
        param.grad.data /= size


def average_models(model, weights):
    """ Model averaging"""

    for key, value in model.state_dict().items():

        value *= weights

        torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)

        model.state_dict()[key] = value
