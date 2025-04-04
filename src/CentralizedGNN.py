import os
os.environ["DGLBACKEND"] = "pytorch"
import csv
import dgl
import dgl.data
import torch
import torch.nn.functional as F
import sys
import json
import pandas as pd
import numpy as np
from dgl.data.utils import load_graphs
from dgl.dataloading import (
    as_edge_prediction_sampler,
    DataLoader,
    MultiLayerFullNeighborSampler,
    negative_sampler,
    NeighborSampler,
)
from utils import linkpred_metrics, to_bidirected_with_reverse_mapping
from models import GraphSAGE, GCN, GAT, GATv2
import random
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
import argparse


@torch.no_grad()
def test(device, g, model, edges, batch_size, acc):
    model.eval()
    
    node_emb = model.inference(g, device, batch_size)
    pos_src = edges[0][0].to(node_emb.device)
    pos_dst = edges[0][1].to(node_emb.device)
    
    neg_src = edges[1][0].to(node_emb.device)
    neg_dst = edges[1][1].to(node_emb.device)
    
    pos_score, neg_score = [], []
    for start in range(0, pos_src.shape[0], batch_size):
        end = min(start + batch_size, pos_src.shape[0])
   
        pos_h_src = node_emb[pos_src[start:end]][:, None, :].to(device)
        pos_h_dst = node_emb[pos_dst[start:end]][:, None, :].to(device)
        pos_pred = model.predictor(pos_h_src * pos_h_dst).squeeze(-1)
        pos_score.append(pos_pred)
    
    for start in range(0, neg_src.shape[0], batch_size):
        end = min(start + batch_size, neg_src.shape[0])
        neg_h_src = node_emb[neg_src[start:end]][:, None, :].to(device)
        neg_h_dst = node_emb[neg_dst[start:end]][:, None, :].to(device)
        neg_pred = model.predictor(neg_h_src * neg_h_dst).squeeze(-1)
        neg_score.append(neg_pred)
    
    pos_score = torch.cat(pos_score, dim=0)
    neg_score = torch.cat(neg_score, dim=0)
    
    accs = linkpred_metrics(torch.flatten(pos_score.cpu()), torch.flatten(neg_score.cpu()), acc)
        
    return accs

def run(path, model_name, epochs, acc, batch_size, lr, eval_steps, device):

    # load dataset
    train_graph = load_graphs(path + '/train_graph.bin', [0])[0][0]
    val_edges = torch.load(path + '/val_edges.pt')
    test_edges = torch.load(path + '/test_edges.pt')

    g_train, reverse_eids = to_bidirected_with_reverse_mapping(train_graph)
    seed_edges = torch.arange(g_train.num_edges())   
    
    # create GNN model
    in_size = g_train.ndata["feat"].shape[1]
    
    if model_name == 'GraphSAGE':
        model = GraphSAGE(in_size, 256).to(device)
    elif model_name == 'GCN':
        model = GCN(in_size, 256).to(device)
    elif model_name == 'GAT':
        model = GAT(in_size, 256).to(device)
    elif model_name == 'GATv2':
        model = GATv2(in_size, 256).to(device)
    
    # create sampler & dataloader
    if model_name == 'GraphSAGE':
        sampler = NeighborSampler([25, 10, 5])
    else:
        sampler = MultiLayerFullNeighborSampler(3)
    sampler = as_edge_prediction_sampler(
        sampler,
        exclude="reverse_id",
        reverse_eids=reverse_eids,
        negative_sampler=negative_sampler.Uniform(1),
    )
    dataloader = DataLoader(
        g_train,
        seed_edges,
        sampler,
        device=device,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=False,
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    results = []
    for epoch in range(0, epochs):
        model.train()
        for it, (input_nodes, pair_graph, neg_pair_graph, blocks) in enumerate(
            dataloader):
            x = blocks[0].srcdata["feat"]
            pos_score, neg_score = model(pair_graph, neg_pair_graph, blocks, x, None)
            score = torch.cat([pos_score, neg_score])
            pos_label = torch.ones_like(pos_score)
            neg_label = torch.zeros_like(neg_score)
            labels = torch.cat([pos_label, neg_label])
            loss = F.binary_cross_entropy_with_logits(score, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        # model testing
        if (epoch+1)%eval_steps == 0:
            val_accs = test(device, g_train, model, val_edges, batch_size, acc)
            test_accs = test(device, g_train, model, test_edges, batch_size, acc)
            print(f'Epoch: {epoch+1:02d}, Val: {val_accs:.4f}, Test: {test_accs:.4f}')
            results.append([val_accs, test_accs])
    
    best_result = max(results, key=lambda x: x[0])
    print('Best Accuracy: ', best_result)

        
if __name__ == "__main__":      
    print('====='*10)
    path = os.path.abspath(os.path.join(os.getcwd(),'..'))
    
    parser = argparse.ArgumentParser(description='Centralized GNN')
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--model_name', type=str, default='GraphSAGE')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--acc', type=str, default='hits100')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--eval_steps', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    print(args)
    
    
    dataset_path = path + '/datasets/' + args.dataset
    # train the model
    print("Training Starts..")
    run(dataset_path, 
        args.model_name, 
        args.epochs, 
        args.acc, 
        args.batch_size, 
        args.lr, 
        args.eval_steps, 
        torch.device(args.device))