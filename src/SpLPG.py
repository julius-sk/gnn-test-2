import os
os.environ["DGLBACKEND"] = "pytorch"
import pickle
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
from dgl.nn import EdgeWeightNorm
from utils import linkpred_metrics, to_bidirected_with_reverse_mapping, average_models
from models import GraphSAGE, GCN, GAT, GATv2
import random
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.multiprocessing import Process
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
import argparse


@torch.no_grad()
def test(device, g, model, edges, batch_size, acc):
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
    
    accs = linkpred_metrics(torch.flatten(pos_score), torch.flatten(neg_score), acc)
        
    return accs


# """ Initialize the distributed environment. """
def init_processes(rank,
                   number_partition,
                   input_path,
                   output_path,
                   model_name, 
                   epochs, 
                   acc, 
                   batch_size, 
                   lr, 
                   eval_steps, 
                   weighted,
                   weights,
                   node_feats,
                   init_backend,
                   fn):
    
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12345'
    torch.distributed.init_process_group(backend=init_backend, rank=rank, world_size=number_partition)
    fn(rank,
       input_path,
       output_path,
       model_name, 
       epochs, 
       acc, 
       batch_size, 
       lr, 
       eval_steps, 
       weighted,
       weights,
       node_feats,
       init_backend)


def run(proc_id, 
        input_path,
        output_path,
        model_name, 
        epochs, 
        acc, 
        batch_size, 
        lr, 
        eval_steps, 
        weighted,
        weights,
        node_feats,
        init_backend):
    
    # use the first device for testing
    if proc_id == 0:
        train_graph = load_graphs(input_path + '/train_graph.bin', [0])[0][0]
        val_edges = torch.load(input_path + '/val_edges.pt')
        test_edges = torch.load(input_path + '/test_edges.pt')

        g_train_full, reverse_eids_full = to_bidirected_with_reverse_mapping(train_graph)
    
    if init_backend == 'nccl':
        device = torch.device(proc_id)
    else:
        device = torch.device(proc_id % torch.cuda.device_count())
    
    # load partitioned graph
    with open(output_path + 'partition_' + str(proc_id) + '-feats.json', 'rb') as fp:
        feat_dict = pickle.load(fp)
    
    glist, _ = load_graphs(output_path + 'partition_' + str(proc_id) + '.bin', [0])   
    g_train, reverse_eids = to_bidirected_with_reverse_mapping(glist[0])
    # print(g_train)

    if weighted:
        norm = EdgeWeightNorm(norm='both')
        norm_edge_weights = norm(g_train, g_train.edata['weight'])
        g_train.edata['weight'] = norm_edge_weights.float()
    
    edge_list = pd.read_csv(output_path + 'partition_' + str(proc_id) + '_sp.txt', sep=' ', names=['src','dst', 'weights'])
    g_p = dgl.graph((torch.tensor(edge_list['src']), torch.tensor(edge_list['dst'])), num_nodes=g_train.num_nodes())
    g_p = dgl.to_bidirected(g_p,copy_ndata=True)
    has_edges = g_p.has_edges_between(g_train.edges()[0], g_train.edges()[1])
    seed_edges = torch.tensor(torch.nonzero(~has_edges).squeeze().tolist())
    
    # create GNN model
    with open(output_path + 'num_feats.txt','r') as f:
        in_size = int(f.read())
    
    if model_name == 'GraphSAGE':
        model = GraphSAGE(in_size, 256, weighted).to(device)
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
    if weighted:
        for epoch in range(0, epochs):
            # model training
            model.train()
            for it, (input_nodes, pair_graph, neg_pair_graph, blocks) in enumerate(
                dataloader):    
                feats_list = []
                # copy node features to device if there are not locally available
                for node_id in blocks[0].srcdata['_ID']:
                    if node_id in feat_dict:
                        feats_list.append(torch.tensor(feat_dict[node_id], dtype=torch.float))
                    else:
                        feats_list.append(node_feats[node_id])
                x = torch.stack(feats_list).to(device)
                
                blocks_edge_weights = [block.edata['weight'] for block in blocks]
                pos_score, neg_score = model(pair_graph, neg_pair_graph, blocks, x, blocks_edge_weights)
                
                score = torch.cat([pos_score, neg_score])
                pos_label = torch.ones_like(pos_score)
                neg_label = torch.zeros_like(neg_score)
                labels = torch.cat([pos_label, neg_label])
                loss = F.binary_cross_entropy_with_logits(score, labels)
                opt.zero_grad()
                loss.backward()
                opt.step()
    
            # model synchronization
            torch.distributed.barrier()
            average_models(model,weights[proc_id])
            torch.distributed.barrier()
            model.eval()
            if proc_id == 0 and (epoch+1)%eval_steps == 0:
                # model testing
                val_accs = test(device, g_train_full, model, val_edges, batch_size, acc)
                test_accs = test(device, g_train_full, model, test_edges, batch_size, acc)
                print(f'Epoch: {epoch+1:02d}, Val: {val_accs:.4f}, Test: {test_accs:.4f}')
                results.append([val_accs, test_accs])
    else:
        for epoch in range(0, epochs):
            # model training
            model.train()
            for it, (input_nodes, pair_graph, neg_pair_graph, blocks) in enumerate(
                dataloader):    
                feats_list = []
                # copy node features to device if there are not locally available
                for node_id in blocks[0].srcdata['_ID']:
                    if node_id in feat_dict:
                        feats_list.append(torch.tensor(feat_dict[node_id], dtype=torch.float))
                    else:
                        feats_list.append(node_feats[node_id])
                x = torch.stack(feats_list).to(device)
                
                pos_score, neg_score = model(pair_graph, neg_pair_graph, blocks, x, None)
                
                score = torch.cat([pos_score, neg_score])
                pos_label = torch.ones_like(pos_score)
                neg_label = torch.zeros_like(neg_score)
                labels = torch.cat([pos_label, neg_label])
                loss = F.binary_cross_entropy_with_logits(score, labels)
                opt.zero_grad()
                loss.backward()
                opt.step()
    
            # model synchronization
            torch.distributed.barrier()
            average_models(model,weights[proc_id])
            torch.distributed.barrier()
            model.eval()
            if proc_id == 0 and (epoch+1)%eval_steps == 0:
                # model testing
                val_accs = test(device, g_train_full, model, val_edges, batch_size, acc)
                test_accs = test(device, g_train_full, model, test_edges, batch_size, acc)
                print(f'Epoch: {epoch+1:02d}, Val: {val_accs:.4f}, Test: {test_accs:.4f}')
                results.append([val_accs, test_accs])
    
    if proc_id == 0:
        best_result = max(results, key=lambda x: x[0])
        print('Best Accuracy: ', best_result)

if __name__ == '__main__':
    print('====='*10)
    path = os.path.abspath(os.path.join(os.getcwd(),'..'))
    
    parser = argparse.ArgumentParser(description='SpLPG')
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--number_partition', type=int, default=4)
    parser.add_argument('--model_name', type=str, default='GraphSAGE')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--acc', type=str, default='hits100')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--eval_steps', type=int, default=10)
    parser.add_argument('--weighted', action='store_true', help='Used for weighted graphs')
    args = parser.parse_args()
    print(args)
    
    dataset = args.dataset
    number_partition = args.number_partition
    model_name = args.model_name
    epochs = args.epochs
    acc = args.acc
    batch_size = args.batch_size
    lr = args.lr
    eval_steps = args.eval_steps
    weighted = args.weighted
    
    input_path = path + '/datasets/' + dataset
    output_path = path + '/output/partitions-' + str(number_partition) + '/' + dataset + '/SpLPG/'
    
    # put node features in shared memory
    node_feats = np.load(input_path + '/feats.npy')
    node_feats = torch.from_numpy(node_feats).float().share_memory_()
    
    # check number of devices and set init backend
    number_devices = torch.cuda.device_count()
    if number_devices == number_partition:
        init_backend='nccl'
    else:
        init_backend='gloo'
    
    # calculate weights for model synchronization
    weights = []
    for i in range(number_partition):
        glist, _ = load_graphs(output_path + 'partition_' + str(i) + '.bin', [0])
        g_train, reverse_eids = to_bidirected_with_reverse_mapping(glist[0])
        n_edges = torch.tensor(g_train.num_edges())
        weights.append(n_edges)
    weights = [i / sum(weights) for i in weights]
    
    # start processes
    processes = []
    torch.multiprocessing.set_start_method("spawn", force=True)
    
    for rank in range(number_partition):
        p = Process(target=init_processes, args=(rank,
                                                number_partition,
                                                input_path,
                                                output_path,
                                                model_name, 
                                                epochs, 
                                                acc, 
                                                batch_size, 
                                                lr, 
                                                eval_steps, 
                                                weighted,
                                                weights,
                                                node_feats,
                                                init_backend,
                                                run))
        
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        
    processes.clear()
    
    
