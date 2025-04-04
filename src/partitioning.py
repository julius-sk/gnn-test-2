import os
import numpy as np
import pandas as pd
import pickle
import json
import csv
from collections import defaultdict
import torch
import dgl
import sys
from dgl.data.utils import save_graphs, load_graphs
import math
from utils import create_dgl_graph, pd_reindex_id, read_set
import random
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
import time
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
import argparse
import numba
from numba.typed import Dict

@numba.njit
def compute_p_edges(src, dst, node_degree):
    p_edges = np.empty(len(src), dtype=np.float64)
    for l in range(len(src)):
        i = src[l]
        j = dst[l]
        p_edges[l] = 1.0 / node_degree[i] + 1.0 / node_degree[j]
    return p_edges / p_edges.sum()


class Partitioning():
    """Graph Partitioning."""

    def __init__(self, dataset: str = None, 
                 input_path: str = None, 
                 output_path: str = None, 
                 number_partition: int = 4, 
                 method: str = 'METIS',
                 seed: int = 42):
        super().__init__()
        """Initialization."""
        
        self.dataset = dataset
        self.input_path = input_path
        self.output_path = output_path
        self.number_partition = number_partition
        self.method = method
        self.seed = seed
        self._set_seed()
        
        isExist = os.path.exists(self.output_path)
        if not isExist:
            os.makedirs(self.output_path)
        
        self.file_name = self.input_path + self.dataset + '/edge_list_linkpred.csv'

        self.g = create_dgl_graph(self.dataset, self.input_path)
        self.g = dgl.to_bidirected(self.g, copy_ndata=True)

        self.number_edges = self.g.num_edges()
        self.number_nodes = self.g.num_nodes()
    
    
    def _set_seed(self):
        """Creating the initial random seed."""
        
        random.seed(self.seed)
        np.random.seed(self.seed)


    def assign_partitions(self):
        """Partition a graph."""
        if self.method == 'METIS':
            partitions = dgl.metis_partition_assignment(self.g, 
                                                        self.number_partition, 
                                                        balance_edges=True, 
                                                        mode='k-way', 
                                                        objtype='cut')
        
        
        elif self.method == 'RandomTMA':
            partitions = [random.randint(0, self.number_partition - 1) for _ in range(self.g.num_nodes())]
            
        
        elif self.method == 'SuperTMA':
            self.number_clusters = 1500
            partitions_metis = dgl.metis_partition_assignment(self.g, 
                                                            self.number_clusters,
                                                            balance_edges=False, 
                                                            mode='k-way',
                                                            objtype='cut').tolist()
        
            rng = np.random.default_rng(seed=self.seed)
            group_list = []
            cluster_per_group = float(self.number_clusters) / self.number_partition
            cluster_list = np.arange(self.number_clusters)
            rng.shuffle(cluster_list)
            p = 0
            for i in range(self.number_partition):
                group_size = int(np.floor(cluster_per_group))
                if rng.random() < cluster_per_group % 1:
                    group_size += 1
                if i != self.number_partition - 1:
                    p_end = p + group_size
                else:
                    p_end = self.number_clusters
                group_list.append(cluster_list[p:p_end])
                p = p_end

            sets_list = []
            for i in range(self.number_partition):
                new_set = set()
                for j in range(len(partitions_metis)):
                    if partitions_metis[j] in group_list[i]:
                        new_set.add(j)
                sets_list.append(new_set)

            max_value = max(max(sublist) for sublist in sets_list)
            partitions = [0] * (max_value + 1)
            for index, sublist in enumerate(sets_list):
                for value in sublist:
                    partitions[value] = index

        elif self.method == 'SpLPG':
            partitions = dgl.metis_partition_assignment(self.g, self.number_partition, 
                                                  balance_edges=True, 
                                                  mode='k-way', 
                                                  objtype='cut')

        
        self.v2p = defaultdict(int)
        for i in range(len(partitions)):
            self.v2p[int(i)] = int(partitions[i])
        
        with open(self.output_path + 'partition' + '.json', 'wb') as json_file:
            pickle.dump(self.v2p, json_file)
    
    
    def sparsification(self):  
        edges_sets = []
        self.nodes_sets = []
        for i in range(self.number_partition):
            filename = os.path.join(self.output_path, 'partition_' + str(i) + '.txt')
            files = open(filename, 'r', newline='')
            reader = csv.reader(files, delimiter=' ')

            node_degree = Dict.empty(key_type=numba.types.int64, value_type=numba.types.float64)
            for n_edges, line in enumerate(reader):
                i, j = int(line[0]), int(line[1])
                if j not in node_degree:
                    node_degree[j] = 0.0
                node_degree[j] += 1
            
            number_edges = n_edges + 1
            files.close()
            
            new_set = read_set(filename)
            new_set_n_nodes = len(new_set)
            self.nodes_sets.append(new_set)
        
            data = np.loadtxt(filename, dtype=float)

            src = data[:, 0].astype(int)
            dst = data[:, 1].astype(int)
            p_edges = compute_p_edges(src, dst, node_degree)
            
            sampled_edges = np.random.choice(np.arange(number_edges), size=int(0.15*number_edges), replace=True, p=p_edges)

            sampled_edges_unique, edges_cnt = np.unique(sampled_edges, return_counts=True)
            edge_weights = edges_cnt / 0.15 / number_edges / p_edges[sampled_edges_unique]
            
            files = open(filename, 'r', newline='')
            reader = csv.reader(files, delimiter=' ')
            edges_set = set()
            edge_weights_index = 0
            for row, line in enumerate(reader):
                if row in sampled_edges_unique:
                    edges_set.add((int(line[0]), int(line[1]), edge_weights[edge_weights_index]))
                    edge_weights_index += 1
            files.close()
            edges_sets.append(edges_set)
         
            
        for i in range(self.number_partition):
            filename = os.path.join(self.output_path, 'partition_' + str(i) + '.txt')
            files = open(filename, 'a', newline='')
            reader = csv.reader(files, delimiter=' ')
            for j in range(self.number_partition):
                if i != j:
                    for edge in edges_sets[j]:
                        writer = csv.writer(files, delimiter=' ')
                        writer.writerow(edge)
            files.close()
        
        for i in range(self.number_partition):
            filename = os.path.join(self.output_path, 'partition_' + str(i) + '_sp.txt')
            files = open(filename, 'w', newline='')
            reader = csv.reader(files, delimiter=' ')
            for j in range(self.number_partition):
                if i != j:
                    for edge in edges_sets[j]:
                        writer = csv.writer(files, delimiter=' ')
                        writer.writerow(edge)
            files.close()
    
    
    def partition_file(self):
        """Partition the edge list file."""
        
        for i in range(self.number_partition):
            with open(os.path.join(self.output_path, 'partition_' + str(i) + '.txt'), 'w') as fp:
                pass
            
        self.file = open(self.file_name, 'r')
        
        file_objects = []
        for i in range(self.number_partition):
            filename = os.path.join(self.output_path, 'partition_' + str(i) + '.txt')
            files = open(filename, 'a', newline='')
            file_objects.append(files)
        
        reader = csv.reader(self.file, delimiter=',')
        
        
        if self.method == 'SpLPG':
            # if self.weighted:
            for _, line in enumerate(reader):
                i, j = int(line[0]), int(line[1])
                
                partition_id = self.v2p[j]
                writer = csv.writer(file_objects[partition_id], delimiter=' ')
                line.append(1.0)
                writer.writerow(line)
                
                if self.v2p[i] != self.v2p[j]:
                    partition_id = self.v2p[i]
                    writer = csv.writer(file_objects[partition_id], delimiter=' ')
                    writer.writerow(line)
            self.file.close()
        
        else:
            for _, line in enumerate(reader):
                i, j = int(line[0]), int(line[1])
                if self.v2p[i] == self.v2p[j]:
                    partition_id = self.v2p[j]
                    writer = csv.writer(file_objects[partition_id], delimiter=' ')
                    writer.writerow(line)
            self.file.close()
        
        for files in file_objects:
            files.close()
    
    
    def partition_features(self):
        """Partition the feature file."""
        
        node_feats = np.load(self.input_path + self.dataset +'/feats.npy', mmap_mode='r')
        n_feats = node_feats.shape[1]
        
        with open(self.output_path + 'num_feats.txt', 'w') as f:
            f.write(str(n_feats))
        
        if self.method == 'SpLPG':
            for i in range(self.number_partition):
                feat = node_feats[list(self.nodes_sets[i])]
                feat_dict = {key: feat[j] for j, key in enumerate(list(self.nodes_sets[i]))}
                
                with open(self.output_path + 'partition_' + str(i) + '-feats.json', 'wb') as json_file:
                    pickle.dump(feat_dict, json_file)
        
        else:
            for i in range(self.number_partition):
                edge_list = np.loadtxt(self.output_path + 'partition_' + str(i) + '.txt', dtype=int)
                node_set = np.unique(edge_list)
                feat = node_feats[node_set]
                np.save(self.output_path + 'partition_' + str(i) + '-feats.npy', feat)
            

    def save_dgl_graph(self):
        """Save the partitioned subgraphs as DGL graph objects."""
        
        if self.method == 'SpLPG':
            for i in range(number_partition):
                edge_list = pd.read_csv(self.output_path + 'partition_' + str(i) + '.txt', sep=' ', names=['src','dst', 'weights'])
                edge_weights = torch.tensor(edge_list['weights'])
                g_p = dgl.graph((torch.tensor(edge_list['src']), torch.tensor(edge_list['dst'])), num_nodes=self.number_nodes)
                g_p.edata['weight'] = edge_weights
                # g_p = dgl.to_bidirected(g_p, copy_ndata=True)
                g_p = dgl.add_reverse_edges(g_p, copy_ndata=True, copy_edata=True)
                g_p = dgl.to_simple(g_p, return_counts=None, copy_ndata=True, copy_edata=True)
                save_graphs(self.output_path + 'partition_' + str(i) + '.bin', [g_p])
          
        else:
            for i in range(self.number_partition):
                edge_list = pd.read_csv(self.output_path + 'partition_' + str(i) + '.txt', sep=' ', names=['src','dst'])
                edge_list, node_mapping = pd_reindex_id(edge_list)

                with open(self.output_path + 'partition_' + str(i) + '-mapping.json', 'wb') as json_file:
                    pickle.dump(node_mapping, json_file)
                
                g_p = dgl.graph((torch.tensor(edge_list['src']), torch.tensor(edge_list['dst'])))
                g_p = dgl.to_bidirected(g_p, copy_ndata=True)

                feat = np.load(self.output_path + 'partition_' + str(i) + '-feats.npy')
                g_p.ndata['feat'] = torch.from_numpy(feat).float()
                save_graphs(self.output_path + 'partition_' + str(i) + '.bin', [g_p])
    
    
    def run(self):
        """Run the partitioning algorithm."""
        self.assign_partitions()
        self.partition_file()
        if self.method in ['SpLPG'] :
            self.sparsification()      
        self.partition_features()
        self.save_dgl_graph()
        print('Partitioning completed.')


if __name__ == "__main__":
    print('====='*10)
    path = os.path.abspath(os.path.join(os.getcwd(),'..'))
    
    parser = argparse.ArgumentParser(description='Graph Partitioning')
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--number_partition', type=int, default=4)
    parser.add_argument('--method', type=str, default='SpLPG')
    args = parser.parse_args()
    print(args)
    
    dataset = args.dataset
    number_partition = args.number_partition
    method = args.method
    
    partitioner = Partitioning(dataset = dataset, 
                            method = method,
                            number_partition = number_partition,
                            input_path = path + '/datasets/',
                            output_path = path + '/output/' + 'partitions-' + str(number_partition) + '/' + dataset + '/' + method + '/')
    partitioner.run()
