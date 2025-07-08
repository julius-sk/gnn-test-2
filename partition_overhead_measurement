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


class PartitioningWithOverheadMeasurement():
    """Graph Partitioning with Memory Overhead Measurement."""

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
        
        # Variables to track memory overhead
        self.original_edges = 0
        self.total_partitioned_edges = 0
        self.cross_partition_edges = 0
        self.partition_edge_counts = []
        
        print(f"Original graph: {self.number_nodes} nodes, {self.number_edges} edges")
    
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

        self.v2p = defaultdict(int)
        for i in range(len(partitions)):
            self.v2p[int(i)] = int(partitions[i])
        
        with open(self.output_path + 'partition' + '.json', 'wb') as json_file:
            pickle.dump(self.v2p, json_file)

    def measure_partition_overhead(self):
        """Measure the memory overhead introduced by partitioning."""
        
        # Count original edges (considering bidirectional)
        edge_list = pd.read_csv(self.file_name, header=None, names=['src', 'dst'])
        self.original_edges = len(edge_list) * 2  # bidirectional
        
        print(f"\nOriginal bidirectional edges: {self.original_edges}")
        
        # Initialize counters
        partition_edges = [0] * self.number_partition
        cross_partition_count = 0
        total_edges_stored = 0
        
        # Track which edges go to which partitions
        edge_assignments = defaultdict(set)  # edge -> set of partitions
        
        # Process each edge
        for _, row in edge_list.iterrows():
            i, j = int(row['src']), int(row['dst'])
            
            # For each edge (i,j), add both directions (i,j) and (j,i)
            edges_to_process = [(i, j), (j, i)]
            
            for src, dst in edges_to_process:
                src_partition = self.v2p[src]
                dst_partition = self.v2p[dst]
                
                edge_key = (min(src, dst), max(src, dst))  # Normalize edge representation
                
                if src_partition == dst_partition:
                    # Edge is within partition
                    edge_assignments[edge_key].add(src_partition)
                    partition_edges[src_partition] += 1
                else:
                    # Cross-partition edge - needs to be stored in both partitions
                    edge_assignments[edge_key].add(src_partition)
                    edge_assignments[edge_key].add(dst_partition)
                    partition_edges[src_partition] += 1
                    partition_edges[dst_partition] += 1
                    cross_partition_count += 1
        
        # Calculate total edges stored across all partitions
        total_edges_stored = sum(partition_edges)
        
        # Calculate overhead
        overhead_edges = total_edges_stored - self.original_edges
        overhead_ratio = overhead_edges / self.original_edges if self.original_edges > 0 else 0
        
        # Additional analysis
        cross_partition_edges_unique = sum(1 for edge_partitions in edge_assignments.values() 
                                         if len(edge_partitions) > 1)
        
        print(f"\n=== MEMORY OVERHEAD ANALYSIS ===")
        print(f"Original edges (bidirectional): {self.original_edges}")
        print(f"Total edges stored across partitions: {total_edges_stored}")
        print(f"Overhead edges: {overhead_edges}")
        print(f"Overhead ratio: {overhead_ratio:.4f} ({overhead_ratio*100:.2f}%)")
        print(f"\nDetailed breakdown:")
        print(f"- Cross-partition edges (unique): {cross_partition_edges_unique}")
        print(f"- Cross-partition edge instances: {cross_partition_count}")
        print(f"- Within-partition edges: {total_edges_stored - cross_partition_count}")
        
        for i in range(self.number_partition):
            print(f"- Partition {i}: {partition_edges[i]} edges")
        
        # Calculate replication factor
        avg_replication = total_edges_stored / (len(edge_assignments) * 2)  # *2 for bidirectional
        print(f"- Average edge replication factor: {avg_replication:.4f}")
        
        # Store results
        self.partition_edge_counts = partition_edges
        self.total_partitioned_edges = total_edges_stored
        self.cross_partition_edges = cross_partition_count
        self.overhead_ratio = overhead_ratio
        
        return {
            'original_edges': self.original_edges,
            'total_partitioned_edges': total_edges_stored,
            'overhead_edges': overhead_edges,
            'overhead_ratio': overhead_ratio,
            'cross_partition_edges': cross_partition_edges_unique,
            'partition_edge_counts': partition_edges,
            'replication_factor': avg_replication
        }

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
        
        for _, line in enumerate(reader):
            i, j = int(line[0]), int(line[1])
            if self.v2p[i] == self.v2p[j]:
                # Within partition edge
                partition_id = self.v2p[j]
                writer = csv.writer(file_objects[partition_id], delimiter=' ')
                writer.writerow(line)
            else:
                # Cross-partition edge - store in both partitions
                for partition_id in [self.v2p[i], self.v2p[j]]:
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
        
        for i in range(self.number_partition):
            edge_list = np.loadtxt(self.output_path + 'partition_' + str(i) + '.txt', dtype=int)
            node_set = np.unique(edge_list)
            feat = node_feats[node_set]
            np.save(self.output_path + 'partition_' + str(i) + '-feats.npy', feat)

    def save_dgl_graph(self):
        """Save the partitioned subgraphs as DGL graph objects."""
        
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

    def save_overhead_results(self, results):
        """Save overhead measurement results to file."""
        
        results_file = self.output_path + 'overhead_analysis.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nOverhead analysis results saved to: {results_file}")
    
    def run(self):
        """Run the partitioning algorithm with overhead measurement."""
        print("Step 1: Assigning partitions...")
        self.assign_partitions()
        
        print("Step 2: Measuring partition overhead...")
        overhead_results = self.measure_partition_overhead()
        
        print("Step 3: Creating partition files...")
        self.partition_file()
        
        print("Step 4: Partitioning features...")
        self.partition_features()
        
        print("Step 5: Saving DGL graphs...")
        self.save_dgl_graph()
        
        print("Step 6: Saving overhead results...")
        self.save_overhead_results(overhead_results)
        
        print('Partitioning completed with overhead measurement.')
        return overhead_results


if __name__ == "__main__":
    print('====='*10)
    path = os.path.abspath(os.path.join(os.getcwd(),'..'))
    
    parser = argparse.ArgumentParser(description='Graph Partitioning with Memory Overhead Measurement')
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--number_partition', type=int, default=4)
    parser.add_argument('--method', type=str, default='METIS')
    args = parser.parse_args()
    print(args)
    
    dataset = args.dataset
    number_partition = args.number_partition
    method = args.method
    
    partitioner = PartitioningWithOverheadMeasurement(
        dataset=dataset, 
        method=method,
        number_partition=number_partition,
        input_path=path + '/datasets/',
        output_path=path + '/output/' + 'partitions-' + str(number_partition) + '/' + dataset + '/' + method + '/'
    )
    
    overhead_results = partitioner.run()
    
    print(f"\n=== FINAL SUMMARY ===")
    print(f"Dataset: {dataset}")
    print(f"Partitioning method: {method}")
    print(f"Number of partitions: {number_partition}")
    print(f"Memory overhead ratio: {overhead_results['overhead_ratio']:.4f} ({overhead_results['overhead_ratio']*100:.2f}%)")
    print(f"This means partitioned data uses {overhead_results['overhead_ratio']*100:.2f}% more memory than original data")
