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
import random
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
import time
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
import argparse

# Import MaxK-GNN dataset loaders
from dgl import AddSelfLoop
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset
from dgl.data import RedditDataset, FlickrDataset, YelpDataset
from ogb.nodeproppred import DglNodePropPredDataset


def load_maxkgnn_dataset(dataset_name, data_path="./data/", selfloop=False):
    """Load datasets in the same way as MaxK-GNN project."""
    
    transform = AddSelfLoop() if selfloop else None
    
    if "ogb" not in dataset_name:
        # Standard DGL datasets
        if dataset_name == 'reddit':
            data = RedditDataset(transform=transform, raw_dir=data_path)
        elif dataset_name == 'flickr':
            data = FlickrDataset(transform=transform, raw_dir=data_path)
        elif dataset_name == 'yelp':
            data = YelpDataset(transform=transform, raw_dir=data_path)
        elif dataset_name == 'cora':
            data = CoraGraphDataset(transform=transform, raw_dir=data_path)
        elif dataset_name == 'citeseer':
            data = CiteseerGraphDataset(transform=transform, raw_dir=data_path)
        elif dataset_name == 'pubmed':
            data = PubmedGraphDataset(transform=transform, raw_dir=data_path)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        g = data[0]
        # FORCE int64 conversion immediately using correct method
        g = g.long()  # This is the correct way to convert to int64
        features = g.ndata["feat"]
        if dataset_name == 'yelp':
            labels = g.ndata["label"].float()
        else:
            labels = g.ndata["label"]
        
        # Get masks
        if hasattr(g.ndata, 'train_mask'):
            train_mask = g.ndata["train_mask"].bool()
            val_mask = g.ndata["val_mask"].bool() 
            test_mask = g.ndata["test_mask"].bool()
        else:
            # Create dummy masks if not available
            num_nodes = g.num_nodes()
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            
        return g, features, labels, (train_mask, val_mask, test_mask), data.num_classes
        
    else:
        # OGB datasets
        data = DglNodePropPredDataset(name=dataset_name, root=data_path)
        split_idx = data.get_idx_split()
        
        g, labels = data[0]
        labels = torch.squeeze(labels, dim=1)
        # FORCE int64 conversion immediately using correct method
        g = g.long()  # This is the correct way to convert to int64
        features = g.ndata["feat"]
        
        # Convert split indices to masks
        train_idx = split_idx["train"]
        valid_idx = split_idx["valid"] 
        test_idx = split_idx["test"]
        
        total_nodes = g.num_nodes()
        train_mask = torch.zeros(total_nodes, dtype=torch.bool)
        train_mask[train_idx] = 1
        valid_mask = torch.zeros(total_nodes, dtype=torch.bool)
        valid_mask[valid_idx] = 1
        test_mask = torch.zeros(total_nodes, dtype=torch.bool)
        test_mask[test_idx] = 1
        
        return g, features, labels, (train_mask, valid_mask, test_mask), data.num_classes


class MaxKGNNPartitioningOverhead():
    """Graph Partitioning with Memory Overhead Measurement for MaxK-GNN datasets."""

    def __init__(self, dataset: str = None, 
                 data_path: str = "./data/", 
                 output_path: str = None, 
                 number_partition: int = 4, 
                 method: str = 'METIS',
                 selfloop: bool = False,
                 seed: int = 42):
        super().__init__()
        """Initialization."""
        
        self.dataset = dataset
        self.data_path = data_path
        self.output_path = output_path or f"./partition_analysis/{dataset}/"
        self.number_partition = number_partition
        self.method = method
        self.selfloop = selfloop
        self.seed = seed
        self._set_seed()
        
        # Create output directory
        os.makedirs(self.output_path, exist_ok=True)
        
        # Load the dataset
        print(f"Loading dataset: {dataset}")
        self.g, self.features, self.labels, self.masks, self.num_classes = load_maxkgnn_dataset(
            dataset, data_path, selfloop
        )
        
        # CRITICAL: Ensure graph is int64 immediately after loading
        print(f"Original graph dtype: {self.g.idtype}")
        if self.g.idtype != torch.int64:
            print("Converting graph to int64...")
            self.g = self.g.long()
            print(f"Graph dtype after conversion: {self.g.idtype}")
        
        # Add self-loop if specified (ensure int64 compatibility)
        if selfloop:
            self.g = dgl.add_self_loop(self.g)
            # Ensure the graph remains int64 after adding self-loops
            if self.g.idtype != torch.int64:
                self.g = self.g.long()
                print(f"Graph dtype after self-loop: {self.g.idtype}")
        
        self.number_edges = self.g.num_edges()
        self.number_nodes = self.g.num_nodes()
        
        # Variables to track memory overhead
        self.original_edges = self.number_edges
        self.total_partitioned_edges = 0
        self.cross_partition_edges = 0
        self.partition_edge_counts = []
        
        print(f"Dataset loaded: {self.number_nodes} nodes, {self.number_edges} edges")
        print(f"Features shape: {self.features.shape}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Graph dtype: {self.g.idtype}")  # Debug info
    
    def _set_seed(self):
        """Creating the initial random seed."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def assign_partitions(self):
        """Partition a graph using METIS or other methods."""
        
        # Triple-check graph is int64 for METIS
        print(f"Graph dtype before METIS: {self.g.idtype}")
        if self.g.idtype != torch.int64:
            print("CRITICAL: Converting graph to int64 for METIS compatibility...")
            self.g = self.g.long()
            print(f"Graph dtype after conversion: {self.g.idtype}")
        
        try:
            if self.method == 'METIS':
                print("Calling METIS partitioning...")
                partitions = dgl.metis_partition_assignment(self.g, 
                                                            self.number_partition, 
                                                            balance_edges=True, 
                                                            mode='k-way', 
                                                            objtype='cut')
                print("METIS partitioning completed successfully!")
            elif self.method == 'Random':
                partitions = torch.randint(0, self.number_partition, (self.number_nodes,))
            else:
                raise ValueError(f"Unknown partitioning method: {self.method}")
        except Exception as e:
            print(f"Error during partitioning: {e}")
            print(f"Graph properties:")
            print(f"  - Number of nodes: {self.g.num_nodes()}")
            print(f"  - Number of edges: {self.g.num_edges()}")
            print(f"  - Node dtype: {self.g.idtype}")
            print(f"  - Device: {self.g.device}")
            raise e
        
        self.v2p = {}
        for i in range(len(partitions)):
            self.v2p[int(i)] = int(partitions[i])
        
        # Save partition assignment
        with open(os.path.join(self.output_path, 'partition_assignment.json'), 'w') as f:
            json.dump(self.v2p, f, indent=2)
        
        print(f"Partitioned {self.number_nodes} nodes into {self.number_partition} partitions using {self.method}")
        
        # Print partition balance
        partition_sizes = [0] * self.number_partition
        for node_id, partition_id in self.v2p.items():
            partition_sizes[partition_id] += 1
        
        print("Partition node distribution:")
        for i, size in enumerate(partition_sizes):
            print(f"  Partition {i}: {size} nodes ({size/self.number_nodes*100:.1f}%)")

    def measure_partition_overhead(self):
        """Measure the memory overhead introduced by partitioning."""
        
        print(f"\nOriginal edges: {self.original_edges}")
        
        # Get edges from the graph
        src_nodes, dst_nodes = self.g.edges()
        src_nodes = src_nodes.numpy()
        dst_nodes = dst_nodes.numpy()
        
        # Initialize counters
        partition_edges = [0] * self.number_partition
        cross_partition_count = 0
        total_edges_stored = 0
        
        # Track which edges go to which partitions
        edge_assignments = defaultdict(set)  # edge -> set of partitions
        
        # Process each edge
        for i in range(len(src_nodes)):
            src = int(src_nodes[i])
            dst = int(dst_nodes[i])
            
            src_partition = self.v2p[src]
            dst_partition = self.v2p[dst]
            
            # Create a normalized edge key (smaller node first)
            edge_key = (min(src, dst), max(src, dst))
            
            # Track which partitions this edge belongs to
            edge_assignments[edge_key].add(src_partition)
            if src_partition != dst_partition:
                edge_assignments[edge_key].add(dst_partition)
        
        # Count edges stored in each partition
        internal_edges = 0
        cross_edges = 0
        
        for edge_key, partitions in edge_assignments.items():
            num_partitions = len(partitions)
            if num_partitions == 1:
                # Internal edge - stored in one partition
                partition_id = list(partitions)[0]
                partition_edges[partition_id] += 1
                internal_edges += 1
            else:
                # Cross-partition edge - stored in multiple partitions
                for partition_id in partitions:
                    partition_edges[partition_id] += 1
                cross_edges += 1
                cross_partition_count += num_partitions  # Count total storage instances
        
        # Calculate totals
        total_edges_stored = sum(partition_edges)
        unique_edges = len(edge_assignments)
        
        # Calculate overhead
        overhead_edges = total_edges_stored - unique_edges
        overhead_ratio = overhead_edges / unique_edges if unique_edges > 0 else 0
        
        print(f"\n=== MEMORY OVERHEAD ANALYSIS ===")
        print(f"Original unique edges: {unique_edges}")
        print(f"Total edges stored across partitions: {total_edges_stored}")
        print(f"Overhead edges (duplications): {overhead_edges}")
        print(f"Overhead ratio: {overhead_ratio:.4f} ({overhead_ratio*100:.2f}%)")
        print(f"\nDetailed breakdown:")
        print(f"- Internal edges (within partition): {internal_edges}")
        print(f"- Cross-partition edges (unique): {cross_edges}")
        print(f"- Cross-partition edge storage instances: {cross_partition_count}")
        
        for i in range(self.number_partition):
            print(f"- Partition {i}: {partition_edges[i]} edges ({partition_edges[i]/total_edges_stored*100:.1f}%)")
        
        # Calculate replication factor
        avg_replication = total_edges_stored / unique_edges if unique_edges > 0 else 0
        print(f"- Average edge replication factor: {avg_replication:.4f}")
        
        # Calculate cut ratio (fraction of edges that are cross-partition)
        cut_ratio = cross_edges / unique_edges if unique_edges > 0 else 0
        print(f"- Cut ratio (cross-partition edges / total edges): {cut_ratio:.4f} ({cut_ratio*100:.2f}%)")
        
        # Store results
        self.partition_edge_counts = partition_edges
        self.total_partitioned_edges = total_edges_stored
        self.cross_partition_edges = cross_edges
        self.overhead_ratio = overhead_ratio
        
        return {
            'dataset': self.dataset,
            'partitioning_method': self.method,
            'number_partitions': self.number_partition,
            'original_nodes': self.number_nodes,
            'original_edges': self.original_edges,
            'unique_edges': unique_edges,
            'total_partitioned_edges': total_edges_stored,
            'overhead_edges': overhead_edges,
            'overhead_ratio': overhead_ratio,
            'internal_edges': internal_edges,
            'cross_partition_edges': cross_edges,
            'cross_partition_storage_instances': cross_partition_count,
            'partition_edge_counts': partition_edges,
            'replication_factor': avg_replication,
            'cut_ratio': cut_ratio,
            'partition_node_distribution': [sum(1 for p in self.v2p.values() if p == i) 
                                          for i in range(self.number_partition)]
        }

    def create_partitioned_subgraphs(self):
        """Create and save partitioned subgraphs."""
        
        print("\nCreating partitioned subgraphs...")
        
        for partition_id in range(self.number_partition):
            # Get nodes in this partition
            partition_nodes = [node for node, p in self.v2p.items() if p == partition_id]
            partition_nodes = torch.tensor(partition_nodes)
            
            # Create subgraph
            subgraph = dgl.node_subgraph(self.g, partition_nodes)
            
            # Add features
            subgraph.ndata['feat'] = self.features[partition_nodes]
            subgraph.ndata['label'] = self.labels[partition_nodes]
            
            # Add masks
            train_mask, val_mask, test_mask = self.masks
            subgraph.ndata['train_mask'] = train_mask[partition_nodes]
            subgraph.ndata['val_mask'] = val_mask[partition_nodes]
            subgraph.ndata['test_mask'] = test_mask[partition_nodes]
            
            # Save subgraph
            subgraph_path = os.path.join(self.output_path, f'partition_{partition_id}.bin')
            save_graphs(subgraph_path, [subgraph])
            
            print(f"  Partition {partition_id}: {subgraph.num_nodes()} nodes, {subgraph.num_edges()} edges")

    def save_overhead_results(self, results):
        """Save overhead measurement results to file."""
        
        results_file = os.path.join(self.output_path, 'overhead_analysis.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Also save a summary CSV for easy comparison
        summary_file = os.path.join(self.output_path, 'overhead_summary.csv')
        summary_data = {
            'dataset': [results['dataset']],
            'method': [results['partitioning_method']], 
            'num_partitions': [results['number_partitions']],
            'nodes': [results['original_nodes']],
            'edges': [results['original_edges']],
            'overhead_ratio': [results['overhead_ratio']],
            'overhead_percent': [results['overhead_ratio'] * 100],
            'cut_ratio': [results['cut_ratio']],
            'replication_factor': [results['replication_factor']]
        }
        
        df = pd.DataFrame(summary_data)
        df.to_csv(summary_file, index=False)
        
        print(f"\nResults saved to:")
        print(f"  Detailed: {results_file}")
        print(f"  Summary: {summary_file}")
    
    def run(self):
        """Run the partitioning algorithm with overhead measurement."""
        print("="*60)
        print(f"PARTITIONING OVERHEAD ANALYSIS")
        print(f"Dataset: {self.dataset}")
        print(f"Method: {self.method}")
        print(f"Partitions: {self.number_partition}")
        print("="*60)
        
        print("\nStep 1: Assigning partitions...")
        self.assign_partitions()
        
        print("\nStep 2: Measuring partition overhead...")
        overhead_results = self.measure_partition_overhead()
        
        print("\nStep 3: Creating partitioned subgraphs...")
        self.create_partitioned_subgraphs()
        
        print("\nStep 4: Saving results...")
        self.save_overhead_results(overhead_results)
        
        print('\nPartitioning analysis completed!')
        return overhead_results


def run_multiple_experiments():
    """Run experiments on multiple datasets and partition counts."""
    
    datasets = ['reddit', 'flickr', 'yelp', 'cora', 'citeseer', 'pubmed']
    partition_counts = [2, 4, 8, 16]
    
    all_results = []
    
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"PROCESSING DATASET: {dataset.upper()}")
        print(f"{'='*60}")
        
        for num_partitions in partition_counts:
            try:
                print(f"\n--- Testing {num_partitions} partitions ---")
                
                analyzer = MaxKGNNPartitioningOverhead(
                    dataset=dataset,
                    data_path="./data/",
                    output_path=f"./partition_analysis/{dataset}_{num_partitions}partitions/",
                    number_partition=num_partitions,
                    method='METIS',
                    selfloop=False
                )
                
                results = analyzer.run()
                all_results.append(results)
                
                print(f"Overhead for {dataset} with {num_partitions} partitions: {results['overhead_ratio']*100:.2f}%")
                
            except Exception as e:
                print(f"Error processing {dataset} with {num_partitions} partitions: {e}")
                continue
    
    # Save combined results
    combined_file = "./partition_analysis/combined_overhead_results.json"
    os.makedirs("./partition_analysis/", exist_ok=True)
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Create summary CSV
    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv("./partition_analysis/combined_summary.csv", index=False)
    
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Results saved to: {combined_file}")
    print("\nOverhead by dataset and partition count:")
    
    for dataset in datasets:
        dataset_results = [r for r in all_results if r['dataset'] == dataset]
        if dataset_results:
            print(f"\n{dataset}:")
            for result in sorted(dataset_results, key=lambda x: x['number_partitions']):
                print(f"  {result['number_partitions']} partitions: {result['overhead_ratio']*100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MaxK-GNN Graph Partitioning Overhead Analysis')
    parser.add_argument('--dataset', type=str, default='reddit', 
                       choices=['reddit', 'flickr', 'yelp', 'cora', 'citeseer', 'pubmed', 
                               'ogbn-arxiv', 'ogbn-products', 'ogbn-proteins'],
                       help='Dataset name')
    parser.add_argument('--data_path', type=str, default='./data/', 
                       help='Path to store/load datasets')
    parser.add_argument('--number_partition', type=int, default=4, 
                       help='Number of partitions')
    parser.add_argument('--method', type=str, default='METIS', 
                       choices=['METIS', 'Random'],
                       help='Partitioning method')
    parser.add_argument('--selfloop', action='store_true', 
                       help='Add self-loops to the graph')
    parser.add_argument('--run_all', action='store_true',
                       help='Run experiments on all datasets with multiple partition counts')
    
    args = parser.parse_args()
    
    if args.run_all:
        run_multiple_experiments()
    else:
        analyzer = MaxKGNNPartitioningOverhead(
            dataset=args.dataset,
            data_path=args.data_path,
            output_path=f"./partition_analysis/{args.dataset}_{args.number_partition}partitions/",
            number_partition=args.number_partition,
            method=args.method,
            selfloop=args.selfloop
        )
        
        results = analyzer.run()
        
        print(f"\n{'='*60}")
        print("FINAL SUMMARY")
        print(f"{'='*60}")
        print(f"Dataset: {args.dataset}")
        print(f"Partitioning method: {args.method}")
        print(f"Number of partitions: {args.number_partition}")
        print(f"Memory overhead ratio: {results['overhead_ratio']:.4f} ({results['overhead_ratio']*100:.2f}%)")
        print(f"Cut ratio: {results['cut_ratio']:.4f} ({results['cut_ratio']*100:.2f}%)")
        print(f"Edge replication factor: {results['replication_factor']:.4f}")
        print(f"\nThis means partitioned data uses {results['overhead_ratio']*100:.2f}% more memory than original data")
