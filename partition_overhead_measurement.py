import os
import numpy as np
import pandas as pd
import pickle
import json
import csv
from collections import defaultdict, Counter
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
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
from tqdm import tqdm

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
        g = g.long()  # Convert to int64 for METIS
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
        g = g.long()  # Convert to int64 for METIS
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


def process_edge_batch(args):
    """Process a batch of edges for partition assignment - optimized for multiprocessing."""
    edge_batch, v2p = args
    
    batch_results = {
        'internal_edges': 0,
        'cross_partition_edges': 0,
        'partition_edge_counts': defaultdict(int),
        'edge_assignments': defaultdict(set)
    }
    
    for src, dst in edge_batch:
        src, dst = int(src), int(dst)
        src_partition = v2p[src]
        dst_partition = v2p[dst]
        
        # Create normalized edge key (smaller node first)
        edge_key = (min(src, dst), max(src, dst))
        
        if src_partition == dst_partition:
            # Internal edge
            batch_results['internal_edges'] += 1
            batch_results['partition_edge_counts'][src_partition] += 1
            batch_results['edge_assignments'][edge_key].add(src_partition)
        else:
            # Cross-partition edge
            batch_results['cross_partition_edges'] += 1
            batch_results['partition_edge_counts'][src_partition] += 1
            batch_results['partition_edge_counts'][dst_partition] += 1
            batch_results['edge_assignments'][edge_key].add(src_partition)
            batch_results['edge_assignments'][edge_key].add(dst_partition)
    
    return batch_results


class FastMaxKGNNPartitioningOverhead():
    """Fast, multi-threaded graph partitioning overhead measurement for MaxK-GNN datasets."""

    def __init__(self, dataset: str = None, 
                 data_path: str = "./data/", 
                 output_path: str = None, 
                 number_partition: int = 4, 
                 method: str = 'METIS',
                 selfloop: bool = False,
                 num_workers: int = None,
                 batch_size: int = 50000,
                 seed: int = 42):
        super().__init__()
        """Initialization."""
        
        self.dataset = dataset
        self.data_path = data_path
        self.output_path = output_path or f"./partition_analysis/{dataset}/"
        self.number_partition = number_partition
        self.method = method
        self.selfloop = selfloop
        self.num_workers = num_workers or min(cpu_count(), 8)  # Limit to avoid memory issues
        self.batch_size = batch_size
        self.seed = seed
        self._set_seed()
        
        # Create output directory
        os.makedirs(self.output_path, exist_ok=True)
        
        # Set OpenMP threads for METIS
        os.environ['OMP_NUM_THREADS'] = str(self.num_workers)
        
        print(f"Using {self.num_workers} workers for parallel processing")
        print(f"Batch size: {self.batch_size}")
    
    def _set_seed(self):
        """Creating the initial random seed."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def load_dataset_fast(self):
        """Load dataset with optimizations."""
        print(f"Loading dataset: {self.dataset}")
        start_time = time.time()
        
        self.g, self.features, self.labels, self.masks, self.num_classes = load_maxkgnn_dataset(
            self.dataset, self.data_path, self.selfloop
        )
        
        # Ensure int64 for METIS
        if self.g.idtype != torch.int64:
            print("Converting graph to int64...")
            self.g = self.g.long()
        
        # Add self-loop if specified
        if self.selfloop:
            self.g = dgl.add_self_loop(self.g)
            if self.g.idtype != torch.int64:
                self.g = self.g.long()
        
        self.number_edges = self.g.num_edges()
        self.number_nodes = self.g.num_nodes()
        
        load_time = time.time() - start_time
        print(f"Dataset loaded in {load_time:.2f}s: {self.number_nodes:,} nodes, {self.number_edges:,} edges")
        print(f"Features shape: {self.features.shape}")
        print(f"Number of classes: {self.num_classes}")

    def assign_partitions_fast(self):
        """Fast partition assignment with proper threading."""
        print(f"\nStep 1: Assigning partitions using {self.method}...")
        start_time = time.time()
        
        if self.method == 'METIS':
            print("Calling METIS partitioning (multi-threaded)...")
            partitions = dgl.metis_partition_assignment(self.g, 
                                                        self.number_partition, 
                                                        balance_edges=True, 
                                                        mode='k-way', 
                                                        objtype='cut')
            print("METIS partitioning completed!")
        elif self.method == 'Random':
            partitions = torch.randint(0, self.number_partition, (self.number_nodes,))
        else:
            raise ValueError(f"Unknown partitioning method: {self.method}")
        
        # Convert to dictionary for fast lookup
        self.v2p = {}
        partitions_np = partitions.numpy()
        for i in range(len(partitions_np)):
            self.v2p[i] = int(partitions_np[i])
        
        partition_time = time.time() - start_time
        print(f"Partitioning completed in {partition_time:.2f}s")
        
        # Print partition balance
        partition_sizes = np.bincount(partitions_np, minlength=self.number_partition)
        print("Partition node distribution:")
        for i, size in enumerate(partition_sizes):
            print(f"  Partition {i}: {size:,} nodes ({size/self.number_nodes*100:.1f}%)")
        
        # Save partition assignment
        with open(os.path.join(self.output_path, 'partition_assignment.json'), 'w') as f:
            json.dump(self.v2p, f, indent=2)

    def measure_partition_overhead_fast(self):
        """Fast multi-threaded measurement of partition overhead."""
        print(f"\nStep 2: Measuring partition overhead (parallel processing)...")
        start_time = time.time()
        
        # Get edges as numpy arrays for faster processing
        src_nodes, dst_nodes = self.g.edges()
        src_nodes = src_nodes.numpy()
        dst_nodes = dst_nodes.numpy()
        
        print(f"Processing {len(src_nodes):,} edges with {self.num_workers} workers...")
        
        # Create edge batches for parallel processing
        edges = list(zip(src_nodes, dst_nodes))
        edge_batches = [edges[i:i + self.batch_size] for i in range(0, len(edges), self.batch_size)]
        
        print(f"Created {len(edge_batches)} batches of size ~{self.batch_size:,}")
        
        # Prepare arguments for multiprocessing
        batch_args = [(batch, self.v2p) for batch in edge_batches]
        
        # Process batches in parallel
        all_results = []
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks
            future_to_batch = {executor.submit(process_edge_batch, args): i 
                             for i, args in enumerate(batch_args)}
            
            # Collect results with progress bar
            with tqdm(total=len(edge_batches), desc="Processing edge batches") as pbar:
                for future in as_completed(future_to_batch):
                    batch_idx = future_to_batch[future]
                    try:
                        result = future.result()
                        all_results.append(result)
                        pbar.update(1)
                    except Exception as exc:
                        print(f'Batch {batch_idx} generated an exception: {exc}')
                        pbar.update(1)
        
        # Aggregate results
        print("Aggregating results...")
        total_internal = sum(r['internal_edges'] for r in all_results)
        total_cross = sum(r['cross_partition_edges'] for r in all_results)
        
        # Combine partition edge counts
        partition_edges = [0] * self.number_partition
        for result in all_results:
            for partition_id, count in result['partition_edge_counts'].items():
                partition_edges[partition_id] += count
        
        # Combine edge assignments for unique edge counting
        all_edge_assignments = defaultdict(set)
        for result in all_results:
            for edge_key, partitions in result['edge_assignments'].items():
                all_edge_assignments[edge_key].update(partitions)
        
        # Calculate final metrics
        unique_edges = len(all_edge_assignments)
        total_edges_stored = sum(partition_edges)
        internal_edges = sum(1 for partitions in all_edge_assignments.values() if len(partitions) == 1)
        cross_edges = unique_edges - internal_edges
        cross_partition_storage_instances = sum(len(partitions) for partitions in all_edge_assignments.values() if len(partitions) > 1)
        
        # Calculate overhead
        overhead_edges = total_edges_stored - unique_edges
        overhead_ratio = overhead_edges / unique_edges if unique_edges > 0 else 0
        avg_replication = total_edges_stored / unique_edges if unique_edges > 0 else 0
        cut_ratio = cross_edges / unique_edges if unique_edges > 0 else 0
        
        processing_time = time.time() - start_time
        print(f"Overhead measurement completed in {processing_time:.2f}s")
        
        print(f"\n=== MEMORY OVERHEAD ANALYSIS ===")
        print(f"Original unique edges: {unique_edges:,}")
        print(f"Total edges stored across partitions: {total_edges_stored:,}")
        print(f"Overhead edges (duplications): {overhead_edges:,}")
        print(f"Overhead ratio: {overhead_ratio:.4f} ({overhead_ratio*100:.2f}%)")
        print(f"\nDetailed breakdown:")
        print(f"- Internal edges (within partition): {internal_edges:,}")
        print(f"- Cross-partition edges (unique): {cross_edges:,}")
        print(f"- Cross-partition storage instances: {cross_partition_storage_instances:,}")
        
        for i in range(self.number_partition):
            print(f"- Partition {i}: {partition_edges[i]:,} edges ({partition_edges[i]/total_edges_stored*100:.1f}%)")
        
        print(f"- Average edge replication factor: {avg_replication:.4f}")
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
            'processing_time_seconds': processing_time,
            'num_workers': self.num_workers,
            'batch_size': self.batch_size,
            'original_nodes': self.number_nodes,
            'original_edges': self.number_edges,
            'unique_edges': unique_edges,
            'total_partitioned_edges': total_edges_stored,
            'overhead_edges': overhead_edges,
            'overhead_ratio': overhead_ratio,
            'internal_edges': internal_edges,
            'cross_partition_edges': cross_edges,
            'cross_partition_storage_instances': cross_partition_storage_instances,
            'partition_edge_counts': partition_edges,
            'replication_factor': avg_replication,
            'cut_ratio': cut_ratio,
            'partition_node_distribution': [sum(1 for p in self.v2p.values() if p == i) 
                                          for i in range(self.number_partition)]
        }

    def create_partitioned_subgraphs_fast(self):
        """Create and save partitioned subgraphs efficiently."""
        print(f"\nStep 3: Creating partitioned subgraphs...")
        start_time = time.time()
        
        # Create subgraphs sequentially to avoid pickle issues
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
            
            print(f"  Partition {partition_id}: {subgraph.num_nodes():,} nodes, {subgraph.num_edges():,} edges")
        
        creation_time = time.time() - start_time
        print(f"Subgraph creation completed in {creation_time:.2f}s")

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
            'replication_factor': [results['replication_factor']],
            'processing_time_s': [results['processing_time_seconds']],
            'num_workers': [results['num_workers']]
        }
        
        df = pd.DataFrame(summary_data)
        df.to_csv(summary_file, index=False)
        
        print(f"\nResults saved to:")
        print(f"  Detailed: {results_file}")
        print(f"  Summary: {summary_file}")
    
    def run(self):
        """Run the fast partitioning algorithm with overhead measurement."""
        total_start = time.time()
        
        print("="*80)
        print(f"FAST PARTITIONING OVERHEAD ANALYSIS")
        print(f"Dataset: {self.dataset}")
        print(f"Method: {self.method}")
        print(f"Partitions: {self.number_partition}")
        print(f"Workers: {self.num_workers}")
        print("="*80)
        
        self.load_dataset_fast()
        self.assign_partitions_fast()
        overhead_results = self.measure_partition_overhead_fast()
        self.create_partitioned_subgraphs_fast()
        self.save_overhead_results(overhead_results)
        
        total_time = time.time() - total_start
        print(f'\nTotal analysis completed in {total_time:.2f}s!')
        print(f"Speedup achieved with {self.num_workers} workers")
        
        return overhead_results


def run_multiple_experiments_fast():
    """Run fast experiments on specified datasets and partition counts."""
    
    # Only test the specified datasets and partition counts
    datasets = ['reddit', 'yelp', 'flickr', 'ogbn-products', 'ogbn-proteins']
    partition_counts = [4, 8, 16]
    
    all_results = []
    total_start = time.time()
    
    print(f"Testing {len(datasets)} datasets with {len(partition_counts)} partition counts each")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Partition counts: {', '.join(map(str, partition_counts))}")
    
    for dataset in datasets:
        print(f"\n{'='*80}")
        print(f"PROCESSING DATASET: {dataset.upper()}")
        print(f"{'='*80}")
        
        for num_partitions in partition_counts:
            try:
                print(f"\n--- Testing {num_partitions} partitions ---")
                
                # Adjust workers and batch size based on dataset size
                if dataset in ['reddit', 'yelp', 'ogbn-products', 'ogbn-proteins']:
                    num_workers = min(cpu_count(), 8)
                    batch_size = 100000
                elif dataset in ['flickr']:
                    num_workers = min(cpu_count(), 6)
                    batch_size = 50000
                else:
                    num_workers = min(cpu_count(), 4)
                    batch_size = 20000
                
                analyzer = FastMaxKGNNPartitioningOverhead(
                    dataset=dataset,
                    data_path="./data/",
                    output_path=f"./partition_analysis/{dataset}_{num_partitions}partitions/",
                    number_partition=num_partitions,
                    method='METIS',
                    selfloop=False,
                    num_workers=num_workers,
                    batch_size=batch_size
                )
                
                results = analyzer.run()
                all_results.append(results)
                
                print(f"✓ {dataset} with {num_partitions} partitions: {results['overhead_ratio']*100:.2f}% overhead "
                      f"(processed in {results['processing_time_seconds']:.1f}s)")
                
            except Exception as e:
                print(f"❌ Error processing {dataset} with {num_partitions} partitions: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Save combined results
    combined_file = "./partition_analysis/combined_overhead_results.json"
    os.makedirs("./partition_analysis/", exist_ok=True)
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Create summary CSV
    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv("./partition_analysis/combined_summary.csv", index=False)
    
    total_time = time.time() - total_start
    
    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    print(f"Total experiments: {len(all_results)}/{len(datasets) * len(partition_counts)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Results saved to: {combined_file}")
    print("\nOverhead by dataset and partition count:")
    
    # Create a summary table
    print(f"\n{'Dataset':<15} {'4 Parts':<10} {'8 Parts':<10} {'16 Parts':<10}")
    print("-" * 50)
    
    for dataset in datasets:
        dataset_results = [r for r in all_results if r['dataset'] == dataset]
        if dataset_results:
            row = f"{dataset:<15}"
            for partition_count in partition_counts:
                result = next((r for r in dataset_results if r['number_partitions'] == partition_count), None)
                if result:
                    row += f" {result['overhead_ratio']*100:>7.2f}%  "
                else:
                    row += f" {'ERROR':<9}"
            print(row)
    
    print(f"\nDetailed results:")
    for dataset in datasets:
        dataset_results = [r for r in all_results if r['dataset'] == dataset]
        if dataset_results:
            print(f"\n{dataset}:")
            for result in sorted(dataset_results, key=lambda x: x['number_partitions']):
                print(f"  {result['number_partitions']} partitions: {result['overhead_ratio']*100:.2f}% overhead, "
                      f"{result['cut_ratio']*100:.2f}% cut ratio, "
                      f"processed in {result['processing_time_seconds']:.1f}s")
        else:
            print(f"\n{dataset}: No successful results")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fast MaxK-GNN Graph Partitioning Overhead Analysis')
    parser.add_argument('--dataset', type=str, default='reddit', 
                       choices=['reddit', 'flickr', 'yelp', 'ogbn-products', 'ogbn-proteins'],
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
    parser.add_argument('--num_workers', type=int, default=None,
                       help='Number of worker processes (default: auto)')
    parser.add_argument('--batch_size', type=int, default=50000,
                       help='Batch size for edge processing')
    parser.add_argument('--run_all', action='store_true',
                       help='Run experiments on all datasets with multiple partition counts')
    
    args = parser.parse_args()
    
    if args.run_all:
        run_multiple_experiments_fast()
    else:
        analyzer = FastMaxKGNNPartitioningOverhead(
            dataset=args.dataset,
            data_path=args.data_path,
            output_path=f"./partition_analysis/{args.dataset}_{args.number_partition}partitions/",
            number_partition=args.number_partition,
            method=args.method,
            selfloop=args.selfloop,
            num_workers=args.num_workers,
            batch_size=args.batch_size
        )
        
        results = analyzer.run()
        
        print(f"\n{'='*80}")
        print("FINAL SUMMARY")
        print(f"{'='*80}")
        print(f"Dataset: {args.dataset}")
        print(f"Partitioning method: {args.method}")
        print(f"Number of partitions: {args.number_partition}")
        print(f"Workers used: {results['num_workers']}")
        print(f"Processing time: {results['processing_time_seconds']:.2f}s")
        print(f"Memory overhead ratio: {results['overhead_ratio']:.4f} ({results['overhead_ratio']*100:.2f}%)")
        print(f"Cut ratio: {results['cut_ratio']:.4f} ({results['cut_ratio']*100:.2f}%)")
        print(f"Edge replication factor: {results['replication_factor']:.4f}")
        print(f"\n🚀 This means partitioned data uses {results['overhead_ratio']*100:.2f}% more memory than original data")
