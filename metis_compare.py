import os
import numpy as np
import pandas as pd
import json
import time
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch
import dgl
from collections import defaultdict
from multiprocessing import cpu_count

# Import MaxK-GNN dataset loaders
from dgl import AddSelfLoop
from dgl.data import RedditDataset


def load_reddit_dataset(data_path="./data/"):
    """Load Reddit dataset."""
    print("Loading Reddit dataset...")
    start_time = time.time()
    
    data = RedditDataset(raw_dir=data_path)
    g = data[0]
    g = g.long()  # Convert to int64 for METIS
    
    load_time = time.time() - start_time
    print(f"Dataset loaded in {load_time:.2f}s: {g.num_nodes():,} nodes, {g.num_edges():,} edges")
    
    return g


def partition_subgraph_worker(args):
    """Worker function to partition a subgraph in parallel."""
    subgraph_nodes, original_graph_edges, target_partitions, worker_id = args
    
    print(f"  Worker {worker_id}: Creating subgraph with {len(subgraph_nodes):,} nodes...")
    
    # Create subgraph from node list
    subgraph_nodes_tensor = torch.tensor(subgraph_nodes, dtype=torch.long)
    subgraph = dgl.node_subgraph(original_graph_edges, subgraph_nodes_tensor)
    
    # Ensure int64 for METIS
    if subgraph.idtype != torch.int64:
        subgraph = subgraph.long()
    
    print(f"  Worker {worker_id}: Subgraph created, partitioning into {target_partitions} parts...")
    start_time = time.time()
    
    # Partition the subgraph
    partitions = dgl.metis_partition_assignment(subgraph, 
                                               target_partitions, 
                                               balance_edges=True, 
                                               mode='k-way', 
                                               objtype='cut')
    
    partition_time = time.time() - start_time
    print(f"  Worker {worker_id}: Partitioning completed in {partition_time:.2f}s")
    
    # Map subgraph node IDs back to original graph node IDs
    original_node_mapping = subgraph.ndata[dgl.NID].numpy()
    
    # Create partition assignment for original nodes
    partition_assignment = {}
    for sub_node_id, partition_id in enumerate(partitions.numpy()):
        original_node_id = original_node_mapping[sub_node_id]
        # Offset partition IDs for worker (worker 0: partitions 0,1; worker 1: partitions 2,3)
        final_partition_id = worker_id * target_partitions + partition_id
        partition_assignment[int(original_node_id)] = int(final_partition_id)
    
    return partition_assignment, partition_time, subgraph.num_edges()


def calculate_partition_quality(graph, partition_assignment, num_partitions):
    """Calculate partition quality metrics."""
    
    # Get edges
    src_nodes, dst_nodes = graph.edges()
    src_nodes = src_nodes.numpy()
    dst_nodes = dst_nodes.numpy()
    
    # Count edge cuts and partition balance
    cut_edges = 0
    internal_edges = 0
    partition_node_counts = [0] * num_partitions
    partition_edge_counts = [0] * num_partitions
    
    # Count nodes per partition
    for node_id, partition_id in partition_assignment.items():
        partition_node_counts[partition_id] += 1
    
    # Count edges and cuts
    for src, dst in zip(src_nodes, dst_nodes):
        src_partition = partition_assignment[int(src)]
        dst_partition = partition_assignment[int(dst)]
        
        if src_partition == dst_partition:
            internal_edges += 1
            partition_edge_counts[src_partition] += 1
        else:
            cut_edges += 1
    
    total_edges = len(src_nodes)
    cut_ratio = cut_edges / total_edges if total_edges > 0 else 0
    
    # Calculate balance metrics
    avg_nodes_per_partition = len(partition_assignment) / num_partitions
    max_deviation = max(abs(count - avg_nodes_per_partition) for count in partition_node_counts)
    balance_ratio = 1.0 - (max_deviation / avg_nodes_per_partition) if avg_nodes_per_partition > 0 else 0
    
    return {
        'cut_edges': cut_edges,
        'cut_ratio': cut_ratio,
        'internal_edges': internal_edges,
        'balance_ratio': balance_ratio,
        'partition_node_counts': partition_node_counts,
        'partition_edge_counts': partition_edge_counts
    }


def scenario_a_hierarchical_partitioning(graph, target_partitions=4):
    """
    Scenario A: Sequential then Parallel
    1. Partition graph into 2 parts sequentially
    2. Partition each part into 2 parts in parallel
    Final result: 4 parts total
    """
    print("\n" + "="*80)
    print("SCENARIO A: HIERARCHICAL PARTITIONING (Sequential → Parallel)")
    print("="*80)
    
    total_start_time = time.time()
    
    # Step 1: Sequential partitioning into 2 parts
    print("\nStep 1: Sequential partitioning into 2 parts...")
    step1_start = time.time()
    
    # Ensure graph is int64
    if graph.idtype != torch.int64:
        graph = graph.long()
    
    initial_partitions = dgl.metis_partition_assignment(graph, 
                                                       2, 
                                                       balance_edges=True, 
                                                       mode='k-way', 
                                                       objtype='cut')
    
    step1_time = time.time() - step1_start
    print(f"Step 1 completed in {step1_time:.2f}s")
    
    # Create node lists for each partition
    partition_0_nodes = []
    partition_1_nodes = []
    
    for node_id, partition_id in enumerate(initial_partitions.numpy()):
        if partition_id == 0:
            partition_0_nodes.append(node_id)
        else:
            partition_1_nodes.append(node_id)
    
    print(f"  Partition 0: {len(partition_0_nodes):,} nodes")
    print(f"  Partition 1: {len(partition_1_nodes):,} nodes")
    
    # Step 2: Parallel partitioning of each part into 2 sub-parts
    print(f"\nStep 2: Parallel partitioning (each part → 2 sub-parts)...")
    step2_start = time.time()
    
    # Prepare arguments for parallel execution
    partition_args = [
        (partition_0_nodes, graph, 2, 0),  # Worker 0: partition partition_0 into 2 parts
        (partition_1_nodes, graph, 2, 1),  # Worker 1: partition partition_1 into 2 parts
    ]
    
    # Execute in parallel
    all_assignments = {}
    total_subgraph_edges = 0
    parallel_times = []
    
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(partition_subgraph_worker, args) for args in partition_args]
        
        for future in as_completed(futures):
            assignment, partition_time, subgraph_edges = future.result()
            all_assignments.update(assignment)
            parallel_times.append(partition_time)
            total_subgraph_edges += subgraph_edges
    
    step2_time = max(parallel_times)  # Parallel time is the max of all workers
    print(f"Step 2 completed in {step2_time:.2f}s (parallel execution)")
    
    total_time = time.time() - total_start_time
    
    # Calculate quality metrics
    quality = calculate_partition_quality(graph, all_assignments, target_partitions)
    
    print(f"\nScenario A Results:")
    print(f"  Total time: {total_time:.2f}s (Step 1: {step1_time:.2f}s + Step 2: {step2_time:.2f}s)")
    print(f"  Cut edges: {quality['cut_edges']:,}")
    print(f"  Cut ratio: {quality['cut_ratio']:.4f} ({quality['cut_ratio']*100:.2f}%)")
    print(f"  Balance ratio: {quality['balance_ratio']:.4f}")
    print(f"  Partition sizes: {quality['partition_node_counts']}")
    
    return {
        'scenario': 'A_Hierarchical',
        'total_time': total_time,
        'step1_time': step1_time,
        'step2_time': step2_time,
        'partition_assignment': all_assignments,
        'quality': quality
    }


def scenario_b_direct_partitioning(graph, target_partitions=4):
    """
    Scenario B: Direct partition-4
    Partition graph into 4 parts directly
    """
    print("\n" + "="*80)
    print("SCENARIO B: DIRECT PARTITIONING")
    print("="*80)
    
    total_start_time = time.time()
    
    print(f"\nDirect partitioning into {target_partitions} parts...")
    
    # Ensure graph is int64
    if graph.idtype != torch.int64:
        graph = graph.long()
    
    # Direct partitioning
    partitions = dgl.metis_partition_assignment(graph, 
                                               target_partitions, 
                                               balance_edges=True, 
                                               mode='k-way', 
                                               objtype='cut')
    
    total_time = time.time() - total_start_time
    
    # Create partition assignment dictionary
    partition_assignment = {}
    partitions_np = partitions.numpy()
    for node_id, partition_id in enumerate(partitions_np):
        partition_assignment[node_id] = int(partition_id)
    
    # Calculate quality metrics
    quality = calculate_partition_quality(graph, partition_assignment, target_partitions)
    
    print(f"\nScenario B Results:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Cut edges: {quality['cut_edges']:,}")
    print(f"  Cut ratio: {quality['cut_ratio']:.4f} ({quality['cut_ratio']*100:.2f}%)")
    print(f"  Balance ratio: {quality['balance_ratio']:.4f}")
    print(f"  Partition sizes: {quality['partition_node_counts']}")
    
    return {
        'scenario': 'B_Direct',
        'total_time': total_time,
        'partition_assignment': partition_assignment,
        'quality': quality
    }


def compare_scenarios(results_a, results_b):
    """Compare results from both scenarios."""
    print("\n" + "="*80)
    print("COMPARISON: HIERARCHICAL vs DIRECT PARTITIONING")
    print("="*80)
    
    # Time comparison
    time_a = results_a['total_time']
    time_b = results_b['total_time']
    time_speedup = time_b / time_a if time_a > 0 else float('inf')
    
    print(f"\n📊 TIME COMPARISON:")
    print(f"  Scenario A (Hierarchical): {time_a:.2f}s")
    print(f"  Scenario B (Direct):       {time_b:.2f}s")
    if time_speedup > 1:
        print(f"  ⚡ Scenario A is {time_speedup:.2f}x FASTER than Scenario B")
    else:
        print(f"  ⚡ Scenario B is {1/time_speedup:.2f}x FASTER than Scenario A")
    
    # Quality comparison  
    cut_a = results_a['quality']['cut_edges']
    cut_b = results_b['quality']['cut_edges']
    cut_improvement = (cut_a - cut_b) / cut_a * 100 if cut_a > 0 else 0
    
    balance_a = results_a['quality']['balance_ratio']
    balance_b = results_b['quality']['balance_ratio']
    
    print(f"\n🎯 QUALITY COMPARISON:")
    print(f"  Cut Edges:")
    print(f"    Scenario A (Hierarchical): {cut_a:,}")
    print(f"    Scenario B (Direct):       {cut_b:,}")
    if cut_improvement > 0:
        print(f"    🏆 Scenario B has {cut_improvement:.1f}% FEWER cut edges")
    else:
        print(f"    🏆 Scenario A has {-cut_improvement:.1f}% FEWER cut edges")
    
    print(f"  Balance Ratio:")
    print(f"    Scenario A (Hierarchical): {balance_a:.4f}")
    print(f"    Scenario B (Direct):       {balance_b:.4f}")
    if balance_b > balance_a:
        print(f"    🏆 Scenario B has BETTER balance")
    else:
        print(f"    🏆 Scenario A has BETTER balance")
    
    # Summary
    print(f"\n📈 SUMMARY:")
    print(f"  Time Winner:    {'Scenario A (Hierarchical)' if time_speedup > 1 else 'Scenario B (Direct)'}")
    print(f"  Quality Winner: {'Scenario B (Direct)' if cut_improvement > 0 else 'Scenario A (Hierarchical)'}")
    
    # Determine overall winner
    if time_speedup > 1.2 and cut_improvement < 5:
        print(f"  🎉 OVERALL WINNER: Scenario A (Hierarchical) - Much faster with acceptable quality loss")
    elif cut_improvement > 10 and time_speedup < 2:
        print(f"  🎉 OVERALL WINNER: Scenario B (Direct) - Much better quality with acceptable time cost")
    else:
        print(f"  🤝 RESULT: Close call - depends on priority (speed vs quality)")
    
    return {
        'time_speedup_a_vs_b': time_speedup,
        'cut_improvement_b_vs_a_percent': cut_improvement,
        'balance_difference': balance_b - balance_a
    }


def save_results(results_a, results_b, comparison, output_file="partition_comparison_results.json"):
    """Save comparison results to file."""
    
    # Prepare results for JSON serialization (remove non-serializable objects)
    results_to_save = {
        'scenario_a': {
            'scenario': results_a['scenario'],
            'total_time': results_a['total_time'],
            'step1_time': results_a.get('step1_time', 0),
            'step2_time': results_a.get('step2_time', 0),
            'quality': results_a['quality']
        },
        'scenario_b': {
            'scenario': results_b['scenario'],
            'total_time': results_b['total_time'],
            'quality': results_b['quality']
        },
        'comparison': comparison
    }
    
    with open(output_file, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")


def main():
    """Main function to run the comparison."""
    
    # Set up multiprocessing
    os.environ['OMP_NUM_THREADS'] = str(min(cpu_count(), 8))
    
    print("🚀 HIERARCHICAL vs DIRECT PARTITIONING COMPARISON")
    print("📊 Dataset: Reddit")
    print("🎯 Target: 4 partitions")
    print("⚡ Using multi-threading for METIS")
    
    # Load dataset
    graph = load_reddit_dataset()
    
    # Run both scenarios
    print("\n🔄 Running both scenarios...")
    
    # Scenario A: Hierarchical (Sequential → Parallel)
    results_a = scenario_a_hierarchical_partitioning(graph, target_partitions=4)
    
    # Scenario B: Direct partitioning
    results_b = scenario_b_direct_partitioning(graph, target_partitions=4)
    
    # Compare results
    comparison = compare_scenarios(results_a, results_b)
    
    # Save results
    save_results(results_a, results_b, comparison)
    
    print("\n✅ Comparison completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare Hierarchical vs Direct Partitioning')
    parser.add_argument('--data_path', type=str, default='./data/', 
                       help='Path to store/load datasets')
    
    args = parser.parse_args()
    
    # Update global data path if provided
    globals()['data_path'] = args.data_path
    
    main()
