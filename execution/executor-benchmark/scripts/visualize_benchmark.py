#!/usr/bin/env python3
"""
Visualization script for JMT sharding benchmark results.
This script reads the JSON output from the benchmark and creates visualizations.
"""

import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import os
from pathlib import Path

def load_benchmark_results(filepath='jmt_benchmark_results.json'):
    """Load benchmark results from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def create_performance_comparison_chart(results_data):
    """Create a bar chart comparing performance with and without sharding."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Extract data
    with_sharding = results_data['with_sharding']
    without_sharding = results_data['without_sharding']
    
    labels = ['With Sharding', 'Without Sharding']
    ops_per_sec = [
        with_sharding['operations_per_second'],
        without_sharding['operations_per_second']
    ]
    
    bars = ax.bar(labels, ops_per_sec, color=['#2E8B57', '#CD5C5C'])
    
    # Add value labels on top of bars
    for bar, value in zip(bars, ops_per_sec):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}',
                ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Operations per Second')
    ax.set_title('JMT Sharding Performance Comparison\n(Higher is Better)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Rotate x-axis labels if needed
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig

def create_latency_comparison_chart(results_data):
    """Create a bar chart comparing latency with and without sharding."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Extract data
    with_sharding = results_data['with_sharding']
    without_sharding = results_data['without_sharding']
    
    labels = ['With Sharding', 'Without Sharding']
    avg_latency_ns = [
        with_sharding['avg_latency_per_operation'],
        without_sharding['avg_latency_per_operation']
    ]
    
    bars = ax.bar(labels, avg_latency_ns, color=['#2E8B57', '#CD5C5C'])
    
    # Add value labels on top of bars
    for bar, value in zip(bars, avg_latency_ns):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}ns',
                ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Average Latency per Operation (nanoseconds)')
    ax.set_title('JMT Sharding Latency Comparison\n(Lower is Better)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Rotate x-axis labels if needed
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig

def create_storage_comparison_chart(results_data):
    """Create a grouped bar chart comparing storage metrics."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Extract data
    with_sharding = results_data['with_sharding']
    without_sharding = results_data['without_sharding']
    
    x = np.arange(2)  # With and without sharding
    width = 0.35
    
    internal_bytes = [
        with_sharding['total_internal_encoded_bytes'],
        without_sharding['total_internal_encoded_bytes']
    ]
    leaf_bytes = [
        with_sharding['total_leaf_encoded_bytes'],
        without_sharding['total_leaf_encoded_bytes']
    ]
    
    bars1 = ax.bar(x - width/2, internal_bytes, width, label='Internal Encoded Bytes', color='#4682B4')
    bars2 = ax.bar(x + width/2, leaf_bytes, width, label='Leaf Encoded Bytes', color='#FFA500')
    
    # Add value labels on top of bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:,}',
                    ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('Encoded Bytes')
    ax.set_title('JMT Storage Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['With Sharding', 'Without Sharding'])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    return fig

def create_summary_dashboard(results_data):
    """Create a summary dashboard with multiple metrics."""
    fig = plt.figure(figsize=(16, 12))
    
    # Grid layout: 2 rows, 2 columns
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Performance comparison
    ax1 = fig.add_subplot(gs[0, 0])
    with_sharding = results_data['with_sharding']
    without_sharding = results_data['without_sharding']
    
    labels = ['With Sharding', 'Without Sharding']
    ops_per_sec = [
        with_sharding['operations_per_second'],
        without_sharding['operations_per_second']
    ]
    
    bars1 = ax1.bar(labels, ops_per_sec, color=['#2E8B57', '#CD5C5C'])
    for bar, value in zip(bars1, ops_per_sec):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1f}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax1.set_ylabel('Ops/sec')
    ax1.set_title('Throughput Comparison')
    ax1.grid(axis='y', alpha=0.3)
    
    # Latency comparison
    ax2 = fig.add_subplot(gs[0, 1])
    avg_latency_ns = [
        with_sharding['avg_latency_per_operation'],
        without_sharding['avg_latency_per_operation']
    ]
    
    bars2 = ax2.bar(labels, avg_latency_ns, color=['#2E8B57', '#CD5C5C'])
    for bar, value in zip(bars2, avg_latency_ns):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.0f}ns',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax2.set_ylabel('Latency (ns)')
    ax2.set_title('Latency Comparison')
    ax2.grid(axis='y', alpha=0.3)
    
    # Duration comparison
    ax3 = fig.add_subplot(gs[1, 0])
    duration_ms = [
        with_sharding['duration_ms'],
        without_sharding['duration_ms']
    ]
    
    bars3 = ax3.bar(labels, duration_ms, color=['#2E8B57', '#CD5C5C'])
    for bar, value in zip(bars3, duration_ms):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{value}ms',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax3.set_ylabel('Duration (ms)')
    ax3.set_title('Total Execution Time')
    ax3.grid(axis='y', alpha=0.3)
    
    # Comparative improvement
    ax4 = fig.add_subplot(gs[1, 1])
    comp_analysis = results_data['comparative_analysis']
    
    improvement_metrics = ['Throughput', 'Latency']
    improvement_values = [
        comp_analysis['throughput_improvement_percentage'],
        comp_analysis['latency_improvement_percentage']
    ]
    
    colors = ['#2E8B57' if val >= 0 else '#CD5C5C' for val in improvement_values]
    bars4 = ax4.bar(improvement_metrics, improvement_values, color=colors)
    
    for bar, value in zip(bars4, improvement_values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:+.1f}%',
                ha='center', va='bottom' if height >= 0 else 'top', 
                fontweight='bold', fontsize=10)
    
    ax4.set_ylabel('Improvement (%)')
    ax4.set_title('Improvement with Sharding\n(Positive = Better Performance)')
    ax4.grid(axis='y', alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    # Overall title
    fig.suptitle('JMT Sharding Benchmark - Comprehensive Dashboard', fontsize=16, fontweight='bold')
    
    return fig

def print_text_summary(results_data):
    """Print a text-based summary of the results."""
    print("="*60)
    print("JMT SHARDING BENCHMARK SUMMARY")
    print("="*60)
    
    with_sharding = results_data['with_sharding']
    without_sharding = results_data['without_sharding']
    comp_analysis = results_data['comparative_analysis']
    
    print(f"\nConfiguration:")
    print(f"  - Number of accounts: {with_sharding['num_accounts']:,}")
    print(f"  - Number of operations: {with_sharding['num_operations']:,}")
    print(f"  - Block size: {with_sharding['block_size']}")
    
    print(f"\nPerformance Results:")
    print(f"  With Sharding:")
    print(f"    - Operations/sec: {with_sharding['operations_per_second']:,.2f}")
    print(f"    - Avg latency/op: {with_sharding['avg_latency_per_operation']:.2f} ns")
    print(f"    - Total duration: {with_sharding['duration_ms']} ms")
    
    print(f"  Without Sharding:")
    print(f"    - Operations/sec: {without_sharding['operations_per_second']:,.2f}")
    print(f"    - Avg latency/op: {without_sharding['avg_latency_per_operation']:.2f} ns")
    print(f"    - Total duration: {without_sharding['duration_ms']} ms")
    
    print(f"\nComparative Analysis:")
    print(f"  - Throughput improvement: {comp_analysis['throughput_improvement_percentage']:+.2f}%")
    print(f"  - Latency improvement: {comp_analysis['latency_improvement_percentage']:+.2f}%")
    
    print(f"\nStorage Metrics:")
    print(f"  With Sharding:")
    print(f"    - Internal encoded bytes: {with_sharding['total_internal_encoded_bytes']:,}")
    print(f"    - Leaf encoded bytes: {with_sharding['total_leaf_encoded_bytes']:,}")
    
    print(f"  Without Sharding:")
    print(f"    - Internal encoded bytes: {without_sharding['total_internal_encoded_bytes']:,}")
    print(f"    - Leaf encoded bytes: {without_sharding['total_leaf_encoded_bytes']:,}")

def main():
    parser = argparse.ArgumentParser(description='Visualize JMT sharding benchmark results')
    parser.add_argument('--input', '-i', default='jmt_benchmark_results.json', 
                       help='Input JSON file with benchmark results')
    parser.add_argument('--output-dir', '-o', default='.', 
                       help='Output directory for plots')
    parser.add_argument('--format', '-f', choices=['png', 'pdf', 'svg'], 
                       default='png', help='Output format for plots')
    parser.add_argument('--show', action='store_true', 
                       help='Display plots interactively')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        print("Make sure to run the JMT sharding benchmark first to generate the results file.")
        return 1
    
    # Load results
    results_data = load_benchmark_results(args.input)
    
    # Print text summary
    print_text_summary(results_data)
    
    # Create visualizations
    perf_fig = create_performance_comparison_chart(results_data)
    latency_fig = create_latency_comparison_chart(results_data)
    storage_fig = create_storage_comparison_chart(results_data)
    dashboard_fig = create_summary_dashboard(results_data)
    
    # Save visualizations
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    perf_fig.savefig(output_dir / f'jmt_performance_comparison.{args.format}', dpi=300, bbox_inches='tight')
    latency_fig.savefig(output_dir / f'jmt_latency_comparison.{args.format}', dpi=300, bbox_inches='tight')
    storage_fig.savefig(output_dir / f'jmt_storage_comparison.{args.format}', dpi=300, bbox_inches='tight')
    dashboard_fig.savefig(output_dir / f'jmt_dashboard.{args.format}', dpi=300, bbox_inches='tight')
    
    print(f"\nVisualizations saved to '{args.output_dir}/' directory:")
    print(f"  - jmt_performance_comparison.{args.format}")
    print(f"  - jmt_latency_comparison.{args.format}")
    print(f"  - jmt_storage_comparison.{args.format}")
    print(f"  - jmt_dashboard.{args.format}")
    
    # Show plots if requested
    if args.show:
        plt.show()
    
    return 0

if __name__ == '__main__':
    exit(main())