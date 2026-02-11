# JMT Sharding Benchmark Visualization Guide

This document explains how to run the JMT (Jellyfish Merkle Tree) sharding benchmark and visualize the results.

## Overview

The JMT sharding benchmark measures the performance of the Aptos blockchain's storage system with and without sharding enabled. It generates structured output that can be visualized to compare performance characteristics.

## Running the Benchmark

### 1. Build the Executor Benchmark

First, build the executor benchmark binary:

```bash
cargo build -p aptos-executor-benchmark --release
```

### 2. Run the JMT Sharding Benchmark

Execute the benchmark with your preferred parameters:

```bash
# Basic usage
./target/release/aptos-executor-benchmark jmt-sharding-benchmark \
  --num-accounts 1000 \
  --num-operations 10000 \
  --block-size 100 \
  --data-dir ./benchmark-data

# Example with more accounts and operations for a comprehensive benchmark
./target/release/aptos-executor-benchmark jmt-sharding-benchmark \
  --num-accounts 5000 \
  --num-operations 50000 \
  --block-size 200 \
  --data-dir ./large-benchmark-data
```

The benchmark will:
1. Run the test with storage sharding enabled
2. Run the test with storage sharding disabled (for comparison)
3. Generate a comparative report
4. Export results to JSON and text formats

## Output Files

After running the benchmark, you'll have these output files:

- `jmt_benchmark_report.txt`: Human-readable comparative report
- `jmt_benchmark_results.json`: Structured data in JSON format for visualization

## Visualizing the Results

### 1. Install Required Dependencies

The visualization script requires Python with the following packages:

```bash
pip install matplotlib pandas numpy
```

Or using conda:

```bash
conda install matplotlib pandas numpy
```

### 2. Run the Visualization Script

```bash
# Basic visualization (creates PNG files in current directory)
python3 execution/executor-benchmark/scripts/visualize_benchmark.py

# Specify custom input/output options
python3 execution/executor-benchmark/scripts/visualize_benchmark.py \
  --input ./jmt_benchmark_results.json \
  --output-dir ./plots \
  --format png

# Display plots interactively
python3 execution/executor-benchmark/scripts/visualize_benchmark.py --show

# Create PDF plots
python3 execution/executor-benchmark/scripts/visualize_benchmark.py --format pdf
```

### 3. Generated Plots

The visualization script creates four types of plots:

1. **Performance Comparison**: Shows operations per second with and without sharding
2. **Latency Comparison**: Shows average latency per operation
3. **Storage Comparison**: Compares internal and leaf encoded bytes
4. **Dashboard**: Comprehensive summary with multiple metrics in one view

## Understanding the Results

### Performance Metrics

- **Operations per second**: Higher values indicate better performance
- **Average latency per operation**: Lower values indicate better performance
- **Total execution time**: Lower values indicate better performance

### Improvement Percentages

The comparative analysis shows:
- Positive percentages for throughput indicate sharding improves performance
- Negative percentages for latency indicate sharding reduces latency (which is good)
- Positive percentages for both indicate that sharding is beneficial

### Storage Metrics

- **Internal Encoded Bytes**: Size of internal nodes in the JMT
- **Leaf Encoded Bytes**: Size of leaf nodes in the JMT
- These metrics help understand storage efficiency

## Example Usage

Here's a complete workflow example:

```bash
# 1. Build the benchmark
cargo build -p aptos-executor-benchmark --release

# 2. Create a temporary directory and run the benchmark
mkdir benchmark-run && cd benchmark-run
../target/release/aptos-executor-benchmark jmt-sharding-benchmark \
  --num-accounts 2000 \
  --num-operations 20000 \
  --block-size 150 \
  --data-dir ./data

# 3. Visualize the results
python3 ../execution/executor-benchmark/scripts/visualize_benchmark.py \
  --output-dir ./plots \
  --format png

# 4. View the generated plots in the ./plots directory
ls plots/
```

## Troubleshooting

### Common Issues

1. **Missing JSON file**: Make sure you ran the benchmark before attempting visualization
2. **Python dependencies**: Ensure matplotlib, pandas, and numpy are installed
3. **Permission errors**: Check that you have write permissions in the output directory

### Memory Requirements

The benchmark may require significant memory for larger datasets. Adjust the parameters accordingly:
- `--num-accounts`: Number of accounts to create
- `--num-operations`: Number of operations to perform
- `--block-size`: Size of each block

## Customization

You can modify the visualization script to create custom charts or focus on specific metrics by editing `execution/executor-benchmark/scripts/visualize_benchmark.py`.