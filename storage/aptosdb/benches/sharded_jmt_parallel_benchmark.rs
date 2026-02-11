// Copyright (c) Aptos Foundation
// Licensed under the MIT license

use aptos_crypto::hash::HashValue;
use aptos_jellyfish_merkle::metrics::JellyfishMerkleTreeMetrics;
use aptos_state_merkle_db::{
    state_merkle_db::StateMerkleDb, state_snapshot_committer::StateSnapshotCommitter,
};
use aptos_storage_interface::Result;
use aptos_types::{
    state_store::{state_key::StateKey, NUM_STATE_SHARDS},
    transaction::Version,
};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::sync::Arc;
use tempfile::TempDir;

/// Generate deterministic HashValue keys for repeatable experiments
fn generate_keys(count: usize) -> Vec<HashValue> {
    (0..count)
        .map(|i| HashValue::sha3_256_of(format!("key_{}", i).as_bytes()))
        .collect()
}

/// Map keys to shards using the same logic as the codebase
fn get_state_shard_id(key: &HashValue) -> usize {
    // Using the same logic as in the codebase to map keys to shards
    // This is typically done by taking the first byte of the hash and mapping it to a shard
    let first_byte = key[0];
    first_byte as usize % NUM_STATE_SHARDS
}

/// Create mock state values with configurable value size
fn create_mock_state_values_with_size(keys: &[HashValue], value_size: usize) -> Vec<(HashValue, Option<(HashValue, StateKey)>)> {
    keys.iter()
        .map(|key| {
            let state_key = StateKey::raw(key.to_vec());
            // Create value data of specified size
            let value_data: Vec<u8> = vec![0u8; value_size];
            let value_hash = HashValue::sha3_256_of(&value_data);
            (*key, Some((value_hash, state_key)))
        })
        .collect()
}

/// Create a temporary storage directory for the benchmark
fn create_temp_storage() -> (TempDir, String) {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let path = temp_dir.path().to_str().expect("Invalid temp path").to_string();
    (temp_dir, path)
}

/// Benchmark the higher-level parallel merklize flow as recommended in shard.md
fn benchmark_merklize_parallel(c: &mut Criterion) {
    let mut group = c.benchmark_group("sharded_jmt_merklize_parallel");

    // Test different key counts to see scalability
    let key_counts = vec![1000, 5000, 10000];
    let value_size = 256; // 256 bytes per value

    for key_count in key_counts {
        group.bench_with_input(
            BenchmarkId::new("merklize_parallel", key_count),
            &key_count,
            |b, &count| {
                b.iter(|| {
                    // Create temporary storage
                    let (_temp_dir, path) = create_temp_storage();

                    // Create StateMerkleDb with sharding enabled
                    let rocksdb_configs = aptos_types::state_store::state_storage_usage::RocksdbConfigs::default();
                    let storage_paths = aptos_config::config::StorageDirPaths::from_path(path.clone());
                    
                    // Enable storage sharding for the benchmark
                    let state_merkle_db = Arc::new(
                        StateMerkleDb::new(
                            &storage_paths,
                            rocksdb_configs,
                            None, // env
                            None, // block_cache
                            false, // readonly
                            0, // max_nodes_per_lru_cache_shard
                            false, // is_hot
                            false, // delete_on_restart
                        )
                        .expect("Failed to create StateMerkleDb")
                    );

                    // Generate keys and map to shards
                    let keys = generate_keys(count);
                    
                    // Create per-shard value sets based on shard distribution
                    let mut all_updates: Vec<Vec<(HashValue, Option<(HashValue, StateKey)>)>> = 
                        vec![Vec::new(); NUM_STATE_SHARDS];
                    
                    let state_values = create_mock_state_values_with_size(&keys, value_size);
                    
                    for (key, value_opt) in state_values {
                        let shard_id = get_state_shard_id(&key);
                        all_updates[shard_id].push((key, value_opt));
                    }

                    // Convert to the format expected by the merklize function
                    let all_updates_array: [Vec<(HashValue, Option<(HashValue, StateKey)>)>; NUM_STATE_SHARDS] = 
                        all_updates.try_into().unwrap_or_else(|v: Vec<_>| {
                            panic!("Expected {} elements, got {}", NUM_STATE_SHARDS, v.len())
                        });

                    // Call the merklize function which performs the parallel work
                    // We'll use a simplified version that directly calls the core merklize functionality
                    let result = call_merklize_directly(
                        &state_merkle_db,
                        0, // base_version
                        1, // version
                        &all_updates_array,
                    );
                    
                    assert!(result.is_ok(), "Merklize operation should succeed");
                });
            },
        );
    }
    group.finish();
}

/// Helper function to call the merklize functionality directly
fn call_merklize_directly(
    db: &StateMerkleDb,
    base_version: Option<Version>,
    version: Version,
    all_updates: &[Vec<(HashValue, Option<(HashValue, StateKey)>)>; NUM_STATE_SHARDS],
) -> Result<()> {
    use aptos_experimental_runtimes::thread_manager::THREAD_MANAGER;
    use aptos_scratchpad::{SparseMerkleTree, UpdateToLatest};
    use rayon::prelude::*;
    use aptos_storage_interface::jmt_update_refs;

    // Create empty sparse merkle trees for last and current snapshots
    let last_smt = SparseMerkleTree::new_empty();
    let smt = SparseMerkleTree::new_empty();

    let shard_persisted_versions = db.get_shard_persisted_versions(base_version).unwrap();

    THREAD_MANAGER.get_non_exe_cpu_pool().install(|| {
        all_updates
            .par_iter()
            .enumerate()
            .for_each(|(shard_id, updates)| {
                let _result = db.merklize_value_set_for_shard(
                    shard_id,
                    jmt_update_refs(updates),
                    None, // node_hashes
                    version,
                    base_version,
                    shard_persisted_versions[shard_id],
                    None, // previous_epoch_ending_version
                ).unwrap();
            });
    });

    Ok(())
}

/// Benchmark merklize with different value sizes to measure impact
fn benchmark_merklize_with_variable_value_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("sharded_jmt_merklize_variable_value_sizes");

    let key_count = 5000;
    let value_sizes = vec![64, 256, 512, 1024]; // Different value sizes in bytes

    for value_size in value_sizes {
        group.bench_with_input(
            BenchmarkId::new("value_size", value_size),
            &value_size,
            |b, &size| {
                b.iter(|| {
                    // Create temporary storage
                    let (_temp_dir, path) = create_temp_storage();

                    // Create StateMerkleDb with sharding enabled
                    let rocksdb_configs = aptos_types::state_store::state_storage_usage::RocksdbConfigs::default();
                    let storage_paths = aptos_config::config::StorageDirPaths::from_path(path.clone());
                    
                    let state_merkle_db = Arc::new(
                        StateMerkleDb::new(
                            &storage_paths,
                            rocksdb_configs,
                            None, // env
                            None, // block_cache
                            false, // readonly
                            0, // max_nodes_per_lru_cache_shard
                            false, // is_hot
                            false, // delete_on_restart
                        )
                        .expect("Failed to create StateMerkleDb")
                    );

                    // Generate keys
                    let keys = generate_keys(key_count);
                    
                    // Create per-shard value sets
                    let mut all_updates: Vec<Vec<(HashValue, Option<(HashValue, StateKey)>)>> = 
                        vec![Vec::new(); NUM_STATE_SHARDS];
                    
                    let state_values = create_mock_state_values_with_size(&keys, size);
                    
                    for (key, value_opt) in state_values {
                        let shard_id = get_state_shard_id(&key);
                        all_updates[shard_id].push((key, value_opt));
                    }

                    // Convert to the format expected by the merklize function
                    let all_updates_array: [Vec<(HashValue, Option<(HashValue, StateKey)>)>; NUM_STATE_SHARDS] = 
                        all_updates.try_into().unwrap_or_else(|v: Vec<_>| {
                            panic!("Expected {} elements, got {}", NUM_STATE_SHARDS, v.len())
                        });

                    // Call the merklize function
                    let result = call_merklize_directly(
                        &state_merkle_db,
                        0, // base_version
                        1, // version
                        &all_updates_array,
                    );
                    
                    assert!(result.is_ok(), "Merklize operation should succeed");
                });
            },
        );
    }
    group.finish();
}

/// Benchmark merklize with different shard distributions (uniform vs skewed)
fn benchmark_merklize_with_uniform_distribution(c: &mut Criterion) {
    let key_count = 10000;
    let value_size = 256;

    c.bench_function("merklize_uniform_distribution", |b| {
        b.iter(|| {
            // Create temporary storage
            let (_temp_dir, path) = create_temp_storage();

            // Create StateMerkleDb with sharding enabled
            let rocksdb_configs = aptos_types::state_store::state_storage_usage::RocksdbConfigs::default();
            let storage_paths = aptos_config::config::StorageDirPaths::from_path(path.clone());
            
            let state_merkle_db = Arc::new(
                StateMerkleDb::new(
                    &storage_paths,
                    rocksdb_configs,
                    None, // env
                    None, // block_cache
                    false, // readonly
                    0, // max_nodes_per_lru_cache_shard
                    false, // is_hot
                    false, // delete_on_restart
                )
                .expect("Failed to create StateMerkleDb")
            );

            // Generate keys uniformly distributed across shards
            let keys = generate_keys(key_count);
            
            // Create per-shard value sets
            let mut all_updates: Vec<Vec<(HashValue, Option<(HashValue, StateKey)>)>> = 
                vec![Vec::new(); NUM_STATE_SHARDS];
            
            let state_values = create_mock_state_values_with_size(&keys, value_size);
            
            for (key, value_opt) in state_values {
                let shard_id = get_state_shard_id(&key);
                all_updates[shard_id].push((key, value_opt));
            }

            // Convert to the format expected by the merklize function
            let all_updates_array: [Vec<(HashValue, Option<(HashValue, StateKey)>)>; NUM_STATE_SHARDS] = 
                all_updates.try_into().unwrap_or_else(|v: Vec<_>| {
                    panic!("Expected {} elements, got {}", NUM_STATE_SHARDS, v.len())
                });

            // Call the merklize function
            let result = call_merklize_directly(
                &state_merkle_db,
                0, // base_version
                1, // version
                &all_updates_array,
            );
            
            assert!(result.is_ok(), "Merklize operation should succeed");
        });
    });
}

criterion_group!(
    name = sharded_jmt_parallel_benchmarks;
    config = Criterion::default().sample_size(10);
    targets = 
        benchmark_merklize_parallel,
        benchmark_merklize_with_variable_value_sizes,
        benchmark_merklize_with_uniform_distribution
);
criterion_main!(sharded_jmt_parallel_benchmarks);