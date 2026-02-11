// Copyright (c) Aptos Foundation
// Licensed pursuant to the Innovation-Enabling Source Code License, available at https://github.com/aptos-labs/aptos-core/blob/main/LICENSE

use crate::{
    db_access::DbAccessUtil,
    db_generator::create_db_with_accounts,
    measurements::OverallMeasuring,
    pipeline::PipelineConfig,
    StorageTestConfig,
};
use aptos_config::config::NO_OP_STORAGE_PRUNER_CONFIG;
use aptos_crypto::hash::CryptoHash;
use aptos_db::AptosDB;
use aptos_executor::block_executor::BlockExecutor;
use aptos_jellyfish_merkle::metrics::{APTOS_JELLYFISH_INTERNAL_ENCODED_BYTES, APTOS_JELLYFISH_LEAF_ENCODED_BYTES};
use aptos_logger::info;
use aptos_storage_interface::{state_store::state_view::db_state_view::LatestDbStateCheckpointView, DbReaderWriter};
use aptos_types::{account_address::AccountAddress, on_chain_config::{FeatureFlag, Features}, state_store::state_key::StateKey};
use aptos_vm::aptos_vm::AptosVMBlockExecutor;
use std::{
    path::PathBuf,
    time::Instant,
};

/// Benchmark for sharding Jellyfish Merkle Tree with RocksDB
pub fn benchmark_jmt_sharding(
    num_accounts: usize,
    num_operations: usize,
    block_size: usize,
    test_folder: PathBuf,
    enable_storage_sharding: bool,
) {
    aptos_logger::Logger::new().init();
    
    info!("Starting JMT sharding benchmark with {} accounts, {} operations, block size {}, sharding enabled: {}", 
          num_accounts, num_operations, block_size, enable_storage_sharding);

    // Setup features
    let mut features = Features::default();
    features.disable(FeatureFlag::CALCULATE_TRANSACTION_FEE_FOR_DISTRIBUTION);
    features.enable(FeatureFlag::NEW_ACCOUNTS_DEFAULT_TO_FA_APT_STORE);
    features.enable(FeatureFlag::OPERATIONS_DEFAULT_TO_FA_APT_STORE);

    // Setup storage configuration
    let storage_test_config = StorageTestConfig {
        pruner_config: NO_OP_STORAGE_PRUNER_CONFIG,
        enable_storage_sharding,
        enable_indexer_grpc: false,
    };

    // Create initial database with accounts
    let storage_dir = test_folder.join("db");
    let checkpoint_dir = test_folder.join("cp");

    create_db_with_accounts::<AptosVMBlockExecutor>(
        num_accounts,
        100_000_000_000, // init_account_balance
        10000,           // block_size
        &storage_dir,
        storage_test_config,
        false, // verify_sequence_numbers
        PipelineConfig::default(),
        features,
        false, // is_keyless
    );

    // Initialize configuration for benchmarking
    let (mut config, _genesis_key) = aptos_genesis::test_utils::test_config_with_custom_features(Features::default());
    config.storage.dir = checkpoint_dir.clone();
    storage_test_config.init_storage_config(&mut config);

    let db = DbReaderWriter::new(
        AptosDB::open(
            config.storage.get_dir_paths(),
            false, /* readonly */
            config.storage.storage_pruner_config,
            config.storage.rocksdb_configs,
            false,
            config.storage.buffered_state_target_items,
            config.storage.max_num_nodes_per_lru_cache_shard,
            None,
            aptos_config::config::HotStateConfig::default(),
        )
        .expect("DB should open."),
    );

    // Start timing the actual benchmark
    let start_time = Instant::now();
    let measuring = OverallMeasuring::start();

    // Create executor and run operations
    let executor = BlockExecutor::<AptosVMBlockExecutor>::new(db.clone());
    let start_version = db.reader.expect_synced_version();

    // Run benchmark operations
    run_jmt_operations(&db, num_operations, block_size);

    // Record metrics
    let num_txns = db.reader.expect_synced_version() - start_version;
    let overall_results = measuring.elapsed(
        format!("JMT Sharding Benchmark (sharding: {})", enable_storage_sharding),
        "".to_string(),
        num_txns,
    );

    let elapsed = start_time.elapsed();
    info!("JMT Sharding Benchmark Results:");
    info!("  Elapsed time: {:.2?}", elapsed);
    info!("  Operations: {}", num_operations);
    info!("  Accounts: {}", num_accounts);
    info!("  Block size: {}", block_size);
    info!("  Sharding enabled: {}", enable_storage_sharding);
    info!("  Throughput: {:.2} ops/sec", num_operations as f64 / elapsed.as_secs_f64());
    info!("  Total internal nodes encoded bytes: {}", APTOS_JELLYFISH_INTERNAL_ENCODED_BYTES.get());
    info!("  Total leaf nodes encoded bytes: {}", APTOS_JELLYFISH_LEAF_ENCODED_BYTES.get());
    info!("  Total transactions processed: {}", num_txns);

    overall_results.print_end();
}

fn run_jmt_operations(db: &DbReaderWriter, num_operations: usize, block_size: usize) {
    // Get initial state
    let state_view = db.reader.latest_state_checkpoint_view().unwrap();
    let total_supply = DbAccessUtil::get_total_supply(&state_view).unwrap();
    
    info!("Initial total supply: {:?}", total_supply);

    // Simulate operations on the JMT
    for i in 0..num_operations {
        if i % block_size == 0 {
            info!("Processing operation {}/{}", i, num_operations);
        }

        // Get current version
        let version = db.reader.expect_synced_version();
        
        // Create a dummy account address for testing
        let dummy_address = AccountAddress::from([i as u8; AccountAddress::LENGTH]);
        let dummy_key = StateKey::raw(&dummy_address.to_vec());
        
        // Perform a JMT operation - get with proof
        if let Ok(result) = db.reader.get_state_value_with_proof_by_version(&dummy_key, version) {
            // Use the result to prevent optimization
            let _proof = result.1;
        }

        // Perform a JMT operation - get root hash via state store
        if let Ok(_root_hash) = db.reader.get_state_value_with_proof_by_version(&dummy_key, version) {
            // Use the result to prevent optimization
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aptos_temppath::TempPath;

    #[test]
    fn test_jmt_sharding_benchmark() {
        let temp_dir = TempPath::new();
        temp_dir.create_as_dir().unwrap();
        
        // Test with sharding enabled
        benchmark_jmt_sharding(
            1000,  // num_accounts
            10000, // num_operations
            100,   // block_size
            temp_dir.path().to_path_buf(),
            true,  // enable_storage_sharding
        );
        
        // Test with sharding disabled for comparison
        let temp_dir2 = TempPath::new();
        temp_dir2.create_as_dir().unwrap();
        benchmark_jmt_sharding(
            1000,  // num_accounts
            10000, // num_operations
            100,   // block_size
            temp_dir2.path().to_path_buf(),
            false, // enable_storage_sharding
        );
    }
}