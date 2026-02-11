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
use aptos_db::AptosDB;
use aptos_jellyfish_merkle::metrics::{APTOS_JELLYFISH_INTERNAL_ENCODED_BYTES, APTOS_JELLYFISH_LEAF_ENCODED_BYTES};
use aptos_logger::info;
use aptos_storage_interface::{state_store::state_view::db_state_view::LatestDbStateCheckpointView, DbReaderWriter};
use aptos_types::{account_address::AccountAddress, on_chain_config::{FeatureFlag, Features}, state_store::state_key::StateKey};
use aptos_vm::aptos_vm::AptosVMBlockExecutor;
use serde::{Deserialize, Serialize};
use std::{
    path::PathBuf,
    time::Instant,
};

// Structure to hold benchmark results
#[derive(Serialize, Deserialize, Debug)]
pub struct JmtBenchmarkResult {
    pub num_accounts: usize,
    pub num_operations: usize,
    pub block_size: usize,
    pub storage_sharding_enabled: bool,
    pub duration_ms: u64,
    pub operations_per_second: f64,
    pub avg_latency_per_operation: f64,
    pub total_internal_encoded_bytes: u64,
    pub total_leaf_encoded_bytes: u64,
    pub total_transactions_processed: u64,
}

/// Benchmark for sharding Jellyfish Merkle Tree with RocksDB
pub fn benchmark_jmt_sharding(
    num_accounts: usize,
    num_operations: usize,
    block_size: usize,
    test_folder: PathBuf,
    enable_storage_sharding: bool,
) -> JmtBenchmarkResult {
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

    // Run benchmark operations
    let start_version = db.reader.expect_synced_version();

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

    // Create and return benchmark results
    JmtBenchmarkResult {
        num_accounts,
        num_operations,
        block_size,
        storage_sharding_enabled: enable_storage_sharding,
        duration_ms: elapsed.as_millis() as u64,
        operations_per_second: num_operations as f64 / elapsed.as_secs_f64(),
        avg_latency_per_operation: elapsed.as_nanos() as f64 / num_operations as f64,
        total_internal_encoded_bytes: APTOS_JELLYFISH_INTERNAL_ENCODED_BYTES.get(),
        total_leaf_encoded_bytes: APTOS_JELLYFISH_LEAF_ENCODED_BYTES.get(),
        total_transactions_processed: num_txns,
    }
}

/// Function to generate a comparative report of benchmarks with and without sharding
pub fn generate_comparative_report(
    result_with_sharding: JmtBenchmarkResult,
    result_without_sharding: JmtBenchmarkResult,
) {
    println!("{}", "\n".to_owned() + &"=".repeat(60));
    println!("JMT SHARDING BENCHMARK COMPARISON REPORT");
    println!("{}", "=".to_owned() + &"=".repeat(58));
    
    // Performance metrics table
    println!("\nPerformance Comparison:");
    println!("{:<30} {:<20} {:<20}", "Metric", "With Sharding", "Without Sharding");
    println!("{:-<30} {:-<20} {:-<20}", "", "", "");
    println!(
        "{:<30} {:<20.2} {:<20.2}",
        "Operations/sec", 
        result_with_sharding.operations_per_second,
        result_without_sharding.operations_per_second
    );
    println!(
        "{:<30} {:<20.2}ms {:<20.2}ms",
        "Total Duration", 
        result_with_sharding.duration_ms as f64,
        result_without_sharding.duration_ms as f64
    );
    println!(
        "{:<30} {:<20.2}ns {:<20.2}ns",
        "Avg Latency/op", 
        result_with_sharding.avg_latency_per_operation,
        result_without_sharding.avg_latency_per_operation
    );
    
    // Calculate improvement ratios
    let ops_improvement = (result_with_sharding.operations_per_second / result_without_sharding.operations_per_second) * 100.0 - 100.0;
    let latency_improvement = (result_without_sharding.avg_latency_per_operation / result_with_sharding.avg_latency_per_operation) * 100.0 - 100.0;
    
    println!("\nImprovement Analysis:");
    if ops_improvement > 0.0 {
        println!("  Throughput increased by {:.2}% with sharding", ops_improvement);
    } else {
        println!("  Throughput decreased by {:.2}% with sharding", ops_improvement.abs());
    }
    
    if latency_improvement > 0.0 {
        println!("  Operation latency improved by {:.2}% with sharding", latency_improvement);
    } else {
        println!("  Operation latency increased by {:.2}% with sharding", latency_improvement.abs());
    }
    
    // Storage metrics
    println!("\nStorage Metrics:");
    println!("{:<35} {:<20} {:<20}", "Metric", "With Sharding", "Without Sharding");
    println!("{:-<35} {:-<20} {:-<20}", "", "", "");
    println!(
        "{:<35} {:<20} {:<20}",
        "Internal Node Encoded Bytes", 
        result_with_sharding.total_internal_encoded_bytes,
        result_without_sharding.total_internal_encoded_bytes
    );
    println!(
        "{:<35} {:<20} {:<20}",
        "Leaf Node Encoded Bytes", 
        result_with_sharding.total_leaf_encoded_bytes,
        result_without_sharding.total_leaf_encoded_bytes
    );
    println!(
        "{:<35} {:<20} {:<20}",
        "Transactions Processed", 
        result_with_sharding.total_transactions_processed,
        result_without_sharding.total_transactions_processed
    );
    
    // Export results to JSON for visualization
    export_results_to_json(&result_with_sharding, &result_without_sharding);
    
    println!("\nReport generated at: {}", std::env::current_dir().unwrap().join("jmt_benchmark_report.txt").display());
}

/// Export benchmark results to JSON format for visualization tools
fn export_results_to_json(
    result_with_sharding: &JmtBenchmarkResult,
    result_without_sharding: &JmtBenchmarkResult,
) {
    // Create a comparative analysis
    let comparative_analysis = serde_json::json!({
        "improvement_ratio_ops_sec": result_with_sharding.operations_per_second / result_without_sharding.operations_per_second,
        "improvement_ratio_avg_latency": result_without_sharding.avg_latency_per_operation / result_with_sharding.avg_latency_per_operation,
        "throughput_improvement_percentage": (result_with_sharding.operations_per_second / result_without_sharding.operations_per_second) * 100.0 - 100.0,
        "latency_improvement_percentage": (result_without_sharding.avg_latency_per_operation / result_with_sharding.avg_latency_per_operation) * 100.0 - 100.0,
    });
    
    // Create a single JSON object with all data
    let export_data = serde_json::json!({
        "with_sharding": result_with_sharding,
        "without_sharding": result_without_sharding,
        "comparative_analysis": comparative_analysis,
    });
    
    // Write JSON results to file
    if let Ok(json_data) = serde_json::to_string_pretty(&export_data) {
        let _ = std::fs::write("jmt_benchmark_results.json", json_data);
    }
    
    // Write human-readable report
    let report_content = format!(
        "JMT Sharding Benchmark Report\n\n\
        With Sharding:\n\
        - Operations/sec: {:.2}\n\
        - Duration: {}ms\n\
        - Avg Latency/op: {:.2}ns\n\
        - Internal Encoded Bytes: {}\n\
        - Leaf Encoded Bytes: {}\n\n\
        Without Sharding:\n\
        - Operations/sec: {:.2}\n\
        - Duration: {}ms\n\
        - Avg Latency/op: {:.2}ns\n\
        - Internal Encoded Bytes: {}\n\
        - Leaf Encoded Bytes: {}\n\n\
        Comparative Analysis:\n\
        - Throughput Improvement: {:.2}%\n\
        - Latency Improvement: {:.2}%\n",
        result_with_sharding.operations_per_second,
        result_with_sharding.duration_ms,
        result_with_sharding.avg_latency_per_operation,
        result_with_sharding.total_internal_encoded_bytes,
        result_with_sharding.total_leaf_encoded_bytes,
        result_without_sharding.operations_per_second,
        result_without_sharding.duration_ms,
        result_without_sharding.avg_latency_per_operation,
        result_without_sharding.total_internal_encoded_bytes,
        result_without_sharding.total_leaf_encoded_bytes,
        (result_with_sharding.operations_per_second / result_without_sharding.operations_per_second) * 100.0 - 100.0,
        (result_without_sharding.avg_latency_per_operation / result_with_sharding.avg_latency_per_operation) * 100.0 - 100.0
    );
    
    let _ = std::fs::write("jmt_benchmark_report.txt", report_content);
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