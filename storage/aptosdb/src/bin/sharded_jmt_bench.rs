// Benchmark: end-to-end sharded JMT write (merklize + state_kv + merkle commit)
// Default: N = 100_000 keys

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::RngCore;
use rand::SeedableRng;
use std::sync::atomic::{AtomicU64, Ordering};
use aptos_experimental_runtimes::thread_manager::THREAD_MANAGER;
use rayon::prelude::*;
use aptos_db::db::AptosDB;
use aptos_db::ledger_db::LedgerDb;
use aptos_db::state_merkle_db::StateMerkleDb;
use aptos_db::state_kv_db::StateKvDb;
use aptos_db::schema::state_value_by_key_hash::StateValueByKeyHashSchema;
use tempfile;

fn bench_sharded_jmt_end2end(c: &mut Criterion) {
    let default_n: usize = 100_000;
    let value_size: usize = 256;

    // Single-case benchmark. If you want multiple sizes, add them to the vec.
    let mut group = c.benchmark_group("sharded_jmt_end2end");
    group.sample_size(10);

    // Setup: create DBs and prepare test data.
    let tmpdir = tempfile::tempdir().expect("tempdir");
    let db_path = tmpdir.path().to_path_buf();

    // Build storage paths and rocksdb configs
    let mut storage_paths = aptos_config::config::StorageDirPaths::from_path(&db_path);
    let mut rocksdb_configs = aptos_config::config::RocksdbConfigs::default();
    rocksdb_configs.enable_storage_sharding = true;

    // Open DBs (ledger_db, optional hot, state_merkle_db, state_kv_db)
    let (_ledger_db, _hot_state_merkle_db, state_merkle_db, state_kv_db): (
        LedgerDb,
        Option<StateMerkleDb>,
        StateMerkleDb,
        StateKvDb,
    ) =
        AptosDB::open_dbs(
            &storage_paths,
            rocksdb_configs,
            None,
            None,
            false,
            0,
            false,
        )
        .expect("open_dbs");

    // Prepare deterministic keys and values. We'll reuse the same keys across benchmark iterations
    // and increment the version on each iteration.
    use aptos_crypto::hash::{CryptoHash, HashValue};
    use aptos_types::state_store::state_key::StateKey;
    use aptos_types::state_store::state_value::StateValue;
    use aptos_storage_interface::jmt_update_refs;

    let mut rng = rand::rngs::StdRng::seed_from_u64(0xBEEF);

    // Generate N keys and split them into shards
    let mut per_shard: Vec<Vec<(HashValue, Option<(HashValue, StateKey)>)>> =
        vec![Vec::new(); aptos_types::state_store::NUM_STATE_SHARDS];
    let mut values_by_shard: Vec<Vec<Vec<u8>>> = vec![Vec::new(); per_shard.len()];

    for i in 0..default_n {
        let mut id_bytes = [0u8; 8];
        id_bytes[..8].copy_from_slice(&((i as u64).to_le_bytes()));
        // Use a raw state key (test-only style)
        let sk = StateKey::raw(&id_bytes);
        let key_hash = CryptoHash::hash(&sk);

        // Generate deterministic value bytes
        let mut v = vec![0u8; value_size];
        rng.fill_bytes(&mut v);
        let value_hash = HashValue::sha3_256_of(&v);

        let shard = sk.get_shard_id();
        per_shard[shard].push((key_hash, Some((value_hash, sk.clone()))));
        values_by_shard[shard].push(v);
    }

    // Atomic version counter for iterations
    let version_counter = AtomicU64::new(1);

    group.bench_with_input(
        BenchmarkId::from_parameter(default_n),
        &default_n,
        |b, &_size| {
            b.iter(|| {
                let version = version_counter.fetch_add(1, Ordering::Relaxed);

                // 1) Merklize per shard -> produce shard batches
                let mut shard_batches = Vec::with_capacity(aptos_types::state_store::NUM_STATE_SHARDS);
                let mut shard_roots = Vec::with_capacity(aptos_types::state_store::NUM_STATE_SHARDS);
                for shard_id in 0..aptos_types::state_store::NUM_STATE_SHARDS {
                    let updates = &per_shard[shard_id];
                    // convert to refs
                    let refs = jmt_update_refs(updates);
                    let (root_node, raw_batch) = state_merkle_db
                        .merklize_value_set_for_shard(
                            shard_id,
                            refs,
                            None,
                            version,
                            None,
                            None,
                            None,
                        )
                        .expect("merklize shard");
                    shard_roots.push(root_node);
                    shard_batches.push(raw_batch);
                }

                // 2) Calculate top levels
                let (root_hash, _leaf_count, top_levels_batch) = state_merkle_db
                    .calculate_top_levels(shard_roots, version, None, None)
                    .expect("calculate_top_levels");

                // 3) Prepare and commit state_kv (per-shard native batches)
                let mut sharded_kv_batches = state_kv_db.new_sharded_native_batches();
                for shard_id in 0..per_shard.len() {
                    let updates = &per_shard[shard_id];
                    let values = &values_by_shard[shard_id];
                    let mut native_batch = &mut sharded_kv_batches[shard_id];
                    for (idx, (key_hash, _opt)) in updates.iter().enumerate() {
                        let vbytes = &values[idx];
                        let sv = StateValue::from(vbytes.clone());
                        native_batch
                            .put::<StateValueByKeyHashSchema>(
                                &(*key_hash, version),
                                &Some(sv),
                            )
                            .expect("put state value");
                    }
                }

                state_kv_db
                    .commit(version, None, sharded_kv_batches)
                    .expect("state_kv commit");

                // 4) Commit merkle raw batches
                // convert Vec<RawBatch> to Vec<impl IntoRawBatch>
                state_merkle_db
                    .commit(
                        version,
                        top_levels_batch,
                        shard_batches,
                    )
                    .expect("state merkle commit");
            })
        },
    );

    group.finish();
}

fn bench_merklize_parallel(c: &mut Criterion) {
    let default_n: usize = 100_000;
    let value_size: usize = 256;

    let mut group = c.benchmark_group("sharded_jmt_merklize_parallel");
    group.sample_size(10);

    let tmpdir = tempfile::tempdir().expect("tempdir");
    let db_path = tmpdir.path().to_path_buf();
    let mut storage_paths = aptos_config::config::StorageDirPaths::from_path(&db_path);
    let mut rocksdb_configs = aptos_config::config::RocksdbConfigs::default();
    rocksdb_configs.enable_storage_sharding = true;

    let (_ledger_db, _hot_state_merkle_db, state_merkle_db, _state_kv_db): (
        LedgerDb,
        Option<StateMerkleDb>,
        StateMerkleDb,
        StateKvDb,
    ) =
        AptosDB::open_dbs(
            &storage_paths,
            rocksdb_configs,
            None,
            None,
            false,
            0,
            false,
        )
    .expect("open_dbs");

    use aptos_crypto::hash::HashValue;
    use aptos_types::state_store::state_key::StateKey;
    use aptos_storage_interface::jmt_update_refs;

    let mut rng = rand::rngs::StdRng::seed_from_u64(0xBEEF);

    // prepare per-shard updates
    let mut per_shard: Vec<Vec<(HashValue, Option<(HashValue, StateKey)>)>> =
        vec![Vec::new(); aptos_types::state_store::NUM_STATE_SHARDS];
    for i in 0..default_n {
        let mut id_bytes = [0u8; 8];
        id_bytes[..8].copy_from_slice(&((i as u64).to_le_bytes()));
        let sk = StateKey::raw(&id_bytes);
        let key_hash = aptos_crypto::hash::CryptoHash::hash(&sk);

        let mut v = vec![0u8; value_size];
        rng.fill_bytes(&mut v);
        let value_hash = HashValue::sha3_256_of(&v);

        let shard = sk.get_shard_id();
        per_shard[shard].push((key_hash, Some((value_hash, sk.clone()))));
    }

    let version = 1u64;

    group.bench_function("merklize_parallel_N_100k", |b| {
        b.iter(|| {
            // Run parallel merklize similar to StateSnapshotCommitter::merklize
            let (shard_root_nodes, batches_for_shards): (Vec<_>, Vec<_>) = THREAD_MANAGER
                .get_non_exe_cpu_pool()
                .install(|| {
                    per_shard
                        .par_iter()
                        .enumerate()
                        .map(|(shard_id, updates)| {
                            let refs = jmt_update_refs(updates);
                            state_merkle_db
                                .merklize_value_set_for_shard(
                                    shard_id,
                                    refs,
                                    None,
                                    version,
                                    None,
                                    None,
                                    None,
                                )
                                .expect("merklize shard")
                        })
                        .collect::<Vec<_>>()
                        .into_iter()
                        .unzip()
                });

            // calculate top levels (single-threaded path)
            let (_root_hash, _leaf_count, _top_batch) = state_merkle_db
                .calculate_top_levels(shard_root_nodes, version, None, None)
                .expect("calculate_top_levels");
        })
    });

    group.finish();
}

criterion_group!(benches, bench_sharded_jmt_end2end, bench_merklize_parallel);
criterion_main!(benches);

