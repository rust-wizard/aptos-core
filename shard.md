Located sharded JMT+DB integration in aptosdb and the Jellyfish crate in jellyfish-merkle. Key files:
State merkle DB struct and sharding logic: state_merkle_db.rs:51
Per-shard put API: batch_put_value_set_for_shard: state_merkle_db.rs:298
Creation of commits for shards: create_jmt_commit_batch_for_shard: state_merkle_db.rs:357
Final write into RocksDB shards: TreeWriter::write_node_batch / commit path: state_merkle_db.rs:917
High-level merklize flow that parallelizes per-shard work: state_snapshot_committer.rs:222
Jellyfish tree implementation & public APIs: lib.rs:86
How sharded JMT writes flow (step-by-step)

Client (e.g., StateSnapshotCommitter) builds per-version value updates across all shards and calls merklize. See merklize in state_snapshot_committer.
The updates are split by shard (based on state key hash / shard id).
For each shard, the JMT logic computes tree updates via JellyfishMerkleTree::batch_put_value_set_for_shard (exposed via StateMerkleDb::batch_put_value_set_for_shard). This yields a TreeUpdateBatch (new nodes + stale indices).
StateMerkleDb::create_jmt_commit_batch_for_shard converts the TreeUpdateBatch into RocksDB RawBatch targeting either the metadata DB (top levels) or the appropriate shard DB.
TreeWriter::write_node_batch (implemented in StateMerkleDb) splits nodes into top-level batch vs per-shard batches and commits them via commit_no_progress. This writes into the multiple RocksDB instances (shards) and metadata DB.
Top-level nodes (non-sharded) are stored in the metadata DB; sharded nodes go into state_merkle_db_shards[shard_id]. Parallelism: commit and shard merklization are parallelized (rayon).
Recommended benchmark entrypoints (what to measure)

Compute-only JMT update time (CPU work): call JellyfishMerkleTree::batch_put_value_set_for_shard (via StateMerkleDb::batch_put_value_set_for_shard) without committing to RocksDB — measures pure JMT CPU cost and in-memory node creation.
Full end-to-end (compute + DB commit): call the full path that produces RawBatch and commit it (via TreeWriter::write_node_batch / commit_no_progress) — measures disk IO and RocksDB overhead across shards. This is the primary target for "write operation of sharded JMT with aptos databases."
Higher-level flow: run StateSnapshotCommitter::merklize to measure the multi-shard parallel pipeline as used in production.
Where to place benchmark code and framework

Use Criterion for stable, reproducible benchmarking. Create a bench crate or benches under storage/aptosdb/benches (or a new small crate under devtools if you prefer isolation). Add criterion = "..." in Cargo.toml. Run with cargo bench (or cargo test --bench depending on harness).
Bench targets:
bench_jmt_compute_only — measures per-shard batch_put_value_set_for_shard.
bench_jmt_commit_end2end — measures compute + commit to RocksDB shards (temp dirs).
bench_merklize_parallel — measures StateSnapshotCommitter::merklize for multi-shard parallel pipeline.
Test-data generation (how to control volume & distribution)

Parameters to expose: total keys N, value size V bytes, batch size B (per-shard batching), shard distribution (uniform or skewed), number of iterations/versions.
Key generation approach:
Generate deterministic HashValue keys (e.g., hash of sequential bytes or HashValue::from_sha3_256 of counter) so experiments are repeatable.
Map each key to a shard using the same get_state_shard_id logic used in code (or call the helper used in tests) to build per-shard value_set. Ensures you can produce controlled uniform or skew distributions.
For full DB writes you need StateKey/StateValue to match storage schemas: produce StateKey instances and StateValue blobs of size V. The JMT write API expects pairs of (HashValue, Option<(HashValue, StateKey)>) (value hash and optional value+key). Use Some((value_hash, state_key)) for writes.
Volume guidance: start small to iterate (1k–10k keys), then scale to 100k–1M depending on machine/I/O. For disk-bound tests, choose sizes that fit your target throughput (e.g., 1M keys × 256B values ≈ 256MB of value payload).
Benchmark harness design details

Test setup:
Create a temporary storage dir via tempfile::tempdir().
Build RocksdbConfigs with enable_storage_sharding = true and pass to StateMerkleDb::new (use StorageDirPaths pointing into tempdir).
Optionally disable other DB components to focus on state_merkle_db (or instantiate StateMerkleDb directly).
Measurement:
For compute-only: call batch_put_value_set_for_shard inside Criterion::iter_batched (prepare data outside timed region, measure only CPU work).
For end-to-end: time the full commit path; ensure you sync or use RocksDB options that reflect real persistence (FSync). Use Criterion::measurement_time and sample sizes appropriate for noisy IO.
Repeat with different B, V, thread pool sizes to profile scaling.
Metrics to capture: latency per batch, throughput (keys/sec, bytes/sec), CPU time, RocksDB write amplification (if measurable), per-shard skew/variance.
Concurrency and knobs

Threading: the JMT path uses Rayon (parallel per-shard). Control Rayon via RAYON_NUM_THREADS or set the global pool in test harness to measure single-threaded vs multi-threaded.
Number of shards: NUM_STATE_SHARDS is 16 by default; your harness can vary effective shards by changing key-to-shard distribution or toggling sharding config (but DB layout expects fixed NUM_STATE_SHARDS).