# Benchmarks

Reproducible micro-benchmarks for the HNSW vector index. Numbers below are from
a developer laptop (single-threaded, `--release`); treat them as relative, not
absolute — run the benchmark on your own hardware.

## Running

```bash
cargo run --release -p kowitodb-index --example bench_hnsw
```

Override the workload via environment variables:

| Var | Default | Meaning |
| --- | --- | --- |
| `BENCH_N` | `10000` | number of indexed vectors |
| `BENCH_DIM` | `384` | vector dimension (matches `all-MiniLM-L6-v2`) |
| `BENCH_Q` | `200` | number of query vectors |
| `BENCH_K` | `10` | neighbors per query |
| `BENCH_EF` | `HnswParams::default().ef_search` (200) | search beam width |

The benchmark builds the index from synthetic unit-norm random vectors (fixed
seed), measures build throughput and per-query latency percentiles, and computes
**recall@k against brute-force cosine ground truth**.

## Results (10k × 384-dim, single thread)

### Recall vs. `ef_search` (the recall/latency knob)

| `ef_search` | recall@10 | mean latency |
| --- | --- | --- |
| 50 | ~60% | ~0.7 ms |
| 100 | ~80% | ~1.2 ms |
| **200 (default)** | **~94%** | **~1.9 ms** |
| 400 | ~99% | ~3.1 ms |

The default `ef_search` was raised from 50 to **200** so the index ships at a
competitive ~94% recall@10 rather than ~60%. Tune it per workload via
`HnswParams { ef_search, .. }`.

### Build throughput

~900 inserts/sec for 10k × 384-dim (single-threaded, `ef_construction=200`).
Build is the current bottleneck and a known area to improve (parallel insert,
fewer allocations, on-disk persistence so the index need not be rebuilt on
restart).

## Caveats / what these numbers are not

- Synthetic uniform-random vectors are a **harder** recall case than real
  clustered embeddings; expect equal-or-better recall on real data at the same
  `ef_search`.
- Single-threaded query latency; the server handles requests concurrently.
- These measure the ANN index only — not end-to-end `ask()` latency (which also
  embeds the query, runs the planner, and assembles context).
