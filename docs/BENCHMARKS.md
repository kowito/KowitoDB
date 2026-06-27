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
| `BENCH_THREADS` | logical cores | concurrent query threads for the throughput test |
| `BENCH_SHARDS` | `min(cores, 16)` | shards for the parallel-build test |

The benchmark builds the index from synthetic unit-norm random vectors (fixed
seed), measures build throughput and per-query latency percentiles, and computes
**recall@k against brute-force cosine ground truth**.

## Results (10k × 384-dim)

Numbers below are after the hot-path optimizations described under
[Optimizations](#optimizations).

### Recall vs. `ef_search` (the recall/latency knob)

| `ef_search` | recall@10 | mean latency |
| --- | --- | --- |
| 50 | ~60% | ~0.5 ms |
| 100 | ~80% | ~0.9 ms |
| **200 (default)** | **~94%** | **~1.5 ms** |
| 400 | ~99% | ~2.4 ms |

The default `ef_search` was raised from 50 to **200** so the index ships at a
competitive ~94% recall@10 rather than ~60%. Tune it per workload via
`HnswParams { ef_search, .. }`.

### Single-thread query throughput

~660 queries/sec at the default `ef_search=200` (~1.5 ms mean), up from
~530 qps before the hot-path optimizations.

### Concurrent throughput (scales with cores)

Queries hold only a read lock, so single-node QPS scales with cores. On a
10-core machine: **~3,700 aggregate queries/sec (~5.7× single-thread)**. Set
`BENCH_THREADS` to vary the thread count.

### Build throughput

Single-index (one global write lock): ~1,150 inserts/sec for 10k × 384-dim
(single-threaded, `ef_construction=200`), up from ~900.

**Sharded parallel build** (`ShardedHnswIndex`, one thread per shard) is far
faster — each shard builds a smaller graph in parallel. On a 10-core machine
with 10 shards: **~0.4 s for 10k vectors (~25,000 inserts/s, ~21× the serial
build)**. The speedup is super-linear because HNSW build cost grows
super-linearly with graph size, so ten 1k-vector graphs build much faster than
one 10k-vector graph — on top of the parallelism. Set `BENCH_SHARDS` to vary
the shard count. The engine uses the sharded index by default (shards =
`min(cores, 16)`), so reindex-on-restart is parallel.

Sharded recall is **equal or better** than the single index (each shard is a
smaller, more accurate graph; the per-shard top-k are merged): ~100% recall@10
at 10 shards on this workload vs ~94% single-index.

### Quantization (memory vs. recall)

int8 scalar quantization (`HnswParams { quantize: true }`, or
`KOWITODB_VECTOR_QUANTIZE=1` on the server) stores vectors at **~4× less
memory** — the key lever for fitting more vectors in RAM. On this workload it
costs a few points of recall: **~91% recall@10** (vs ~100% sharded full
precision). It assumes ~unit-norm embeddings (the models KowitoDB uses produce
these). Off by default.

**RaBitQ-style 1-bit binary quantization** (`HnswParams { binary_quantize: true }`,
or `KOWITODB_VECTOR_BINARY_QUANTIZE=1`) goes further — **~32× less memory** than
f32 (one sign bit per dimension plus two scalars). Each vector is rotated by a
structured random rotation (random ±1 signs + a normalized fast Walsh–Hadamard
transform, O(d log d)) so the sign codes are a good estimator; the graph is then
navigated by a **popcount Hamming fast path** and the final top-k is re-scored
with the unbiased asymmetric RaBitQ estimator.

**Matryoshka adaptive retrieval** (`HnswParams { coarse_dim: Some(d) }`, or
`KOWITODB_VECTOR_COARSE_DIM=d`) keeps full f32 vectors but navigates the graph
on only the first `d` dimensions, then refines the top-k at full dimension — a
query-*speed* lever for MRL-trained embeddings (whose prefixes are valid).

Measured on a **clustered** dataset (n=10k, dim=384, ef_search=200 — clustered
because random unit vectors are a worst case for compression: in high dimensions
all pairwise distances concentrate, leaving 1-bit codes and dimension-prefixes
little to separate):

| Mode | recall@10 | latency vs full | memory |
|------|-----------|-----------------|--------|
| Full f32 | ~99% | 1.0× | 1× |
| int8 (random set) | ~91% | ~1.0× | ¼× |
| **Binary 1-bit** | **~43%** | **~0.28× (≈3.6× faster)** | **1/32×** |
| **Binary + int8 rerank** | **~79%** | **~0.27× (≈3.7× faster)** | **¼×** |
| **Matryoshka (96/384)** | **~42%** | **~0.48× (≈2.1× faster)** | 1× |

The speedups are real and consistent. Plain 1-bit binary's low recall is the
honest cost of aggressive approximation; the **oversample→rescore** pattern
(`binary_rerank` / `KOWITODB_VECTOR_BINARY_RERANK=1`) recovers most of it —
retaining an int8 copy and re-scoring the oversampled top-k **roughly doubles
recall (~43% → ~79%) at the same latency**, trading the 32× memory win down to
int8's ~4×. It stays faster than plain int8 because navigation still uses the
popcount Hamming fast path; only the final top-k touches int8. For MRL
embeddings, real prefixes preserve neighborhoods far better than this generic
synthetic, so Matryoshka recall in practice is higher. All modes are off by
default; binary takes precedence over int8 when both are set. Reproduce with
`cargo run --release -p kowitodb-index --example bench_hnsw`.

## Optimizations

The hot path (graph traversal + distance) was tuned without changing recall:

- **Squared distance** — HNSW only compares distances, so the `sqrt` was removed
  from the inner loop and applied only to the k returned results.
- **Fast hashing (ahash)** — UUID keys are hashed with ahash instead of the
  default SipHasher across the node map and visited/neighbor sets.
- **Alloc-free traversal** — neighbor expansion iterates the set in place instead
  of collecting a `Vec` per visited node.
- **Per-thread beam scratch** — the visited set and the two heaps are reused from
  a `thread_local` across queries, so a warm query path allocates only its
  returned vectors (allocator contention otherwise caps concurrent QPS).
- **Scalar greedy descent** — `search_layer_greedy` returns the single nearest id
  instead of a `Vec` per layer; the rotation lock is skipped entirely unless
  binary quantization is on.
- **8-accumulator distance** — `squared_dist` (and the int8 kernel) sum over 8
  independent accumulators instead of one. A single `.sum()` is latency-bound on
  the FP pipeline (each add waits on the previous); 8 lanes break that chain so
  the CPU pipelines them and the loop auto-vectorizes cleanly. ~**+18%
  single-thread QPS** on its own (f32 query latency ~1.79ms → ~1.54ms),
  recall-neutral.

Net effect at `ef_search=200`: ~35% lower query latency than the original
brute-force-replacement, ~27% higher build throughput, recall unchanged (~94%).

## Maximizing QPS

Concurrent throughput is **core-bound** once the hot path is allocation-free —
on a 4 P-core + 6 E-core box, 10 query threads already reach ~6.3× the
single-thread rate (near the hardware ceiling). So beyond the recall-neutral
tuning above, **QPS is bought by doing less work per query**, which trades
recall — choose the operating point for your workload:

| Lever | QPS effect | recall effect |
|-------|-----------|---------------|
| `ef_search` 200 → 100 | **~1.6× QPS** | ~98% → ~95% |
| `ef_search` 200 → 50 | **~2.6× QPS** | ~98% → ~93% |
| int8 quantization | ~1× (memory win) | ~91% |
| binary + int8 rerank | **~3.7× lower latency** | ~79% |
| binary 1-bit | **~3.6× lower latency** | data-dependent |

These compose: e.g. `ef_search=100` **and** `binary_rerank` stack a ~1.6× and a
~3–4× factor. For a fixed recall target, pick the lowest `ef_search` that hits it
(measure with `bench_hnsw`), then add quantization for the memory/bandwidth win.
The sharded index also scales build and adds per-query shard parallelism.

## Caveats / what these numbers are not

- Synthetic uniform-random vectors are a **harder** recall case than real
  clustered embeddings; expect equal-or-better recall on real data at the same
  `ef_search`.
- Single-threaded query latency; the server handles requests concurrently.
- These measure the ANN index only — not end-to-end `ask()` latency (which also
  embeds the query, runs the planner, and assembles context).
