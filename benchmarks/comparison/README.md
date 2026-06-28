# KowitoDB vs Qdrant vs Milvus — ANN comparison

A reproducible, apples-to-apples ANN benchmark. All three systems index the
**same** vectors, answer the **same** queries, and are scored against the
**same** brute-force ground truth, at **matched** HNSW parameters
(`M`, `ef_construction`, `ef_search`).

## How it works

1. The Rust side generates a deterministic dataset, benchmarks the KowitoDB
   HNSW index, and writes the dataset to a shared binary file:
   ```
   cargo run --release -p kowitodb-index --example bench_compare -- dataset.bin
   ```
   Env knobs: `CMP_N`, `CMP_DIM`, `CMP_Q`, `CMP_K`, `CMP_M`, `CMP_EFC`,
   `CMP_EFS` (comma-separated `ef_search` list).

2. Bring up the competitors and run them on the **same** `dataset.bin`:
   ```
   docker compose up -d qdrant            # lightweight (one container)
   docker compose up -d etcd minio milvus # Milvus standalone stack
   pip install requests numpy pymilvus
   python bench_qdrant.py dataset.bin --m 16 --efc 128 --efs 32,64,128,256
   python bench_milvus.py dataset.bin --m 16 --efc 128 --efs 32,64,128,256
   ```

Each prints CSV rows `system,ef_search,recall@k,qps_1thread,p50_us,p95_us`.

## Results (measured)

Dataset: **50 000 × 128** random unit vectors, 1 000 queries, k=10, cosine,
HNSW `M=16`, `ef_construction=128`. Single machine (Apple Silicon, 4P+6E).

| ef_search | KowitoDB recall | Qdrant recall | Milvus recall |
|-----------|----------------:|--------------:|--------------:|
| 32        | 0.429           | **0.594**     | 0.200         |
| 64        | 0.602           | **0.698**     | 0.330         |
| 128       | 0.786           | **0.820**     | 0.495         |
| 256       | 0.919           | **0.930**     | 0.695         |

Throughput (single-thread queries/s — **see the caveat below**, these are not
directly comparable):

| ef_search | KowitoDB (embedded) | Qdrant (service) | Milvus (service) |
|-----------|--------------------:|-----------------:|-----------------:|
| 32        | ~6700               | ~370             | ~850             |
| 64        | ~3300               | ~390             | ~1150            |
| 128       | ~2100               | ~430             | ~730             |
| 256       | ~850                | ~330             | ~430             |

## How to read this — honest caveats

- **Recall is the fair, apples-to-apples column.** It's network-independent and
  computed against identical ground truth. Takeaways:
  - **Qdrant has the best graph quality** at a given `ef` — its HNSW neighbor
    selection keeps the diversity heuristic from the paper. KowitoDB is close,
    and the gap **closes at high `ef`** (0.919 vs 0.930 at ef=256).
  - **KowitoDB sits between Qdrant and Milvus** at matched `ef`, and clearly
    above Milvus's default configuration here.
  - **Standard-HNSW mode (opt-in):** KowitoDB defaults to fast "keep closest M"
    selection with **unbounded degree** (no pruning). The full standard recipe —
    diversity heuristic (Alg. 4) **+ degree pruning** — is available via
    `HnswParams::diversify_neighbors` / `CMP_DIVERSIFY=1`. Measured A/B with this
    harness:
    - **Clustered data (real-embedding-like): a Pareto win at low `ef`.** ef=16
      recall **0.923 → 0.955 at the same ~36k QPS**; ef=32 0.985 → 0.994.
      Pruning bounds degree (keeps queries fast) while diversity keeps the graph
      navigable.
    - **Uniform random data: it *hurts*.** Pruning to M removes edges that the
      denser unbounded graph was using for recall (ef=32: 0.43 → 0.23). So the
      default leaves degrees unbounded — better on structureless data — and the
      standard recipe is opt-in for real clustered embeddings.
    - Neither standard technique closes the random-data gap to Qdrant on its own;
      that gap is down to implementation maturity / `ef_construction` tuning, not
      a single missing algorithm. This is an honest negative result.
  - **Milvus's lower recall is most likely a configuration/segmentation artifact**
    (segment sizing, default index params on a small set), not a fundamental
    limit — treat it as "default-config" rather than "best Milvus can do."
  - Random unit vectors are a deliberately **hard** recall case for everyone
    equally; real clustered embeddings yield higher recall at the same `ef`.

- **Throughput is NOT directly comparable.** KowitoDB is measured **in-process**
  (as a library — no network), while Qdrant and Milvus are measured as
  **localhost services**, so their per-query latency includes a client↔server
  round-trip (visible as a ~1–2.5 ms `p50` floor) that dominates the actual
  search time. So the QPS table mostly says *"embedded library vs networked
  service"*, which is a real deployment difference but **not** a measure of raw
  ANN speed. To compare search speed fairly you would run KowitoDB behind its own
  gRPC server and measure all three over the network — that round-trip would
  bring KowitoDB's numbers down to the same order as the others.

## What this benchmark is and isn't

- **Is:** a fair, reproducible **recall/quality** comparison of the HNSW indexes
  at matched parameters on identical data, plus an embedded-vs-service throughput
  snapshot.
- **Isn't:** a tuned, production-config bake-off (each system has many knobs:
  quantization, segment sizing, multi-threaded search, GPU, etc.), nor a
  network-fair latency comparison. Don't cite the throughput table as "X is N×
  faster than Y."
