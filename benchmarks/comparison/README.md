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

50 000 × 128, 1 000 queries, k=10, cosine, HNSW `M=16`, `ef_construction=128`,
single machine (Apple Silicon, 4P+6E). `kowitodb` = default (unbounded degree);
`kowitodb-std` = standard-HNSW mode (`diversify_neighbors`).

**Recall@10 — uniform random vectors** (a deliberately hard, structure-free case):

| ef  | kowitodb | kowitodb-std | Qdrant   | Milvus |
|-----|---------:|-------------:|---------:|-------:|
| 32  | 0.431    | 0.234        | **0.700**| 0.196  |
| 64  | 0.602    | 0.380        | **0.786**| 0.337  |
| 128 | 0.788    | 0.564        | **0.881**| 0.497  |
| 256 | 0.921    | 0.763        | **0.963**| 0.692  |

**Recall@10 — clustered vectors** (200 clusters; representative of real
embeddings):

| ef  | kowitodb | kowitodb-std | Qdrant   | Milvus |
|-----|---------:|-------------:|---------:|-------:|
| 16  | 0.915    | 0.957        | **0.976**| 0.908  |
| 32  | 0.994    | 0.994        | **0.997**| 0.986  |
| 64  | 0.9997   | 0.9996       | 0.9998   | 0.999  |
| 128 | 0.9999   | 0.9999       | **1.000**| 0.9999 |

On **clustered (real-like) data the systems converge** (~0.99+ by ef=32);
KowitoDB is competitive with Qdrant and **ahead of Milvus's default config**, and
`kowitodb-std` closes most of the low-ef gap (ef=16: 0.915 → 0.957 vs Qdrant
0.976). On **uniform random data Qdrant leads clearly** — implementation maturity,
not one missing algorithm (see below).

Throughput (single-thread q/s — **NOT directly comparable**, see caveats):
KowitoDB is measured **embedded** (~6–40k q/s, no network); Qdrant and Milvus as
**localhost services** (~0.4–1.4k q/s, dominated by a ~1–2.5 ms HTTP/gRPC
round-trip floor). This reflects deployment mode (library vs service), not raw
ANN speed.

## How to read this — honest caveats

- **Recall is the fair, apples-to-apples column.** It's network-independent and
  computed against identical ground truth. Takeaways:
  - **Qdrant has the best graph quality** at a given `ef`, most visibly on the
    hard uniform-random set. **On clustered (real-like) data the gap is small and
    everyone converges** by ef≈32 (~0.99+) — which is the regime real embeddings
    live in.
  - **KowitoDB is competitive with Qdrant on clustered data and ahead of Milvus's
    default config**; on uniform random data it trails Qdrant but leads Milvus.
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
