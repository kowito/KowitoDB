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

3. Benchmark the embedded Rust competitor [embedvec](https://crates.io/crates/embedvec)
   on the **same** `dataset.bin` (it's an excluded crate so its heavy deps never
   touch the workspace):
   ```
   cd embedvec-bench
   CMP_M=16 CMP_EFC=128 CMP_EFS=32,64,128,256 cargo run --release -- ../dataset.bin
   ```

4. [velesdb](https://crates.io/crates/velesdb-core) is benchmarked the same way
   but is **not bundled here** — its VelesDB Core License 1.0
   (Elastic-License-2.0-style) prohibits competitive use, so we publish only its
   numbers (as with Qdrant/Milvus). To reproduce, in a throwaway crate *outside*
   this repo with `velesdb-core = "3.4"`, read the same `dataset.bin` and:
   ```rust
   db.create_vector_collection_with_hnsw("c", dim, DistanceMetric::Cosine,
       StorageMode::Full, Some(m), Some(efc))?;          // matched M / ef_construction
   col.upsert_bulk(&points)?;                            // Point::without_payload(id, vec)
   let hits = col.search_with_ef(query, k, ef_search)?;  // hits[i].point.id, hits[i].score
   ```

Each prints CSV rows `system,ef_search,recall@k,qps_1thread,p50_us,p95_us`.

## Results (measured)

50 000 × 128, 1 000 queries, k=10, cosine, HNSW `M=16`, `ef_construction=128`,
single machine (Apple Silicon, 4P+6E). `kowitodb` = default (unbounded degree);
`kowitodb-std` = standard-HNSW mode (`diversify_neighbors`).

**Recall@10 — uniform random vectors** (a deliberately hard, structure-free case):

| ef  | kowitodb | kowitodb-std | Qdrant\* | Milvus\*| embedvec | velesdb |
|-----|---------:|-------------:|---------:|-------:|---------:|--------:|
| 32  | 0.424    | 0.239        | **0.700**| 0.196  | 0.081    | 0.352   |
| 64  | 0.604    | 0.382        | **0.786**| 0.337  | 0.170    | 0.527   |
| 128 | 0.789    | 0.559        | **0.881**| 0.497  | 0.291    | 0.728   |
| 256 | 0.921    | 0.768        | **0.963**| 0.692  | 0.393    | 0.892   |

**Recall@10 — clustered vectors** (200 clusters; representative of real
embeddings):

| ef  | kowitodb | kowitodb-std | Qdrant\* | Milvus\*| embedvec | velesdb |
|-----|---------:|-------------:|---------:|-------:|---------:|--------:|
| 16  | 0.912    | 0.953        | **0.976**| 0.908  | 0.201    | 0.294   |
| 32  | 0.989    | 0.995        | **0.997**| 0.986  | 0.206    | 0.301   |
| 64  | 0.9997   | 0.9996       | 0.9998   | 0.999  | 0.232    | 0.311   |
| 128 | 0.9999   | 0.9999       | **1.000**| 0.9999 | 0.253    | 0.332   |

\* Qdrant/Milvus carried from prior measured runs against the **same** seeded
`dataset.bin` (their Docker services were not re-run for this refresh). KowitoDB's
parallel graph build is non-deterministic run-to-run, so its recall has ~±1%
single-run noise; numbers above are one fresh run.

### Visual (recall@10)

```
RANDOM 50k×128, M=16, efc=128          0    0.25  0.5   0.75   1.0
                                       +-----+-----+-----+-----+
ef=128  qdrant*       0.881  ███████████████████████████████████····
        kowitodb      0.789  ███████████████████████████████········
        velesdb       0.728  ████████████████████████████████········
        kowitodb-std  0.559  █████████████████████████···············
        milvus*       0.497  ██████████████████████··················
        embedvec      0.291  █████████████···························
ef=256  qdrant*       0.963  ██████████████████████████████████████··
        kowitodb      0.921  █████████████████████████████████████···
        velesdb       0.892  ███████████████████████████████████████·
        kowitodb-std  0.768  ██████████████████████████████████······
        milvus*       0.692  ██████████████████████████████·········
        embedvec      0.393  █████████████████·······················

CLUSTERED 50k×128 (200 clusters), M=16, efc=128
ef=16   qdrant*       0.976  ███████████████████████████████████████████·
        kowitodb-std  0.953  ██████████████████████████████████████████··
        kowitodb      0.912  ████████████████████████████████████████····
        milvus*       0.908  ████████████████████████████████████████····
        velesdb       0.294  █████████████·······························
        embedvec      0.201  █████████···································
ef=32   qdrant*       0.997  ████████████████████████████████████████████
        kowitodb-std  0.995  ████████████████████████████████████████████
        kowitodb      0.989  ███████████████████████████████████████████·
        milvus*       0.986  ███████████████████████████████████████████·
        velesdb       0.301  █████████████·······························
        embedvec      0.206  █████████···································
```

On **clustered (real-like) data the mature engines saturate by ef≈32** (KowitoDB,
Qdrant, Milvus all ~0.99); the two embedded crates flatline far below — velesdb
for tuning reasons (recovers with more `ef_construction`), embedvec structurally
(does not). On **random data** the order is Qdrant > kowitodb > velesdb >
kowitodb-std > milvus > embedvec, converging as `ef` rises.

> **velesdb (v3.4.0) — competent, but `ef_construction`-sensitive on clustered
> data.** Unlike embedvec, velesdb is a real, capable HNSW engine: on uniform
> random data at matched params it's the strongest non-Qdrant column (0.35 → 0.89),
> and a query for a stored vector correctly returns that vector at cosine 1.0. Its
> *low clustered numbers above are a parameter effect, not a ceiling* — at the
> matched `M=16, ef_construction=128` its graph is too sparse for 200 tight
> clusters, but with a **richer graph (`M=32, ef_construction=256`) it recovers to
> 0.82 @ ef=128 and 0.93 @ ef=512**, with self-query back to 1.0. So read velesdb's
> clustered row as "needs more build effort here," whereas KowitoDB reaches 0.99+
> at the lean matched params (more robust graph construction at low
> `ef_construction`). Measured **out-of-repo** (not bundled): velesdb-core is under
> a VelesDB Core License 1.0 (Elastic-License-2.0-style) that prohibits competitive
> use, so only its numbers are reproduced here — like Qdrant/Milvus, run it
> yourself with the harness sketch below.

> **embedvec (v0.8.0) at this scale.** embedvec is a young, single-file-friendly
> HNSW crate. At matched params on 50 000 vectors its recall is low and — tellingly
> — **flat in `ef`** on clustered data (0.20 → 0.25 from ef=16 to 128), the
> signature of a graph-connectivity ceiling rather than under-search. This is
> **not** a harness artifact: re-running **embedvec's own recall test** (its data
> generator + its brute-force ground truth) scaled from its shipped 500 vectors up
> to 50 000 reproduces it — recall@10 falls 0.94 (2k) → 0.79 (10k) → 0.69 (50k),
> and a query for a *stored* vector fails to return that vector (it's unreachable
> in the graph), even at `ef_search=1000` and `M=32, ef_construction=200`.
> embedvec's test suite only validates at ~500 vectors. Treat this as "early-stage
> crate, not yet tuned for tens of thousands of vectors," not a tuned bake-off.
> Harness: [`embedvec-bench/`](embedvec-bench/).

Among the mature engines, KowitoDB is competitive with Qdrant and **ahead of
Milvus's default config** on clustered data, and `kowitodb-std` closes most of the
low-ef gap (ef=16: 0.912 → 0.953 vs Qdrant 0.976). On **uniform random data Qdrant
leads clearly** — implementation maturity, not one missing algorithm (see below).

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
