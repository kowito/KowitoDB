# kowitodb-index

The retrieval engines behind **[KowitoDB](https://github.com/kowito/KowitoDB)** —
the database built for AI agents. One crate, six index types that `ai.ask()`
queries in parallel and fuses.

## The indexes

| Type | What it does |
|------|--------------|
| **`HnswIndex`** / **`HnswParams`** | Approximate nearest-neighbor vector search (HNSW) with int8 / RaBitQ-binary / Matryoshka quantization and on-disk snapshots. |
| **`ShardedHnswIndex`** | Sharded HNSW for parallel build and horizontal partitioning. |
| **`FullTextIndex`** | Full-text / BM25 keyword search (Tantivy). |
| **`GraphIndex`** | Knowledge graph with relationship traversal. |
| **`MetadataIndex`** | Structured metadata filtering. |
| **`TimeIndex`** | Time-range queries over timestamps. |
| **`MultiVectorIndex`** | Late-interaction / ColBERT-style MaxSim multi-vector retrieval. |

```rust
use kowitodb_index::{HnswIndex, HnswParams};

let index = HnswIndex::new(HnswParams { m: 16, ef_construction: 128, ..Default::default() });
index.insert(id, embedding);                  // embedding: Vec<f32>
let hits = index.search(&query, 10);          // -> Vec<(ObjectId, f32)>, nearest first
```

A reproducible recall/throughput comparison of this HNSW index vs Qdrant, Milvus,
embedvec, and velesdb lives in
[`benchmarks/comparison/`](https://github.com/kowito/KowitoDB/tree/main/benchmarks/comparison).

## Where it fits

```
kowitodb-core (types)
  └─ kowitodb-index  ← you are here (retrieval)
       └─ kowitodb-planner (ranking) → kowitodb-server (engine + gRPC) → kowitodb (CLI)
```

For the full feature tour, quickstart, and SDKs, see the
**[project README](https://github.com/kowito/KowitoDB#readme)**.

## License

[MIT](https://github.com/kowito/KowitoDB/blob/main/LICENSE)
