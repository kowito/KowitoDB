# embedvec-bench

Benchmarks [embedvec](https://crates.io/crates/embedvec) on the **same**
`dataset.bin` as KowitoDB / Qdrant / Milvus, scored against the **same**
brute-force ground truth at **matched** HNSW params — the fair, apples-to-apples
recall column.

This is a **standalone crate, excluded from the workspace** (see the root
`Cargo.toml` `exclude`), so embedvec's dependency tree never touches the
published KowitoDB crates or CI.

```bash
# dataset.bin is written by `bench_compare` (see ../README.md)
CMP_M=16 CMP_EFC=128 CMP_EFS=32,64,128,256 cargo run --release -- ../dataset.bin
```

Env: `CMP_M`, `CMP_EFC`, `CMP_EFS` (comma-separated `ef_search` list). Output is
CSV rows `system,ef_search,recall@k,qps_1thread,p50_us,p95_us`.

## Note on results

embedvec (v0.8.0) is an early-stage crate whose recall degrades with dataset
size — at 50 000 vectors recall@10 stays ~0.19–0.41 at matched params and is
nearly flat in `ef_search`, a graph-connectivity ceiling. This reproduces with
**embedvec's own recall test** scaled past its shipped 500-vector validation set,
so it's a property of the index, not this harness. See the comparison
[`../README.md`](../README.md) for the full table and caveats.
