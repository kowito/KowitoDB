# KowitoDB Roadmap

Research-grounded direction for KowitoDB. The thesis: **the moat is the
integrated substrate** — vector + native BM25 + reranker + knowledge graph +
agent memory + query planner + SQL in one engine. The highest-leverage work is
whatever exploits ≥2 of those pieces that a pure vector DB doesn't have.

Sources for the research items are in [BENCHMARKS.md](BENCHMARKS.md)'s sibling
research notes; each item below tags maturity and the key paper/system.

**Legend:** ✅ shipped · 🔜 in progress · 📋 planned · 🔬 potential / exploratory · ⛔ deliberately not pursuing

---

## Recently shipped (v0.2 → v0.8)

- ✅ Data plane: batch insert, metadata-filtered retrieval, list/pagination
- ✅ Real embeddings (on-device Candle / OpenAI / Ollama), HNSW recall fix (~94%)
- ✅ On-disk vector index persistence (no rebuild on restart)
- ✅ Hot-path optimization (ahash, squared-dist, alloc-free) — ~22% faster queries
- ✅ Sharded parallel HNSW — ~20× faster build, ~5.5× concurrent query, horizontal partitioning
- ✅ int8 scalar quantization — ~4× less vector memory
- ✅ Persisted agent memory; auth/TLS/health/reflection/Prometheus; LangChain + LlamaIndex adapters

---

## Game-changers (research-grounded, prioritized)

### 1. Contextual Retrieval — 🔜 in progress
Prepend chunk-specific context to the text that gets embedded **and** BM25-indexed,
while storage returns the original content.
- **Evidence (VERIFIED 3-0):** Contextual Embeddings + Contextual BM25 cut top-20
  retrieval failure rate **49%**; with reranking **67%**. — Anthropic, 2024.
- **Why us:** needs vector **and** BM25 **and** rerank — we already have all three.
- **First cut (this work):** deterministic context from structured fields
  (metadata + keywords). **Potential:** LLM-generated context (faithful Anthropic
  version) behind an optional generative-client hook.

### 2. Cross-encoder / LLM reranker — 📋 planned
Replace the planner's score-fusion `rerank_simple` with a real reranker (the step
that takes the 49% → 67% failure-rate reduction above).
- **Maturity:** production-proven (cross-encoders, Cohere/BGE rerankers).
- **Potential:** on-device Candle cross-encoder (e.g. `bge-reranker`) so it stays
  offline-capable, mirroring the local-embeddings feature.

### 3. Mem0-style memory consolidation, fused into the graph — 📋 planned (category-definer)
Replace flat turn storage with an LLM-driven extract → consolidate → update
pipeline (ADD/UPDATE/DELETE/NOOP) and link memories as nodes/edges in the graph.
- **Maturity:** Mem0 (Chhikara et al., Apr 2025), lineage MemGPT (2023).
- **Why us:** we already persist agent memory **and** have a graph index — this is
  the most *differentiating* item; it makes "agent-memory OS" real.

### 4. Query routing in the planner — 📋 planned
Route each query by intent: fact → vector/BM25, reasoning/multi-hop → graph,
analytical → SQL.
- **Evidence:** systematic 2025 evals show GraphRAG does **not** uniformly beat RAG;
  routing/integration beats either alone (~+6% via integration). — arxiv 2502.11371.
- **Why us:** we already have a planner with intent detection + all three backends.

### 5. RaBitQ quantization (+ DiskANN) — 🔬 potential (the real scale lever)
Upgrade int8 SQ → ~1 bit/dim with a *theoretical* error bound (~32× compression);
DiskANN + RaBitQ enables billion-scale on SSD.
- **Maturity:** RaBitQ SIGMOD 2024; production in VectorChord/DiskANN.
- **Effort:** medium-high (math-heavy). Slots into the existing `quantize` param.

---

## Potential / exploratory (🔬)

- **Late interaction (ColBERTv2 / PLAID):** multi-vector MaxSim; +6–10× storage
  compression, PLAID ~45× CPU speedup. Highest quality ceiling, highest effort
  (changes the index model). — arxiv 2112.01488, 2205.09707.
- **Matryoshka embeddings (MRL):** nested dims → adaptive-precision retrieval
  (cheap coarse pass → full-dim rerank). Pairs with quantization + reranker; low
  effort if the embedding model supports it (OpenAI `text-embedding-3` does).
- **LazyGraphRAG-style auto-graph:** cheap entity/relation extraction at ingest to
  enrich the graph index (~0.1% of full GraphRAG indexing cost, *unverified*).
- **Out-of-core / billion-scale:** DiskANN-style on-disk graph + RaBitQ; the path
  beyond RAM.
- **Multimodal embeddings:** image/text in the same store (knowledge objects are
  already content-agnostic).
- **Corrective gate (CRAG-style):** a lightweight retrieval-quality evaluator that
  gates re-retrieval (mechanism verified 2-0; "big quality win" claim did not
  survive verification — adopt as a cheap confidence gate, not a miracle).

## Scale / infra (📋)

- **Parallel HNSW build via fine-grained locking** (current build holds a global
  write lock; sharding sidesteps it but per-shard build is still serial).
- **Cross-machine sharding + replication** — true horizontal scale / HA. Large
  distributed-systems effort; sharding (shipped) is the in-process foundation.

## Deliberately not pursuing (⛔)

- **Heavy LLM GraphRAG (full community summarization):** task-dependent, not a
  blanket win, and very costly to index (~79M tokens in one benchmarked config).
  Prefer routing (#4) + cheap graph enrichment instead.
- **GPU indexes (CAGRA):** off-thesis for a single-node, on-device-friendly engine.
