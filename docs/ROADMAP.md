# KowitoDB Roadmap

Research-grounded direction for KowitoDB. The thesis: **the moat is the
integrated substrate** — vector + native BM25 + reranker + knowledge graph +
agent memory + query planner + SQL in one engine. The highest-leverage work is
whatever exploits ≥2 of those pieces that a pure vector DB doesn't have.

Sources for the research items are in [BENCHMARKS.md](BENCHMARKS.md)'s sibling
research notes; each item below tags maturity and the key paper/system.

**Legend:** ✅ shipped · 🔜 in progress · 📋 planned · 🔬 potential / exploratory · ⛔ deliberately not pursuing

---

## Recently shipped (v0.2 → v0.10)

- ✅ Retrieval quality: Contextual Retrieval (v0.9), CRAG-style corrective gate
  (v0.10), Mem0 searchable memory + graph links (v0.15–0.16), importance- and
  recency-weighted ranking (v0.17–0.18); RRF hybrid fusion + rule-engine query
  routing were already present

- ✅ Data plane: batch insert, metadata-filtered retrieval, list/pagination
- ✅ Real embeddings (on-device Candle / OpenAI / Ollama), HNSW recall fix (~94%)
- ✅ On-disk vector index persistence (no rebuild on restart)
- ✅ Hot-path optimization (ahash, squared-dist, alloc-free) — ~22% faster queries
- ✅ Sharded parallel HNSW — ~20× faster build, ~5.5× concurrent query, horizontal partitioning
- ✅ int8 scalar quantization — ~4× less vector memory
- ✅ Persisted agent memory; auth/TLS/health/reflection/Prometheus; LangChain + LlamaIndex adapters

---

## Game-changers (research-grounded, prioritized)

### 1. Contextual Retrieval — ✅ shipped (v0.9.0)
Embed and BM25-index a context-augmented text (deterministic preamble from
metadata + keywords) while storage returns the original content.
- **Evidence (VERIFIED 3-0):** Contextual Embeddings + Contextual BM25 cut top-20
  retrieval failure rate **49%**; with reranking **67%**. — Anthropic, 2024.
- **Follow-up (📋):** LLM-generated context (faithful Anthropic version) behind an
  optional generative-client hook.

### 2. Reranking — RRF fusion ✅ done · cross-encoder ✅ shipped (v0.19.0)
- ✅ **Already present:** the reranker does Reciprocal Rank Fusion across
  vector/BM25/graph/metadata/time with per-source weights, multi-source boosting,
  and normalization (`Reranker::rerank`).
- ✅ **Shipped (v0.19):** a learned **cross-encoder reranker** (the step that takes
  Anthropic's 49% → 67%). A `CrossEncoder` trait (`rerank.rs`) is always compiled
  so the engine can hold an optional second-stage reranker; the on-device Candle
  `bge-reranker-base` impl (`BertForSequenceClassification`: BERT encoder + a
  single-logit `[CLS]` head) is behind the `cross-encoder-rerank` feature,
  mirroring local-embeddings. Enabled via `KOWITODB_RERANKER_PROVIDER=local`;
  `ask()` re-scores and re-sorts the loaded top-k through it after RRF fusion.
  Trait + wiring are unit-tested with a mock; the Candle path is compile-verified
  (runtime needs the HF model). `KOWITODB_RERANKER_MODEL` overrides the model.

### CRAG-style corrective gate — ✅ shipped (v0.10.0)
When retrieval confidence is low (few results / little cross-source agreement),
broaden the search across vector + keyword and re-rank. The *mechanism* (a
lightweight retrieval-quality evaluator) was the verified part of CRAG.
`KOWITODB_CORRECTIVE_RETRIEVAL=0` disables.

### 3. Mem0-style memory → searchable knowledge — 🔜 in progress (category-definer)
- ✅ **Shipped (v0.15):** `remember_turn` promotes each conversation turn into a
  searchable knowledge object (stable id → idempotent; `system` turns excluded),
  so past conversation is retrievable via `ai.ask()` and lives in the same store
  as ingested knowledge. The `RecordTurn` RPC routes through it.
- ✅ **Shipped (v0.16):** **memory↔entity graph edges** — a promoted memory is
  linked (`mentions`) to the existing knowledge it references (top full-text
  matches), so memories and facts are mutually traversable in the graph.
- 📋 **Remaining:** LLM-driven extract → consolidate → update (ADD/UPDATE/DELETE/
  NOOP) for salient-fact distillation.
- **Why us:** we already persist agent memory **and** have a graph index — the
  most *differentiating* item; it makes "agent-memory OS" real.

### 4. Query routing in the planner — ✅ enhanced (v0.20.0) · 📋 NL→SQL remaining
- ✅ **Already present:** a rule-engine planner maps detected intent → retrieval
  actions (vector / keyword / graph / metadata / time), so routing exists.
- ✅ **Shipped (v0.20):** **intent-conditioned fusion weights** — the reranker's
  per-source RRF weights are now scaled by the detected intent
  (`Reranker::rerank_for_intent`), so e.g. temporal queries lean on the time
  index, entity queries on the graph + exact full-text, code queries on exact
  full-text, and analytical/listing queries on broad metadata recall. Added an
  `Analytical` intent ("how many / average / count of …") routed to wide
  structured recall rather than tight top-k semantic. Unit-tested (intent shifts
  ranking; `General` matches the base weights).
- 📋 **Remaining:** NL→SQL routing for analytical queries (execute real
  DataFusion aggregates) — needs a generative client for the NL→SQL step.
- **Evidence:** 2025 evals show routing/integration beats RAG-or-GraphRAG alone
  (~+6%). — arxiv 2502.11371.

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

## Scale / infra

- ✅ **Distributed mode (v0.11)** — a `gateway` coordinator fronts N data nodes:
  writes are partitioned by id (consistent `id % N`) and optionally replicated;
  reads scatter-gather across nodes and merge (search/ask de-dup + re-rank;
  stats/list aggregate). Speaks the same gRPC API, so SDKs are unchanged.
- 🔜 **Toward HA (v0.12–0.14):** tunable **write quorum** (`--write-quorum`),
  **failure-aware reads** (tolerate partial failure, error only on total outage),
  a **heartbeat health layer** (v0.13 — probe nodes every ~5s, skip unhealthy on
  reads, auto-recover), and **read-repair** (v0.14 — `get` heals replicas missing
  an object, converging after a W-of-R quorum write). Remaining 📋: Raft/consensus
  for linearizable reads, automatic rebalancing on membership change, version
  reconciliation for divergent copies, and a distributed-SQL planner.
- 📋 **Parallel HNSW build via fine-grained locking** (per-shard build is still
  serial; sharding sidesteps the global write lock at the cluster level).

## Honest limits of distributed mode (v0.11)

Real horizontal distribution, but **not** production HA: best-effort replication
(write fan-out, no quorum/consensus), no automatic rebalancing or failure
recovery, cross-shard SQL aggregates are per-shard partials, and the gateway is a
single coordinator. These are the 📋 items above.

## Deliberately not pursuing (⛔)

- **Heavy LLM GraphRAG (full community summarization):** task-dependent, not a
  blanket win, and very costly to index (~79M tokens in one benchmarked config).
  Prefer routing (#4) + cheap graph enrichment instead.
- **GPU indexes (CAGRA):** off-thesis for a single-node, on-device-friendly engine.
