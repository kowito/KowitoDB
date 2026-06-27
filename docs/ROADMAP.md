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
- ✅ **Follow-up shipped (v0.25):** **LLM-generated context** (the faithful
  Anthropic version) — with an `LlmClient` configured and
  `KOWITODB_LLM_CONTEXTUAL=1`, each object's indexed text is prefixed by an
  LLM-written situating sentence (stored content untouched); falls back to the
  deterministic preamble otherwise. Mock-tested.

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

### 3. Mem0-style memory → searchable knowledge — ✅ shipped (v0.25.0)
- ✅ **Shipped (v0.15):** `remember_turn` promotes each conversation turn into a
  searchable knowledge object (stable id → idempotent; `system` turns excluded),
  so past conversation is retrievable via `ai.ask()` and lives in the same store
  as ingested knowledge. The `RecordTurn` RPC routes through it.
- ✅ **Shipped (v0.16):** **memory↔entity graph edges** — a promoted memory is
  linked (`mentions`) to the existing knowledge it references (top full-text
  matches), so memories and facts are mutually traversable in the graph.
- ✅ **Shipped (v0.25):** **LLM-driven distillation** — with an `LlmClient`
  configured, a recorded turn is distilled to the single salient durable fact
  before promotion (or dropped on `NOOP`), so memory holds clean facts rather
  than raw chatter; the stable-id idempotency keys off the distilled fact. Mock-
  tested. (Full ADD/UPDATE/DELETE reconciliation against prior memories is a
  natural extension of the same `distill_memory` seam.)
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
- ✅ **NL→SQL shipped (v0.25):** with an `LlmClient` configured,
  `answer_with_sql()` translates an analytical question into a single SQL query
  over the `knowledge` table and executes it through the existing DataFusion
  path — real aggregates, not approximations. Mock-tested end-to-end (mock SQL →
  rows over the store); returns `None` (retrieval fallback) when no client.
- **Evidence:** 2025 evals show routing/integration beats RAG-or-GraphRAG alone
  (~+6%). — arxiv 2502.11371.

### 5. RaBitQ quantization (+ DiskANN) — ✅ shipped (v0.21.0) · 📋 DiskANN remaining
- ✅ **Shipped (v0.21):** RaBitQ-style **1-bit binary quantization** (~32× smaller
  vectors vs f32, ~8× vs int8). Each vector is rotated by a structured random
  rotation (random ±1 sign flip + normalized fast Walsh–Hadamard transform —
  orthonormal, O(d log d), decorrelates coordinates so sign quantization is a
  well-behaved estimator), stored as one sign bit/dim plus two scalars, and the
  graph is navigated in the rotated basis. Distances use the **unbiased RaBitQ
  estimator** (`‖o‖² + ‖q‖² − 2·factor·⟨sign,q̃⟩/√D`). Enabled with
  `KOWITODB_VECTOR_BINARY_QUANTIZE=1` (or `HnswParams::binary_quantize`); the
  rotation persists in the index snapshot. Recall@10 validated vs brute force in
  a unit test; exact matches stay near top-1.
- ✅ **Popcount fast path (v0.23):** binary navigation now scores via a
  **Hamming/popcount** distance over the packed sign codes (no float ops),
  refined to the asymmetric estimator for the final top-k — a measured **~3.4×
  query speedup** on top of the memory win (see BENCHMARKS). Generalized the
  graph traversal to a `Scorer` (full / coarse-prefix / Hamming).
- ✅ **Oversample → rescore (v0.24):** `binary_rerank` /
  `KOWITODB_VECTOR_BINARY_RERANK=1` retains an int8 copy of each vector and
  re-scores the oversampled top-k with it — the production pattern that recovers
  recall the 1-bit codes lose. Measured: **~43% → ~79% recall@10 at the same
  ~3.7× speedup**, trading the 32× memory win down to int8's ~4× (navigation
  still uses the popcount fast path; only the final top-k touches int8).
- **Honest scope:** binary is now a memory **and** speed win, with a recall knob
  spanning 1/32× memory (low recall) → int8 memory (high recall). 📋 **Remaining:**
  **DiskANN**-style on-disk graph for billion-scale on SSD (where full vectors
  live on disk and only codes stay in RAM).
- **Maturity:** RaBitQ SIGMOD 2024; production in VectorChord/DiskANN.

---

## Potential / exploratory (🔬)

- **Late interaction (ColBERTv2 / PLAID):** multi-vector MaxSim; +6–10× storage
  compression, PLAID ~45× CPU speedup. Highest quality ceiling, highest effort
  (changes the index model). — arxiv 2112.01488, 2205.09707.
- ✅ **Matryoshka embeddings (MRL) — shipped (v0.22.0):** adaptive-dimension
  retrieval. With `HnswParams::coarse_dim = Some(d)` (or
  `KOWITODB_VECTOR_COARSE_DIM=d`) the HNSW graph is navigated using only the
  first `d` dimensions (a cheap coarse pass), then the final top-k is refined at
  full dimension so the returned ranking stays exact. Requires MRL-trained
  embeddings whose prefixes are valid (OpenAI `text-embedding-3`); ignored under
  binary quantization (the rotation precludes prefixes). Recall@10 ≥ 0.7 vs brute
  force validated in a unit test. Pairs with the quantization + reranker layers.
- ✅ **LazyGraphRAG-style auto-graph — shipped (v0.26):** a cheap deterministic
  entity extractor at ingest (capitalized proper nouns + keywords) links each
  object to prior objects sharing an entity via bidirectional `co_mentions`
  graph edges, so the graph is useful even without explicit relationships — the
  LazyGraphRAG insight (a light extractor enriches the graph at ~0.1% of full
  GraphRAG cost) without any LLM. Default on (`KOWITODB_AUTO_GRAPH=0` disables),
  fan-out bounded. Unit-tested.
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
  an object, converging after a W-of-R quorum write).
- ✅ **Last-write-wins reconciliation (v0.26):** `get` now returns the *freshest*
  copy (latest `updated_at`) across replicas and repairs any replica that is
  missing, staler, or content-divergent — so divergent copies converge on read,
  not just missing ones. Unit-tested with seeded divergent replicas.
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
