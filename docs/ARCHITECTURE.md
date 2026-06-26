# KowitoDB Architecture

This document describes how KowitoDB is structured and how a query flows through
it. It is written against the v0.1.0 code; where behavior is partial or a
stated design goal rather than reality, that is called out explicitly.

## Contents

- [Workspace layout](#workspace-layout)
- [Crate-by-crate breakdown](#crate-by-crate-breakdown)
- [The knowledge object](#the-knowledge-object)
- [The retrieval pipeline](#the-retrieval-pipeline)
- [The six indexes](#the-six-indexes)
- [The storage abstraction](#the-storage-abstraction)
- [The DataFusion SQL path](#the-datafusion-sql-path)
- [Cross-cutting subsystems](#cross-cutting-subsystems)
- [Persistence model](#persistence-model)

## Workspace layout

A Cargo workspace of seven crates:

```
kowitodb-core      Knowledge object, shared types, error types.
kowitodb-storage   StorageBackend trait + sled engine (+ optional Lance engine).
kowitodb-index     The six indexes: HNSW, brute-force vector, Tantivy full-text,
                   metadata, time, graph.
kowitodb-planner   Intent analysis, rule engine, execution plans, reranker,
                   context optimizer, cost tracker, query cache.
kowitodb-sql       SQL: DataFusion TableProvider/SqlContext + a lightweight parser.
kowitodb-server    KowitoDBEngine (wires everything together), gRPC service,
                   embedding clients, agent memory, metrics.
kowitodb           CLI binary (serve / ask / insert / sql / stats).
```

Dependency direction is strictly downward: `kowitodb` → `kowitodb-server` →
{`index`, `planner`, `sql`, `storage`} → `kowitodb-core`.

## Crate-by-crate breakdown

### `kowitodb-core`

Defines the domain model and shared types:

- `KnowledgeObject` — the central record (see below).
- `ObjectId` (a UUID v4), `Embedding` (a `Vec<f32>`), `Timestamp`,
  `Relationship`.
- `KowitoError` / `Result` — the workspace error type (`Storage`, `Io`,
  `Serialization`, `Planner`, … variants).

No I/O, no async — pure data.

### `kowitodb-storage`

Owns the `StorageBackend` trait and its implementations.

- `StoredObject` — the serialized projection of a `KnowledgeObject` (content +
  JSON-encoded metadata/keywords/relationships/embeddings + importance +
  ISO-8601 timestamps).
- `StorageFilter` — a filter struct (id, keyword, importance floor, date range,
  limit) for `search`.
- `StorageEngine` — the **default sled-backed** engine, with an in-process
  `DashMap` content cache. `open(path)` is persistent; `new_in_memory()` uses a
  throwaway temporary sled DB for tests.
- `LanceStorage` — the optional **Lance** engine (feature `lance`).

See [The storage abstraction](#the-storage-abstraction).

### `kowitodb-index`

The retrieval indexes. Each is a self-contained, thread-safe (`Arc<RwLock<…>>`
or `DashMap`) structure. See [The six indexes](#the-six-indexes).

### `kowitodb-planner`

The query-planning and post-processing logic — the part `idea.md` calls "the
moat". No external search engines here; pure planning logic.

- `IntentAnalyzer` / `DetectedIntent` / `Intent` — rule-based intent
  classification and entity extraction.
- `RuleEngine` — maps an intent to an ordered list of retrieval actions.
- `QueryPlanner` (`optimizer.rs`) — ties intent + rules into an `ExecutionPlan`.
- `ExecutionPlan` / `PlanStep` / `PlanStepType` — the plan representation, with
  an `explain()` rendering used in API responses.
- `Reranker` / `RankedResult` — Reciprocal Rank Fusion across index sources.
- `ContextOptimizer` / `AssembledContext` — dedup + token-budgeted assembly.
- `CostTracker` / `CostModel` — running USD cost estimate.
- `QueryCache` / `CacheStats` — TTL + capacity cache (used for the plan cache).

### `kowitodb-sql`

- `SqlContext` / `KnowledgeTableProvider` — the DataFusion integration.
- `parse` (`parse_sql`) + `SqlStatement` / `WhereClause` — the lightweight
  index-routed parser path.

### `kowitodb-server`

- `KowitoDBEngine` — the orchestrator that holds storage, all indexes, the
  planner, the optimizers, the embedding client, the plan cache, and agent
  memory, and implements `insert` / `get` / `delete` / `ask` / `sql_query` /
  `sql_select` / `stats`.
- `KowitoDBService` — the tonic gRPC service that adapts the engine to the proto.
- `EmbeddingClient` trait + `ProxyEmbeddingClient` (deterministic, default) and
  `OpenAiEmbeddingClient` (OpenAI-compatible HTTP).
- `AgentMemory` — in-memory conversation/session store.
- `MetricsCollector` — request counters and latencies.
- `serve(engine, addr)` — builds the tonic server and serves.

### `kowitodb` (binary)

A `clap`-based CLI: `serve`, `ask`, `insert`, `sql`, `stats`. `serve` starts the
gRPC server; the other commands construct an embedded engine against the data
directory and run in-process.

## The knowledge object

Every record stored in KowitoDB is a `KnowledgeObject`:

| Field | Type | Notes |
| --- | --- | --- |
| `id` | UUID v4 | Generated on construction. |
| `content` | string | Raw text. |
| `embeddings` | map model → vector | Zero or more named embeddings. Auto-filled on insert if empty. |
| `metadata` | map string → JSON value | Arbitrary key/value attributes. |
| `keywords` | list of strings | Indexed into full-text search. |
| `relationships` | list of `Relationship` | `relation_type`, `target_id`, optional `weight`. |
| `importance` | f32 | 0.0–1.0, clamped on the wire. |
| `created_at` / `updated_at` | timestamps | RFC-3339 in storage. |
| `version_history` | list | Present in the type; not populated by the current write path. |

On `insert`, the engine fans the object out to the object store **and** every
index simultaneously.

## The retrieval pipeline

`KowitoDBEngine::ask(question, max_results)` is the core path. Steps, in order:

1. **Plan (cached).** Look up `(DetectedIntent, ExecutionPlan)` in the plan
   cache keyed by the raw question. On miss, run `QueryPlanner::plan`:
   - `IntentAnalyzer` classifies the question into one `Intent` and extracts
     entities (named tokens, dates, keywords, metadata filters, comparison/code
     flags).
   - `RuleEngine` evaluates ordered rules against the intent/entities and emits
     an ordered, de-duplicated list of retrieval actions.
   - The plan is assembled from those actions, then terminal `Merge` →
     `Rerank` → `BuildContext` steps are appended. `plan.explain()` renders the
     human-readable plan string returned to the caller.
   - The result is cached; **any `insert`/`delete` clears the plan cache** so
     stale plans are not reused against a changed corpus.

2. **Embed the query.** The engine embeds the question via the configured
   `EmbeddingClient` (default: deterministic proxy) for the vector step, and
   records the embedding call with the cost tracker.

3. **Execute the plan steps.** `execute_plan` walks `plan.steps` and runs the
   ones it implements:
   - `VectorSearch` → `HnswIndex::search(query_embedding, limit)`.
   - `KeywordSearch` → `FullTextIndex::search(keywords-or-question, limit)`.
   - `TimeFilter` → `TimeIndex::before(now)` when the query carries dates.
   - `MetadataFilter` → `MetadataIndex::query_exact(key, value)` per extracted
     filter.
   - Each producing step yields an `IndexResult { ids, scores, source }`. Other
     step types (e.g. `CodeSearch`) are logged and deferred.

4. **Graph traversal.** `execute_graph_traversal` seeds from all IDs found so
   far and runs `GraphIndex::scored_bidirectional_traverse` (depth 2 for
   entity-heavy queries, else depth 1), following both outgoing and incoming
   edges. New nodes (not already in the seed set) are emitted as a `Graph`
   `IndexResult`. Traversal scores decay with depth (`1 / (1 + depth)`).

5. **Rerank.** `Reranker::rerank_simple` fuses all `IndexResult`s using
   Reciprocal Rank Fusion (k = 60), weights each source (Vector 1.5, Graph 1.3,
   FullText 1.2, Metadata 0.8, Time 0.7), applies a cross-source agreement boost
   (`× (1 + 0.15·(sources−1))`) for objects found by multiple indexes, then
   normalizes scores to `[0, 1]` and sorts descending.

6. **Load content.** The top `max_results` ranked IDs are resolved to real
   content via the content cache, falling back to the object store.

7. **Assemble context.** `ContextOptimizer::assemble` deduplicates near-identical
   chunks (Jaccard similarity over word sets), sorts by relevance, and greedily
   fills a token budget (default 4096; ~4 chars ≈ 1 token), trimming long chunks
   at sentence/word boundaries. It reports `total_tokens` and a
   `compression_ratio` (`1 − trimmed/raw`).

8. **Respond.** The engine returns results, the plan explanation, the detected
   intent string, the total context tokens, and the compression ratio. The
   gRPC layer maps this to `AskResponse`.

```
question
  │  plan cache hit? ──► (intent, plan)
  ▼  miss
IntentAnalyzer ─► RuleEngine ─► ExecutionPlan (+Merge/Rerank/BuildContext)
  ▼
embed(question)
  ▼
execute_plan ──► [Vector] [Keyword] [Time] [Metadata]  ──► IndexResults
  ▼
graph traversal (bidirectional, depth-scored)          ──► IndexResult(Graph)
  ▼
Reranker (RRF + source weights + agreement boost + normalize)
  ▼
load content (cache → storage)
  ▼
ContextOptimizer (dedup → sort → token budget)
  ▼
AskResponse { results, plan_explanation, detected_intent, total_tokens, compression_ratio }
```

## The six indexes

All six live in `kowitodb-index`. The `IndexSource` enum tags results by origin
(`Vector`, `FullText`, `Metadata`, `Time`, `Graph`).

| Index | Backing structure | Algorithm | Persisted? |
| --- | --- | --- | --- |
| **HNSW vector** | in-memory graph (`HashMap` of nodes/layers) | Custom Hierarchical Navigable Small World ANN; distance → similarity `1/(1+d)`; params `m`, `ef_construction`, `ef_search` | **No** |
| **Brute-force vector** | `HashMap<(id, model), vec>` | Exact cosine similarity; the swappable predecessor to HNSW | **No** |
| **Full-text** | Tantivy index on disk | Inverted index + BM25; fields `id`, `content`, `keywords`, `metadata`; `commit()` to make writes searchable | **Yes** (`{index-path}/tantivy/`) |
| **Metadata** | nested `HashMap` (key → value → ids) | Exact match + substring (`query_contains`) | **No** |
| **Time** | `BTreeMap<i64, Vec<id>>` + reverse map | `before` / `after` / `between` range queries on creation-time ms | **No** |
| **Graph** | dual adjacency maps (forward + reverse) | BFS, bidirectional traversal, depth scoring, shortest path | **No** |

The "six" are HNSW + full-text + metadata + time + graph, with the brute-force
vector index retained as a swappable alternative to HNSW. In the live `ask`
path the engine uses **HNSW** for vector search.

## The storage abstraction

`StorageBackend` is the single trait the rest of the engine writes through:

```rust
#[async_trait]
pub trait StorageBackend: Send + Sync {
    async fn put(&self, obj: StoredObject) -> Result<()>;
    async fn get(&self, id: ObjectId) -> Result<Option<StoredObject>>;
    async fn delete(&self, id: ObjectId) -> Result<bool>;
    async fn search(&self, filter: StorageFilter) -> Result<Vec<StoredObject>>;
    async fn count(&self) -> Result<usize>;
    async fn list_ids(&self) -> Result<Vec<ObjectId>>;
}
```

### sled (default)

`StorageEngine` stores each object as JSON under its UUID key in a sled tree and
caches content in a `DashMap`. `get` checks the cache first, then sled; `search`
performs a full scan applying the `StorageFilter` in Rust (no secondary index).
Persistent across restarts. `new_in_memory()` opens a temporary sled DB and is
used by tests.

### Lance (optional, feature `lance`)

`LanceStorage` persists `StoredObject`s as an Arrow table in a
[Lance](https://lancedb.github.io/lance/) dataset:

- The Arrow schema mirrors every `StoredObject` field one-to-one.
- `put` is an upsert: delete-then-append keyed on `id`. The dataset is created
  lazily on the first write.
- `get` filters by `id = '…'`; `search` materializes all rows and filters in
  Rust to match the sled engine's semantics (predicate pushdown into Lance is a
  noted future optimization).
- The server can select it via `KowitoDBEngine::new_with_lance(uri, index_path)`
  when `kowitodb-server` is built with `--features lance`.

Both backends produce identical `StoredObject`s, so the indexes and SQL layer
are backend-agnostic.

## The DataFusion SQL path

`kowitodb-sql` exposes stored objects to Apache DataFusion through a custom
`TableProvider`:

- `KnowledgeTableProvider::arrow_schema()` defines a flat table: `id`,
  `content`, `importance`, `created_at`, `updated_at`, `keywords`, `metadata`.
  `keywords` and `metadata` are surfaced as their **JSON string** encodings, so
  they stay queryable (`metadata LIKE '%"stage":"series_a"%'`) without pinning a
  metadata schema.
- On `scan`, the provider reads the current object set from the
  `StorageBackend` (`search(StorageFilter::default())`), packs it into one Arrow
  `RecordBatch`, wraps it in a `MemTable`, and delegates projection/limit to
  DataFusion. Remaining predicates are applied by a `FilterExec` the optimizer
  inserts above the scan. This makes each query a snapshot of storage at scan
  time.
- `SqlContext::new(storage)` registers the provider under both `knowledge` and
  `objects`. `query_rows(sql)` runs the query and returns each row as a
  `HashMap<String, String>` (values stringified via Arrow display formatting).

`KowitoDBEngine::sql_select(sql)` is the engine entry point for this path and
supports projection, `WHERE`, `ORDER BY`, `GROUP BY`, aggregates, and `LIMIT`.

The separate, lighter `KowitoDBEngine::sql_query` (used by the `kowitodb sql`
CLI command) parses a SQL subset with `parse_sql` and routes `WHERE` clauses
directly to the metadata / full-text / time indexes, intersecting candidate
sets with AND semantics. It does not do projection or aggregation — use
`sql_select` for those.

## Cross-cutting subsystems

- **Embedding clients** (`kowitodb-server`): the `EmbeddingClient` trait with a
  default `ProxyEmbeddingClient` (deterministic hash-based pseudo-embeddings,
  128-dim, cached) and an `OpenAiEmbeddingClient` (any OpenAI-compatible
  `/embeddings` endpoint, with retry/backoff and a result cache). The default
  engine wires the **proxy**; the OpenAI client is available but not wired into
  the default constructor.

- **Plan cache** (`QueryCache`): TTL + max-entry bounded, tracks
  hits/misses/entries (`CacheStats`). The engine uses it to cache
  `(intent, plan)` by question and clears it on every write.

- **Cost tracker** (`CostTracker`): accumulates a USD estimate from a
  `CostModel` — embedding calls, index lookups (free), LLM input tokens, LLM
  output tokens. Surfaced as `total_cost_usd` in stats.

- **Agent memory** (`AgentMemory`): in-memory sessions holding conversation
  turns, working-memory facts, pinned objects, and metadata. Exposed in stats
  as `active_agent_sessions`, but **not reachable over the gRPC API** in v0.1.0.

- **Metrics** (`MetricsCollector`): request counters (ask/remember/insert/sql/
  errors) and cumulative ask latency, behind a `RwLock`.

## Persistence model

This is the most important architectural caveat, so it is restated here
explicitly.

| Component | Persisted across restart? |
| --- | --- |
| Object store (sled or Lance) | **Yes** |
| Full-text index (Tantivy) | **Yes** |
| HNSW vector index | No (in-memory) |
| Brute-force vector index | No (in-memory) |
| Metadata index | No (in-memory) |
| Time index | No (in-memory) |
| Graph index | No (in-memory) |
| Plan cache | No (ephemeral) |
| Agent memory | No (ephemeral) |

The in-memory indexes are populated **only** by `insert` at runtime. The engine
constructors (`new`, `new_with_lance`, `new_in_memory`) open storage and the
full-text index and create **empty** in-memory indexes — there is no startup
pass that re-reads the object store to rebuild them. Consequences and
mitigations are covered in [`OPERATIONS.md`](OPERATIONS.md).
