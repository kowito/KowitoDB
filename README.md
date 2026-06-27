# KowitoDB — AI Knowledge Operating System

An open-source Rust database that stores **knowledge objects** (content +
embeddings + metadata + keywords + graph relationships) and exposes a high-level
AI retrieval API — `ai.ask()` and `ai.remember()` — alongside vector, keyword,
graph, metadata, and time search, plus SQL. It serves over gRPC.

Instead of stitching together a vector database, a full-text engine, a graph
store, a cache, and a reranker in application code:

```python
# Conventional RAG stack:
embedding() -> vector.search() -> sql.query() -> graph.search() -> rerank() -> build_context() -> llm()

# With KowitoDB:
db.ask("Which enterprise customers renewed after Series A?")
```

KowitoDB detects the query's intent, builds an execution plan, runs the relevant
indexes, traverses the knowledge graph, reranks with Reciprocal Rank Fusion,
and assembles a token-budgeted context — behind a single call.

> **Status:** v0.1.0, single-node, pre-1.0. Optional API-key auth and TLS are
> available but **off by default**, and the secondary indexes are in-memory
> (rebuilt from storage on startup). See
> [Known Limitations](#known-limitations) before deploying. Read
> [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) and
> [`docs/OPERATIONS.md`](docs/OPERATIONS.md) first.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
  - [Build](#build)
  - [Run the server](#run-the-server)
  - [Use a client](#use-a-client)
  - [CLI (embedded mode)](#cli-embedded-mode)
- [The gRPC API](#the-grpc-api)
- [SQL](#sql)
- [Storage Backends](#storage-backends)
- [SDKs](#sdks)
- [Known Limitations](#known-limitations)
- [Documentation](#documentation)
- [License](#license)

---

## Features

| Capability | What it does | Status |
| --- | --- | --- |
| `ai.ask()` | Intent detection → planned multi-index retrieval → graph traversal → rerank → context assembly | Implemented |
| `ai.remember()` | Store content; auto-embed if no vector supplied; index across all subsystems | Implemented |
| Vector search | In-process HNSW approximate nearest neighbor (custom implementation) | Implemented |
| Keyword search | Tantivy full-text index (BM25), persisted to disk | Implemented |
| Graph traversal | Bidirectional, depth-scored traversal over named relationships | Implemented |
| Metadata search | Exact and substring matching over arbitrary key/value pairs | Implemented |
| Time search | Range queries (`before` / `after` / `between`) over creation timestamps | Implemented |
| Query planner | Rule-based intent classifier + rule engine that selects retrieval steps per query | Implemented |
| Reranking | Reciprocal Rank Fusion with per-source weights and cross-source agreement boosting | Implemented |
| Context optimizer | Jaccard dedup + token-budgeted assembly (default 4096 tokens) | Implemented |
| SQL (DataFusion) | Real `TableProvider` supporting projection, `WHERE`, `ORDER BY`, `GROUP BY`, aggregates, `LIMIT` | Implemented |
| SQL (index-routed) | Lightweight hand-rolled parser that maps a SQL subset to native indexes | Implemented |
| Plan cache | TTL + capacity-bounded cache of `(intent, plan)` keyed by question | Implemented |
| Cost tracker | Estimates embedding + LLM-token + index cost in USD | Implemented |
| Update + versioning | In-place edit (re-embeds on content change) with version history persisted to storage | Implemented |
| Agent memory | Conversation sessions, working memory, pinned objects; `RecordTurn`/`GetSession` over gRPC; persisted to a sled store under `{index_path}/sessions` | Implemented |
| Bulk + pagination | `BatchInsert` for bulk ingestion; `List` with `offset`/`limit` + total count | Implemented |
| Filtered retrieval | Exact-match `metadata_filter` on `Ask`/`Search`, applied via the metadata index | Implemented |
| Embedding clients | Deterministic proxy (default) + OpenAI-compatible HTTP client, selected via env | Implemented |
| Index rebuild on restart | `open()` re-reads the object store and repopulates the in-memory vector/metadata/time/graph indexes | Implemented |
| Storage: sled | Default persistent embedded key/value store | Implemented |
| Storage: Lance | Optional columnar/Arrow dataset backend behind the `lance` feature | Implemented |
| Authentication (API key) | Optional Bearer / `x-api-key` interceptor via `--api-key` | Implemented (off by default) |
| TLS | Optional via `--tls-cert` / `--tls-key` | Implemented (off by default) |
| Health-check + reflection | gRPC health service + reflection, always on (unauthenticated) | Implemented |
| Prometheus metrics | `/metrics` + `/healthz` HTTP endpoint via `--metrics-addr` | Implemented |
| Distributed / multi-node | — | **Not implemented** |

## Architecture

```
            Python / TypeScript / Go SDKs        kowitodb CLI
                        |                              |
                     gRPC (tonic)                 embedded engine
                        |                              |
              +------------------------------------------------+
              |               KowitoDBEngine                   |
              |                                                |
              |   QueryPlanner: IntentAnalyzer + RuleEngine    |
              |        |                                       |
              |   ExecutionPlan ---> step-by-step execution    |
              |        |                                       |
              |  +-----+------+------+----------+-----------+   |
              |  |     |      |      |          |           |   |
              | HNSW  Full-  Meta-  Time      Graph     (Vector |
              | (vec) text   data  index    (traverse)  brute-  |
              |       (Tan-  index                       force) |
              |       tivy)                                     |
              |        |      |      |          |               |
              |     Merge -> Rerank (RRF) -> ContextOptimizer   |
              |                                                |
              |   CostTracker | QueryCache | AgentMemory        |
              |   EmbeddingClient (proxy | OpenAI-compatible)   |
              +------------------------------------------------+
                        |                              |
              StorageBackend trait          DataFusion SqlContext
                  /            \              (KnowledgeTableProvider)
            sled (default)   Lance (--features lance)
            [persistent]      [persistent, Arrow/columnar]
```

Persistence note: the **sled (or Lance) object store** and the **Tantivy
full-text index** persist to disk. The HNSW vector, graph, metadata, and time
indexes are **in-memory**, but they are **rebuilt from the object store on
startup**: `KowitoDBEngine::open()` runs a reindex pass that re-reads every
stored object (using its persisted embedding — no embedding API calls) and
repopulates the in-memory indexes. Cost is O(stored objects) at startup. See
[`docs/OPERATIONS.md`](docs/OPERATIONS.md).

A crate-by-crate breakdown and the full retrieval pipeline are in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Quick Start

### Build

Prerequisites: a recent stable Rust toolchain (edition 2021) and `cargo`.

```bash
git clone <repo-url> kowitodb
cd kowitodb
cargo build --release
```

The binary is produced at `target/release/kowitodb`.

To build with the optional Lance storage backend:

```bash
cargo build --release -p kowitodb-server --features lance
```

> The `lance` feature is wired through `kowitodb-server`. The default
> `kowitodb` CLI binary always uses the sled backend; see
> [Storage Backends](#storage-backends).

### Run the server

```bash
./target/release/kowitodb serve \
  --addr 127.0.0.1:50051 \
  --storage-path ./data/storage \
  --index-path ./data/index
```

Defaults (all flags optional):

| Flag | Env | Default | Meaning |
| --- | --- | --- | --- |
| `--addr`, `-a` | — | `127.0.0.1:50051` | gRPC bind address |
| `--storage-path`, `-s` | — | `./data/storage` | sled object-store directory |
| `--index-path`, `-i` | — | `./data/index` | Tantivy full-text index directory |
| `--api-key` | `KOWITODB_API_KEY` | _(unset)_ | Require this key on every call (`Bearer` or `x-api-key`). Off when unset. |
| `--tls-cert` | `KOWITODB_TLS_CERT` | _(unset)_ | PEM TLS certificate chain (enables TLS with `--tls-key`). |
| `--tls-key` | `KOWITODB_TLS_KEY` | _(unset)_ | PEM TLS private key. |
| `--metrics-addr` | `KOWITODB_METRICS_ADDR` | _(unset)_ | Expose Prometheus `/metrics` + `/healthz` HTTP on this address (e.g. `0.0.0.0:9090`). |

The gRPC health-checking service and reflection are always on (so `grpcurl` and
liveness probes work), and are intentionally unauthenticated.

**Embeddings** are configured via environment. When `KOWITODB_EMBEDDING_PROVIDER`
is unset the server uses a deterministic dev proxy; set it to use a real model:

| Variable | Meaning |
| --- | --- |
| `KOWITODB_EMBEDDING_PROVIDER` | `openai` or `ollama` (anything else / unset → proxy). |
| `OPENAI_API_KEY` / `KOWITODB_OPENAI_API_KEY` | API key for the `openai` provider. |
| `KOWITODB_OPENAI_BASE_URL` | Override the OpenAI-compatible base URL (default `https://api.openai.com/v1`). |
| `KOWITODB_EMBEDDING_MODEL` | Model name (default `text-embedding-3-small`, or `nomic-embed-text` for Ollama). |
| `KOWITODB_OLLAMA_URL` | Ollama base URL (default `http://localhost:11434/v1`). |

Logging is controlled by `RUST_LOG` (via `tracing-subscriber`'s `EnvFilter`);
it defaults to `info`:

```bash
RUST_LOG=kowitodb=debug,info ./target/release/kowitodb serve
```

### Use a client

```python
from kowitodb import KowitoDBClient

db = KowitoDBClient("localhost:50051")

db.remember(
    "Acme Corp renewed their enterprise license in March 2024 after Series A funding of $15M.",
    keywords=["acme", "renewal", "series a", "enterprise"],
    metadata={"company": "Acme Corp", "stage": "series_a"},
    importance=0.9,
)

resp = db.ask("Which enterprise customers renewed after Series A?")
print("Intent:", resp.detected_intent)
print(resp.plan_explanation)
for r in resp.results:
    print(f"[{r.relevance_score:.2f}] ({r.retrieval_source}) {r.content}")
```

Parallel examples for TypeScript and Go are in [`docs/SDKS.md`](docs/SDKS.md).

### CLI (embedded mode)

The `kowitodb` binary also runs the engine in-process — no server required.
All embedded commands take the same `--storage-path` / `--index-path` flags as
`serve`, and operate on the same data directory.

```bash
# Insert
kowitodb insert "OpenAI raised $6.6B in 2024" \
  --keywords openai,funding --metadata company=OpenAI --importance 0.8

# Ask
kowitodb ask "Which companies raised funding?" --max-results 5

# SQL (index-routed path)
kowitodb sql "SELECT content FROM knowledge WHERE metadata.company = 'OpenAI'"

# Stats
kowitodb stats
```

> Embedded commands open their own engine instance against the data directory.
> `ask`, `sql`, and `stats` use `open()`, which rebuilds the in-memory indexes
> from the persisted object store first, so they see everything previously
> written to that directory. (sled holds an exclusive lock, so only one process
> may open a given directory at a time — for concurrent access, run `serve` and
> use a client.) See [`docs/OPERATIONS.md`](docs/OPERATIONS.md).

## The gRPC API

The service is defined in [`proto/kowitodb.proto`](proto/kowitodb.proto)
(`package kowitodb`, service `KowitoDB`).

| RPC | Request → Response | Purpose |
| --- | --- | --- |
| `Insert` | `InsertRequest` → `InsertResponse` | Insert a knowledge object (content, embeddings, metadata, keywords, relationships, importance). Returns the new ID. |
| `BatchInsert` | `BatchInsertRequest` → `BatchInsertResponse` | Insert many objects in one call (bulk ingestion). Returns the new IDs in order. |
| `Get` | `GetRequest` → `GetResponse` | Fetch a knowledge object by UUID. |
| `Update` | `UpdateRequest` → `UpdateResponse` | In-place edit by ID (content/metadata/keywords/importance). Records a version-history entry and re-embeds on content change. Returns `updated` and the new `version` count. |
| `Delete` | `DeleteRequest` → `DeleteResponse` | Delete by ID; reports whether it existed. |
| `List` | `ListRequest` → `ListResponse` | Enumerate stored objects with `offset`/`limit` pagination. Returns the page plus the `total` count. |
| `Search` | `SearchRequest` → `SearchResponse` | Direct search by `query` + `top_k`, optionally constrained by an exact-match `metadata_filter`. Returns `SearchResult`s plus a plan explanation. |
| `Ask` | `AskRequest` → `AskResponse` | High-level `ai.ask()`: returns `AskResult`s with `relevance_score` and `retrieval_source`, the `detected_intent`, and a `plan_explanation`. Accepts an optional exact-match `metadata_filter`. |
| `Remember` | `RememberRequest` → `RememberResponse` | High-level `ai.remember()`: store content with optional embeddings/metadata/keywords/importance. Returns the ID. |
| `Sql` | `SqlRequest` → `SqlResponse` | Run a SQL query through the DataFusion engine (projection/`WHERE`/`ORDER BY`/`GROUP BY`/aggregates/`LIMIT`). Returns rows as ordered column-name → value maps. |
| `RecordTurn` | `RecordTurnRequest` → `RecordTurnResponse` | Append a turn (`user`/`assistant`/`system`/`observation`) to an agent session. Returns the new turn count. |
| `GetSession` | `GetSessionRequest` → `GetSessionResponse` | Fetch the conversation turns for an agent session. |
| `Stats` | `StatsRequest` → `StatsResponse` | Returns `total_objects`, `vector_count`, `index_size_bytes`, `graph_nodes`, `graph_edges`, `active_agent_sessions`, `total_cost_usd`, `cache_entries`, and `cache_hit_rate`. |

Key message shapes (see the proto for the full definitions):

```proto
message AskRequest  { string question = 1; int32 max_results = 2; int32 max_context_tokens = 3; }
message AskResult   { string id = 1; string content = 2; float relevance_score = 3;
                      map<string,string> metadata = 4; string retrieval_source = 5; }
message AskResponse { repeated AskResult results = 1; string plan_explanation = 2; string detected_intent = 3; }
```

Notes on current behavior:
- `AskRequest.max_results` is clamped to `[1, 100]` by the server.
- `AskRequest.max_context_tokens`, when greater than 0, is honored as the
  context-token budget for that request; otherwise the optimizer's default
  (4096 tokens) applies.
- `embeddings` are accepted on `Insert`/`Remember`; if none are supplied, the
  server auto-embeds the content using its configured embedding client, and the
  generated embedding is persisted (so it survives the restart reindex).

## SQL

KowitoDB has two SQL paths:

1. **DataFusion path** (`KowitoDBEngine::sql_select`) — a real Apache DataFusion
   `TableProvider` (`KnowledgeTableProvider`) exposes stored objects as a table
   named `knowledge` (aliased `objects`) with columns `id`, `content`,
   `importance`, `created_at`, `updated_at`, `keywords`, `metadata` (the last
   two as JSON strings). It supports projection, `WHERE`, `ORDER BY`,
   `GROUP BY`, aggregates, and `LIMIT`. Rows come back as
   `Vec<HashMap<String, String>>` (every value stringified).

   ```sql
   SELECT COUNT(*) AS n FROM knowledge;
   SELECT content, importance FROM knowledge
     WHERE importance >= 0.5 ORDER BY importance DESC;
   SELECT id FROM knowledge WHERE metadata LIKE '%series_a%';
   ```

2. **Index-routed path** (`KowitoDBEngine::sql_query`, used by the `kowitodb sql`
   CLI command) — a lightweight hand-rolled parser that maps a small SQL subset
   (`metadata.key = ...`, `... LIKE '%...%'`, `created_at`, `keyword`, `content`,
   `LIMIT`) directly to the native metadata/full-text/time indexes and returns
   loaded objects.

   ```sql
   SELECT * FROM knowledge WHERE metadata.stage = 'series_a';
   SELECT content FROM knowledge WHERE keyword LIKE '%enterprise%' LIMIT 10;
   ```

> The gRPC service exposes a dedicated `Sql` RPC backed by the DataFusion path
> (`sql_select`), and the Python/TypeScript/Go SDK `sql()` helpers call it,
> returning rows as column maps. The `kowitodb sql` CLI command uses the lighter
> index-routed path.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for how the DataFusion
provider materializes batches from storage.

## Storage Backends

KowitoDB writes objects through the `StorageBackend` trait. Two implementations
ship:

| Backend | Feature flag | Persistence | Notes |
| --- | --- | --- | --- |
| **sled** (default) | none | Disk | Embedded key/value store with an in-process content cache. Used by every default constructor. |
| **Lance** | `lance` | Disk (Arrow/columnar) | A Lance dataset, upsert via delete-then-append keyed on `id`. Drop-in alternative; `id` and `min_importance` predicates are pushed into the native Lance scan, other predicates are filtered in Rust. |

The server can run on Lance via `KowitoDBEngine::new_with_lance(uri, index_path)`
when `kowitodb-server` is built with `--features lance`. The default `kowitodb`
CLI binary uses sled. See [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).

## SDKs

| Language | Package / Module | Transport |
| --- | --- | --- |
| Python | `kowitodb` (`sdk/python`) | `grpcio` |
| TypeScript | `@kowitodb/sdk` (`sdk/typescript`) | `@grpc/grpc-js` + `@grpc/proto-loader` |
| Go | `github.com/kowito/kowitodb/sdk/go` | `google.golang.org/grpc` |

All three expose the same surface: `remember`, `ask`, `forget`, `insert`,
`get`, `update`, `search`, `sql`, `record_turn`/`recordTurn`,
`get_session`/`getSession`, and `stats` (which now carries the full field set).
`sql()` calls the dedicated `Sql` RPC (DataFusion). Install instructions,
parallel "remember then ask" examples, and stub-regeneration steps are in
[`docs/SDKS.md`](docs/SDKS.md).

## Known Limitations

These are accurate to the current code and are documented in detail in
[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) and
[`docs/OPERATIONS.md`](docs/OPERATIONS.md):

- **Auth and TLS are off by default.** API-key auth (`--api-key`) and TLS
  (`--tls-cert`/`--tls-key`) are available but disabled unless configured; when
  off, the gRPC server binds plaintext and accepts any caller. The health-check
  and reflection services are always on and intentionally unauthenticated, so
  restrict network access at the infrastructure layer regardless.
- **Single-node only.** No replication, sharding, or clustering.
- **Secondary indexes are in-memory** (HNSW vector, graph, metadata, time).
  They are **rebuilt from the object store on startup** via `open()`, at a cost
  of O(stored objects); they are not separately persisted. The working set must
  fit in RAM.
- **Default embeddings are a deterministic hash proxy** unless you set
  `KOWITODB_EMBEDDING_PROVIDER`. A real OpenAI-compatible / Ollama client is
  wired in and selected by env. Token counts use a ~4-chars-per-token heuristic.
- **Agent conversation memory is in-memory and not persisted** — it is exposed
  over gRPC (`RecordTurn`/`GetSession`) and counted in `Stats`, but is lost on
  restart.
- **Keyword and time predicates still scan storage**; `id` and `importance`
  filters are pushed down. `Search`/`Ask` results cap at 100.

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — crates, the retrieval
  pipeline, the six indexes, the storage abstraction, the DataFusion SQL path.
- [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) — release builds, configuration,
  Dockerfile, sizing, persistence, observability.
- [`docs/OPERATIONS.md`](docs/OPERATIONS.md) — backup/restore, upgrades,
  metrics, plan cache, scaling boundaries.
- [`docs/SDKS.md`](docs/SDKS.md) — Python / TypeScript / Go usage and codegen.

## License

MIT.
