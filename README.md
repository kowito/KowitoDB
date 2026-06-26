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

> **Status:** v0.1.0, single-node, pre-1.0. The gRPC endpoint has **no
> authentication or TLS**, and several indexes are in-memory only. See
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
| Agent memory | In-memory conversation sessions, working memory, pinned objects | Implemented (not exposed over gRPC) |
| Embedding clients | Deterministic proxy (default) + OpenAI-compatible HTTP client | Implemented |
| Storage: sled | Default persistent embedded key/value store | Implemented |
| Storage: Lance | Optional columnar/Arrow dataset backend behind the `lance` feature | Implemented |
| Authentication / TLS | — | **Not implemented** |
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
indexes are **in-memory only** and are populated as objects are inserted — they
are not rebuilt from the object store on restart. See
[Known Limitations](#known-limitations) and
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

| Flag | Default | Meaning |
| --- | --- | --- |
| `--addr`, `-a` | `127.0.0.1:50051` | gRPC bind address |
| `--storage-path`, `-s` | `./data/storage` | sled object-store directory |
| `--index-path`, `-i` | `./data/index` | Tantivy full-text index directory |

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
> Because the non-full-text indexes are in-memory, an embedded `ask`/`sql`
> process only sees objects whose indexes it built in that same process. For
> cross-process querying, run `serve` and use a client. See
> [`docs/OPERATIONS.md`](docs/OPERATIONS.md).

## The gRPC API

The service is defined in [`proto/kowitodb.proto`](proto/kowitodb.proto)
(`package kowitodb`, service `KowitoDB`).

| RPC | Request → Response | Purpose |
| --- | --- | --- |
| `Insert` | `InsertRequest` → `InsertResponse` | Insert a knowledge object (content, embeddings, metadata, keywords, relationships, importance). Returns the new ID. |
| `Get` | `GetRequest` → `GetResponse` | Fetch a knowledge object by UUID. |
| `Delete` | `DeleteRequest` → `DeleteResponse` | Delete by ID; reports whether it existed. |
| `Search` | `SearchRequest` → `SearchResponse` | Direct search by `query` + `top_k`. Internally runs the same `ask` pipeline and returns `SearchResult`s plus a plan explanation. |
| `Ask` | `AskRequest` → `AskResponse` | High-level `ai.ask()`: returns `AskResult`s with `relevance_score` and `retrieval_source`, the `detected_intent`, and a `plan_explanation`. |
| `Remember` | `RememberRequest` → `RememberResponse` | High-level `ai.remember()`: store content with optional embeddings/metadata/keywords/importance. Returns the ID. |
| `Stats` | `StatsRequest` → `StatsResponse` | Returns `total_objects`, `vector_count`, and `index_size_bytes`. |

Key message shapes (see the proto for the full definitions):

```proto
message AskRequest  { string question = 1; int32 max_results = 2; int32 max_context_tokens = 3; }
message AskResult   { string id = 1; string content = 2; float relevance_score = 3;
                      map<string,string> metadata = 4; string retrieval_source = 5; }
message AskResponse { repeated AskResult results = 1; string plan_explanation = 2; string detected_intent = 3; }
```

Notes on current behavior:
- `AskRequest.max_results` is clamped to `[1, 100]` by the server.
- `AskRequest.max_context_tokens` is present in the proto but is **not yet
  applied** by the server; the context optimizer uses its built-in default
  (4096 tokens).
- `embeddings` are accepted on `Insert`/`Remember`; if none are supplied, the
  server auto-embeds the content using its configured embedding client.

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

> The gRPC service does not expose a dedicated SQL RPC. The SDK `sql()` helpers
> route their query string through the `Search` RPC, which runs the `ask`
> pipeline — not the SQL engine. Full SQL is available through the Rust engine
> API and the `kowitodb sql` CLI command.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for how the DataFusion
provider materializes batches from storage.

## Storage Backends

KowitoDB writes objects through the `StorageBackend` trait. Two implementations
ship:

| Backend | Feature flag | Persistence | Notes |
| --- | --- | --- | --- |
| **sled** (default) | none | Disk | Embedded key/value store with an in-process content cache. Used by every default constructor. |
| **Lance** | `lance` | Disk (Arrow/columnar) | A Lance dataset, upsert via delete-then-append keyed on `id`. Drop-in alternative; predicate pushdown is not yet implemented (scans materialize in memory). |

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
`get`, `search`, `stats` (Python/TS also have `sql`). Install instructions,
parallel "remember then ask" examples, and stub-regeneration steps are in
[`docs/SDKS.md`](docs/SDKS.md).

## Known Limitations

These are accurate to the current code and are documented in detail in
[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) and
[`docs/OPERATIONS.md`](docs/OPERATIONS.md):

- **No authentication and no TLS.** The gRPC server binds plaintext and accepts
  any caller. Restrict network access at the infrastructure layer.
- **Single-node only.** No replication, sharding, or clustering.
- **Most indexes are in-memory and not rebuilt from storage on restart.** Only
  the sled/Lance object store and the Tantivy full-text index persist. The HNSW
  vector, graph, metadata, and time indexes are populated by `insert` at runtime
  and are empty after a restart until objects are re-inserted. This is the most
  important operational caveat.
- **Default embeddings are a deterministic hash proxy**, not a real model. The
  OpenAI-compatible client exists but is not wired into the default server
  constructor. Token counts use a ~4-chars-per-token heuristic.
- **`max_context_tokens` is accepted but ignored.** No dedicated SQL RPC; SDK
  `sql()` routes through `Search`.
- **Agent memory is in-memory and not exposed over gRPC.**

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
