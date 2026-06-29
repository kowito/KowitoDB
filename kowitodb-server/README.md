# kowitodb-server

The engine and gRPC server behind **[KowitoDB](https://github.com/kowito/KowitoDB)**
— the database built for AI agents. This is the crate to depend on for **embedded
(in-process) use**; the [`kowitodb`](https://crates.io/crates/kowitodb) crate wraps
it as a CLI + standalone server.

It ties the whole stack together: storage + the six indexes + the AI planner +
SQL + agent memory + embeddings, behind one `ai.ask()` / `ai.remember()` surface.

## Key types

- **`KowitoDBEngine`** — the embeddable engine: insert knowledge, `ask()`, run
  SQL, build GraphRAG community summaries (`CommunitySummary`).
- **`KowitoDBService`** — the Tonic gRPC service (proto in `kowitodb_server::proto`).
- **`AgentMemory`** — persisted conversation memory (`AgentSession`,
  `ConversationTurn`, `TurnRole`).
- **Embeddings** — on-device (`LocalEmbeddingClient`) or remote
  (`OpenAiEmbeddingClient`, `ProxyEmbeddingClient`) via the `EmbeddingClient` trait.
- **LLM** — `LlmClient` / `OpenAiLlmClient` for answer synthesis.
- **`Cluster` / `ClusterService`** — distributed gateway mode.
- **`ServerConfig`**, **`MetricsCollector`** (Prometheus).

```rust
use kowitodb_core::KnowledgeObject;
use kowitodb_server::KowitoDBEngine;

let engine = KowitoDBEngine::new_in_memory()?;           // or ::new(storage, index)
engine.insert(KnowledgeObject::new("Acme renewed their enterprise license")).await?;
let answer = engine.ask("who renewed?", 5).await?;       // (question, max_results)
```

## Where it fits

```
kowitodb-core / -storage / -index / -planner / -sql
  └─ kowitodb-server  ← you are here (engine + gRPC)
       └─ kowitodb (CLI + standalone server)
```

For the full feature tour, quickstart, SDKs (Python/TS/Go), and benchmarks, see
the **[project README](https://github.com/kowito/KowitoDB#readme)**.

## License

[MIT](https://github.com/kowito/KowitoDB/blob/main/LICENSE)
