# kowitodb-core

Foundation types for **[KowitoDB](https://github.com/kowito/KowitoDB)** — the
database built for AI agents (vector + full-text + graph + SQL + agent memory
behind one `ai.ask()`).

This is the dependency-light base crate every other KowitoDB crate builds on. It
carries no index, storage, or server code — just the shared vocabulary.

## What's inside

- **`KnowledgeObject`** — the central record: content, embedding, keywords,
  metadata, relationships, importance, and timestamps.
- **`types`** — shared primitives (IDs, scores, query/result shapes).
- **`KowitoError` / `Result`** — the crate-wide error type used across the stack.

```rust
use kowitodb_core::knowledge::KnowledgeObject;

let obj = KnowledgeObject::new("Acme renewed their enterprise license");
```

## Where it fits

```
kowitodb-core  ← you are here (types)
  └─ kowitodb-storage / kowitodb-index / kowitodb-planner / kowitodb-sql
       └─ kowitodb-server (engine + gRPC)  →  kowitodb (CLI)
```

For the full feature tour, quickstart, and SDKs, see the
**[project README](https://github.com/kowito/KowitoDB#readme)**.

## License

[MIT](https://github.com/kowito/KowitoDB/blob/main/LICENSE)
