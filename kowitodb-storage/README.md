# kowitodb-storage

Pluggable persistence for **[KowitoDB](https://github.com/kowito/KowitoDB)** — the
database built for AI agents.

Defines the storage abstraction that the KowitoDB engine writes
[`KnowledgeObject`](https://crates.io/crates/kowitodb-core)s through, with
swappable backends.

## What's inside

- **`StorageEngine`** — the backend trait (get / put / scan / delete) the engine
  is written against, so storage is a swap, not a rewrite.
- **sled backend** — the pure-Rust embedded default; zero external services.
- **`LanceStorage`** — optional columnar [Lance](https://lancedb.github.io/lance/)
  backend for analytics-friendly, disk-efficient storage.
- **`schema`** — the on-disk record layout shared by the backends.

## Where it fits

```
kowitodb-core (types)
  └─ kowitodb-storage  ← you are here (persistence)
       └─ kowitodb-server (engine + gRPC)  →  kowitodb (CLI)
```

For the full feature tour, quickstart, and SDKs, see the
**[project README](https://github.com/kowito/KowitoDB#readme)**.

## License

[MIT](https://github.com/kowito/KowitoDB/blob/main/LICENSE)
