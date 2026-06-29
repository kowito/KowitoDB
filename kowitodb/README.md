# kowitodb

**The database built for AI agents.** Vector search, full-text (BM25), a knowledge
graph, metadata/time indexes, agent memory, and real SQL — in one engine, behind
a single `ai.ask()`. No glue between six systems.

This crate is the **CLI + gRPC server**. For library use, see
[`kowitodb-server`](https://crates.io/crates/kowitodb-server) (the `KowitoDBEngine`).

## Install

```bash
cargo install kowitodb
```

## See it work in 2 seconds

```bash
kowitodb demo          # seeds in-memory data, runs ai.ask() + SQL — no setup
```

## Run it

```bash
kowitodb serve --addr 127.0.0.1:50051 \
  --storage-path ./data/storage --index-path ./data/index

# ...or use it directly, no SDK (embedded mode):
kowitodb insert "Acme renewed their enterprise license" -k acme,renewal
kowitodb ask    "who renewed?"
kowitodb sql    "SELECT content FROM knowledge WHERE importance >= 0.8"
```

Set `--api-key` (or `KOWITODB_API_KEY`) to require auth; `--metrics-addr` exposes
Prometheus `/metrics` + `/healthz`. Scale out with `kowitodb gateway`.

## Learn more

Full feature tour, configuration, SDKs (Python/TypeScript/Go), benchmarks, and an
honest comparison vs Qdrant/Milvus are in the
**[project README](https://github.com/kowito/KowitoDB#readme)**.

## License

[MIT](https://github.com/kowito/KowitoDB/blob/main/LICENSE)
