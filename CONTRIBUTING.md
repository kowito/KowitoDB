# Contributing to KowitoDB

Thanks for your interest! This guide gets you from a clean checkout to a passing
build in a couple of minutes.

## Prerequisites

- **Rust** (stable, 2021 edition) — install via [rustup](https://rustup.rs).
- **protoc** (Protocol Buffers compiler) — the server crate compiles `.proto`:
  - macOS: `brew install protobuf`
  - Debian/Ubuntu: `apt-get install -y protobuf-compiler`

## Build & run

```bash
# Build everything
cargo build --workspace

# Run the server (dev mode — no release build needed)
cargo run -p kowitodb -- serve

# Or query in embedded mode (no server)
cargo run -p kowitodb -- ask "what do you know?"
```

## Before you open a PR — run exactly what CI runs

CI (`.github/workflows/ci.yml`) gates on these three commands. Run them locally
and make sure they're green:

```bash
cargo fmt --all                                      # format (use --check in CI)
cargo clippy --workspace --all-targets -- -D warnings  # lint (warnings = errors)
cargo test --workspace                                # tests
```

A one-liner for the full gate:

```bash
cargo fmt --all --check && \
cargo clippy --workspace --all-targets -- -D warnings && \
cargo test --workspace
```

## Optional features

Some functionality is behind feature flags (heavy deps, model downloads):

```bash
cargo build -p kowitodb-server --features local-embeddings      # on-device Candle embeddings
cargo build -p kowitodb-server --features cross-encoder-rerank  # on-device reranker
cargo build -p kowitodb-server --features lance                 # Lance columnar storage
```

## Benchmarks

```bash
cargo run --release -p kowitodb-index --example bench_hnsw      # HNSW recall/latency/QPS
cargo run --release -p kowitodb-index --example bench_compare   # vs Qdrant/Milvus harness
```

See [`benchmarks/comparison/README.md`](benchmarks/comparison/README.md) for the
cross-engine comparison.

## Conventions

- Match the surrounding code's style; `cargo fmt` is the source of truth.
- Keep `clippy` clean (`-D warnings`).
- Add a test for behavioral changes; the index/server crates have good examples
  to copy.
- Configuration is via `KOWITODB_*` environment variables — see the
  "Configuration" section of the root [README](README.md).

## License

By contributing you agree your contributions are licensed under the project's
[MIT License](LICENSE).
