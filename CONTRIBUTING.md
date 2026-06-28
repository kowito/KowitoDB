# Contributing to KowitoDB

Thanks for your interest! This guide gets you from a clean checkout to a passing
build in a couple of minutes.

## Prerequisites

- **Rust** (stable, 2021 edition) — install via [rustup](https://rustup.rs).
- **protoc** (Protocol Buffers compiler) — the server crate compiles `.proto`:
  - macOS: `brew install protobuf`
  - Debian/Ubuntu: `apt-get install -y protobuf-compiler`

## Build & run

A `Makefile` wraps the common tasks — run `make` (or `make help`) for the full
list. The essentials:

```bash
make build        # build the workspace
make run          # run the gRPC server (dev mode, no release build)
make example      # run the embedded-library example (no server)
make ci           # the exact CI gate: fmt-check + clippy + test
```

The raw cargo equivalents (if you prefer not to use `make`):

```bash
cargo build --workspace
cargo run -p kowitodb -- serve
cargo run -p kowitodb -- ask "what do you know?"
cargo run -p kowitodb-server --example embedded
```

## Before you open a PR — run exactly what CI runs

CI (`.github/workflows/ci.yml`) gates on fmt + clippy + test. Run the same gate
locally with one command:

```bash
make ci
```

…which is equivalent to:

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
