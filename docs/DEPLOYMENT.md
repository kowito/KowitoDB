# KowitoDB Deployment

Production deployment guidance for the KowitoDB gRPC server. Read the
[security and persistence caveats](#security-posture) before exposing it
anywhere.

## Contents

- [Build a release binary](#build-a-release-binary)
- [Run the server](#run-the-server)
- [Configuration](#configuration)
- [Storage backend selection](#storage-backend-selection)
- [Dockerfile](#dockerfile)
- [Continuous integration](#continuous-integration)
- [Resource and sizing guidance](#resource-and-sizing-guidance)
- [Persistence and data directory](#persistence-and-data-directory)
- [Observability](#observability)
- [Security posture](#security-posture)

## Build a release binary

```bash
cargo build --release
# -> target/release/kowitodb
```

The workspace defines a tuned `[profile.release]` in the root `Cargo.toml`, so
release builds are optimized for production out of the box:

```toml
# Cargo.toml (workspace root)
[profile.release]
opt-level = 3       # full optimization
lto = "thin"        # cross-crate inlining at reasonable link time
codegen-units = 1   # better codegen (slower compile, faster binary)
strip = "symbols"   # smaller binary, no debug symbols
panic = "unwind"    # keep unwinding so the server survives per-request panics
incremental = false
```

`lto = "thin"` plus `codegen-units = 1` trade longer compile time for a
faster, smaller binary — expect the optimized release build (which compiles
Arrow/DataFusion) to take several minutes from clean.

To build the server with the optional Lance backend:

```bash
cargo build --release -p kowitodb-server --features lance
```

The `lance` feature is declared on `kowitodb-server` (and forwarded to
`kowitodb-storage`). The default `kowitodb` CLI binary does **not** enable it
and always uses sled; to serve over Lance you build/run the server crate with
the feature and use `KowitoDBEngine::new_with_lance`. See
[Storage backend selection](#storage-backend-selection).

## Run the server

```bash
./target/release/kowitodb serve \
  --addr 0.0.0.0:50051 \
  --storage-path /var/lib/kowitodb/storage \
  --index-path /var/lib/kowitodb/index
```

The `serve` command creates the storage and index directories if they do not
exist, opens the engine (rebuilding the in-memory indexes from storage — see
[Persistence and data directory](#persistence-and-data-directory)), and serves
the `KowitoDB` gRPC service on `--addr`.

A hardened invocation with auth, TLS, and a Prometheus endpoint:

```bash
KOWITODB_EMBEDDING_PROVIDER=openai OPENAI_API_KEY=sk-... \
./target/release/kowitodb serve \
  --addr 0.0.0.0:50051 \
  --storage-path /var/lib/kowitodb/storage \
  --index-path /var/lib/kowitodb/index \
  --api-key "$KOWITODB_API_KEY" \
  --tls-cert /etc/kowitodb/tls/cert.pem \
  --tls-key /etc/kowitodb/tls/key.pem \
  --metrics-addr 0.0.0.0:9090
```

## Configuration

All configuration is via CLI flags (each of which falls back to an environment
variable) plus a few embedding/logging env vars. There is no config file.

### CLI flags (`serve`)

| Flag | Short | Env | Default | Description |
| --- | --- | --- | --- | --- |
| `--addr` | `-a` | — | `127.0.0.1:50051` | Socket address to bind the gRPC server. Use `0.0.0.0:50051` to accept remote connections. |
| `--storage-path` | `-s` | — | `./data/storage` | Directory for the sled object store. |
| `--index-path` | `-i` | — | `./data/index` | Directory for the Tantivy full-text index (created under `{index-path}/tantivy/`). |
| `--api-key` | — | `KOWITODB_API_KEY` | _(unset)_ | Require this key on every gRPC call, presented as `authorization: Bearer <key>` or `x-api-key: <key>`. Auth is off when unset. |
| `--tls-cert` | — | `KOWITODB_TLS_CERT` | _(unset)_ | Path to a PEM TLS certificate chain. Enables TLS together with `--tls-key`. |
| `--tls-key` | — | `KOWITODB_TLS_KEY` | _(unset)_ | Path to the PEM TLS private key. |
| `--metrics-addr` | — | `KOWITODB_METRICS_ADDR` | _(unset)_ | Bind an HTTP server exposing Prometheus `/metrics` and `/healthz` (e.g. `0.0.0.0:9090`). |

The default bind address is loopback-only (`127.0.0.1`). Change it deliberately,
and see [Security posture](#security-posture) first. The gRPC health-checking
service and reflection are always enabled (and unauthenticated) regardless of
these flags.

### Environment variables

| Variable | Effect |
| --- | --- |
| `RUST_LOG` | Sets the `tracing-subscriber` `EnvFilter`. Defaults to `info` when unset. Examples: `RUST_LOG=info`, `RUST_LOG=kowitodb=debug,warn`. |
| `KOWITODB_API_KEY`, `KOWITODB_TLS_CERT`, `KOWITODB_TLS_KEY`, `KOWITODB_METRICS_ADDR` | Fallbacks for the corresponding `serve` flags above. |
| `KOWITODB_EMBEDDING_PROVIDER` | Selects the embedding provider: `openai`, `ollama`, or (unset/other) the deterministic dev proxy. |
| `OPENAI_API_KEY` / `KOWITODB_OPENAI_API_KEY` | API key for the `openai` provider. |
| `KOWITODB_OPENAI_BASE_URL` | OpenAI-compatible base URL (default `https://api.openai.com/v1`). |
| `KOWITODB_EMBEDDING_MODEL` | Embedding model name (default `text-embedding-3-small`; `nomic-embed-text` for Ollama). |
| `KOWITODB_OLLAMA_URL` | Ollama base URL (default `http://localhost:11434/v1`). |

There is no env var for the bind port (use `--addr`) and no env var that switches
the storage backend (that is a build-time feature plus a code-level constructor
choice — see below).

### Embedding provider

Set `KOWITODB_EMBEDDING_PROVIDER` to use a real embedder; otherwise the server
runs the deterministic hash proxy (fine for development, not for semantic search
quality). For OpenAI:

```bash
export KOWITODB_EMBEDDING_PROVIDER=openai
export OPENAI_API_KEY=sk-...
# optional: KOWITODB_EMBEDDING_MODEL=text-embedding-3-small
```

For a local Ollama:

```bash
export KOWITODB_EMBEDDING_PROVIDER=ollama
# optional: KOWITODB_OLLAMA_URL=http://localhost:11434/v1
# optional: KOWITODB_EMBEDDING_MODEL=nomic-embed-text
```

The provider is selected once at engine startup (`OpenAiConfig::from_env`).
Embeddings are persisted on insert, so the choice only affects newly embedded
content — re-embed existing objects (via `update`) if you switch models.

## Storage backend selection

| Backend | How to select | Persistence |
| --- | --- | --- |
| sled (default) | Default `serve` / CLI; nothing to do. | Disk |
| Lance | Build `kowitodb-server` with `--features lance` and call `KowitoDBEngine::new_with_lance(uri, index_path)` from a server entry point. | Disk (Arrow/columnar) |

The shipped `kowitodb serve` binary constructs the engine with
`KowitoDBEngine::new(...)` (sled). There is no CLI flag to switch to Lance in
v0.1.0; selecting Lance requires a small server binary that calls
`new_with_lance` and is built with the `lance` feature. The Lance `uri` may be a
local path or any URI Lance supports.

## Dockerfile

The repository root ships a real multi-stage [`Dockerfile`](../Dockerfile). It:

- builds the tuned release binary in a `rust:1-bookworm` stage, installing
  `protobuf-compiler` (`protoc`) — which `tonic-build` requires to compile the
  gRPC definitions;
- ships the binary on `debian:bookworm-slim` with `ca-certificates`;
- mounts a `/data` volume, exposes **50051** (gRPC) and **9090**
  (Prometheus `/metrics` + `/healthz`), and its default `CMD` runs
  `serve` with `--metrics-addr 0.0.0.0:9090` against `/data/storage` and
  `/data/index`.

Build and run:

```bash
docker build -t kowitodb:0.1.0 .
docker run --rm \
  -p 50051:50051 -p 9090:9090 \
  -v kowitodb-data:/data \
  kowitodb:0.1.0
```

Override the default `CMD` to add `--api-key`, `--tls-cert`/`--tls-key`, or
embedding env vars (pass the latter with `-e`/`--env-file`). Because auth and
TLS are off by default, do this before exposing the container beyond a trusted
network — see [Security posture](#security-posture).

For the Lance backend you would need a server entry point that calls
`new_with_lance` and a build with `--features lance`; the stock `kowitodb serve`
binary does not select Lance.

## Continuous integration

[`.github/workflows/ci.yml`](../.github/workflows/ci.yml) runs on pushes to
`main` and on pull requests:

- **Rust:** `cargo fmt --all --check`, `cargo clippy --workspace --all-targets
  -- -D warnings` (warnings are errors), and `cargo test --workspace`, with
  `protoc` installed and the build cache warmed.
- **SDKs:** builds and vets the Go SDK (`go build ./... && go vet ./...`) and
  type-checks the TypeScript SDK (`npx tsc --noEmit`).

## Resource and sizing guidance

KowitoDB is a single process. Plan capacity around two facts:

1. **The full object set is held in RAM across the in-memory indexes**, and the
   sled content cache holds object content. The HNSW graph, the metadata maps,
   the time `BTreeMap`, and the graph adjacency maps all scale with the number
   of objects and their embedding dimension.
2. **CPU**: `ask` is single-request CPU-bound on HNSW search, RRF reranking, and
   context dedup. The runtime is Tokio (`features = ["full"]`), so concurrent
   requests are served on the async runtime, but heavy per-request work is not
   parallelized across cores beyond what the index locks allow.

Rough guidance (validate against your own corpus):

| Dimension | Guidance |
| --- | --- |
| Memory | Budget for: (embedding_dim × 4 bytes × object_count) for vectors, plus HNSW graph overhead (~`m` neighbor links/node), plus metadata/graph/time maps, plus the sled content cache. Size the host to hold the full working set with headroom. |
| Disk | sled object store + Tantivy index both grow with corpus size and are persisted. Provision generously; sled does not aggressively reclaim space. |
| CPU | More cores help throughput under concurrent `ask` load and speed up the release build; a single `ask` is largely serial. |
| Embedding latency | With the default proxy embedder, embedding is local and cheap. With the OpenAI-compatible client, each uncached embed is a network round-trip (with retry/backoff) — provision for that latency and rate limits. |

Because everything is in one process and most indexes are in-memory, the
practical ceiling is "fits comfortably in one machine's RAM." See
[OPERATIONS.md](OPERATIONS.md) for scaling boundaries.

## Persistence and data directory

Two directories matter, both under the paths you pass to `serve`:

```
{storage-path}/          sled object store (persistent)
{index-path}/tantivy/    Tantivy full-text index (persistent)
```

What persists and what does not:

- **Persistent on disk:** the object store (sled, or a Lance dataset if used) —
  including embeddings and version history — and the Tantivy full-text index.
- **In-memory, rebuilt from storage on startup:** the HNSW vector index, the
  metadata index, the time index, and the graph index. `serve` (and the
  `ask`/`sql`/`stats` CLI commands) call `KowitoDBEngine::open()`, which runs a
  reindex pass over the persisted object store before serving — so all search
  modes work immediately after a restart, with no re-ingestion required.
- **Not persistent and not rebuilt:** the plan cache and agent memory (both
  ephemeral), and the brute-force vector index (not on the live `ask` path).

The reindex pass uses the persisted embeddings — it makes **no** embedding API
calls — and skips the already-persisted full-text index. Its cost is
O(stored objects) and is paid once at startup, so plan for a slightly longer
warm-up on large corpora. See
[OPERATIONS.md → Index persistence and restarts](OPERATIONS.md#index-persistence-and-restarts).

Back up both directories together; do not snapshot one without the other. Backup
and restore procedures are in [OPERATIONS.md](OPERATIONS.md).

## Observability

- **Logging / tracing.** The binary initializes `tracing-subscriber` with an
  `EnvFilter` from `RUST_LOG` (default `info`). The server, engine, indexes, and
  embedding clients emit `tracing` spans/events (insert, ask, delete, cache
  hits, OpenAI calls, etc.). The `tracing-subscriber` dependency includes the
  `json` feature, so structured JSON logging can be enabled in code if desired;
  the default binary uses the human-readable formatter.
- **Metrics.** `MetricsCollector` tracks ask/remember/insert/sql/error counts,
  cumulative and average ask latency, and uptime. When `--metrics-addr` is set,
  they are exposed in Prometheus text format at `GET /metrics` on that address.
  The `Stats` RPC additionally reports object/vector/graph counts, cache stats,
  active agent sessions, and the estimated cost.
- **Health checks.** Use any of:
  - `GET /healthz` on the metrics address (returns `ok`) when `--metrics-addr`
    is set;
  - the always-on **gRPC health-checking service** (`grpc.health.v1.Health`) —
    e.g. `grpc_health_probe -addr=host:50051`;
  - the always-on **gRPC reflection** service for tooling like `grpcurl`.
  Both gRPC services are unauthenticated, so probes work without the API key.

See [OPERATIONS.md](OPERATIONS.md) for interpreting `Stats` and the metrics.

## Security posture

Read this before binding to anything other than loopback.

- **Auth is off by default.** Set `--api-key` (env `KOWITODB_API_KEY`) to require
  a Bearer / `x-api-key` token on every gRPC call. When unset, the server accepts
  any client that can reach the port.
- **TLS is off by default.** Set `--tls-cert` and `--tls-key` to terminate TLS in
  the server itself. When unset, the server speaks plaintext gRPC and SDK clients
  connect with insecure channels.
- **The health-check and reflection gRPC services are always on and
  unauthenticated by design**, so liveness probes and tooling work without
  credentials. They expose service metadata (reflection) but not your data; keep
  the endpoint off the public internet regardless.
- **Default bind is loopback** (`127.0.0.1:50051`). Keep it that way unless you
  have a network boundary you trust.

Recommended hardening:

1. Turn on `--api-key` and TLS for any non-loopback deployment; rotate the key
   out of band.
2. Keep KowitoDB on a private network / inside the cluster; never expose the
   port to the public internet. A reverse proxy or service mesh (Envoy/Linkerd)
   can add mTLS and richer authz in front if you need more than a static key.
3. Restrict ingress with security groups / network policies to known clients.
4. Run the process as an unprivileged user with write access only to the data
   directory, and keep TLS key files readable only by that user.
