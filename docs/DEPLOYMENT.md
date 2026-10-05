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

To build the binary with the optional Lance backend available:

```bash
cargo build --release -p kowitodb --features lance
# -> target/release/kowitodb, then: kowitodb serve --storage lance
```

The `lance` feature on the `kowitodb` binary crate forwards to
`kowitodb-server/lance` (and on to `kowitodb-storage`). Building
`-p kowitodb-server --features lance` only builds the library, not the binary.
A default build (no feature) still accepts `--storage lance` but exits with an
error telling you to rebuild with the feature. See
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
| `--index-path` | `-i` | — | `./data/index` | Directory for the Tantivy full-text index (`{index-path}/tantivy/`), the vector-index snapshot, and the agent-session store (`{index-path}/sessions`). |
| `--storage` | — | `KOWITODB_STORAGE` | `sled` | Storage backend: `sled` or `lance` (`lance` requires a build with `--features lance`). |
| `--lance-uri` | — | `KOWITODB_LANCE_URI` | `{storage-path}/lance` | Lance dataset URI/path, used with `--storage lance`. |
| `--max-results` | — | `KOWITODB_MAX_RESULTS` | `100` | Upper bound on results returned by `Ask`/`Search`. |
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
| `KOWITODB_API_KEY`, `KOWITODB_TLS_CERT`, `KOWITODB_TLS_KEY`, `KOWITODB_METRICS_ADDR`, `KOWITODB_STORAGE`, `KOWITODB_LANCE_URI`, `KOWITODB_MAX_RESULTS` | Fallbacks for the corresponding `serve` flags above. |
| `KOWITODB_EMBEDDING_PROVIDER` | Selects the embedding provider: `openai`, `ollama`, or (unset/other) the deterministic dev proxy. |
| `OPENAI_API_KEY` / `KOWITODB_OPENAI_API_KEY` | API key for the `openai` provider. |
| `KOWITODB_OPENAI_BASE_URL` | OpenAI-compatible base URL (default `https://api.openai.com/v1`). |
| `KOWITODB_EMBEDDING_MODEL` | Embedding model name (default `text-embedding-3-small`; `nomic-embed-text` for Ollama). |
| `KOWITODB_OLLAMA_URL` | Ollama base URL (default `http://localhost:11434/v1`). |

There is no env var for the bind address (use `--addr`) or for the
storage/index paths (use `--storage-path`/`--index-path`). The storage backend is
selected with `--storage` / `KOWITODB_STORAGE` (see below).

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
| sled (default) | Default `serve` / CLI; nothing to do (`--storage sled`). | Disk |
| Lance | Build with `cargo build --release -p kowitodb --features lance`, then `kowitodb serve --storage lance` (or `KOWITODB_STORAGE=lance`), optionally with `--lance-uri` / `KOWITODB_LANCE_URI`. | Disk (Arrow/columnar) |

```bash
cargo build --release -p kowitodb --features lance
KOWITODB_STORAGE=lance ./target/release/kowitodb serve \
  --storage-path /var/lib/kowitodb/storage \
  --index-path /var/lib/kowitodb/index
# Lance dataset defaults to {storage-path}/lance; override with --lance-uri
```

With `--storage lance`, `serve` opens the engine via
`KowitoDBEngine::new_with_lance(uri, index_path)`; the `uri` may be a local path
or any URI Lance supports. Only `serve` selects the backend — the embedded
`ask`/`sql`/`stats` CLI commands always open the sled store. A binary built
without the feature rejects `--storage lance` at startup with an error.

## Dockerfile

The repository root ships a real multi-stage [`Dockerfile`](../Dockerfile). It:

- builds the tuned release binary (`cargo build --release --locked -p kowitodb`)
  in a `rust:1-bookworm` stage, caching third-party dependencies in their own
  layer (the placeholder workspace crates used for that layer are fully
  discarded before the real build);
- ships the binary on `debian:bookworm-slim` with `ca-certificates` and `curl`;
- runs as the unprivileged user `kowitodb` (uid/gid **10001**), which owns
  `/data`;
- mounts a `/data` volume, exposes **50051** (gRPC) and **9090**
  (Prometheus `/metrics` + `/healthz`), and its default `CMD` runs
  `serve --addr 0.0.0.0:50051` with `--metrics-addr 0.0.0.0:9090` against
  `/data/storage` and `/data/index`;
- declares a `HEALTHCHECK` that polls `http://127.0.0.1:9090/healthz` (if you
  override `CMD` without `--metrics-addr 0.0.0.0:9090`, run with
  `--no-healthcheck`) and `STOPSIGNAL SIGTERM` for graceful shutdown.

Build and run:

```bash
docker build -t kowitodb .
docker run --rm \
  -p 50051:50051 -p 9090:9090 \
  -e KOWITODB_API_KEY="$(openssl rand -hex 32)" \
  -v kowitodb-data:/data \
  kowitodb
```

A named volume (as above) inherits the image's `/data` ownership. With a bind
mount (`-v "$PWD/data:/data"`), the host directory must be writable by uid 10001
— or run the container as your own user, as `make docker-run` does:
`docker run --user "$(id -u):$(id -g)" …`.

The image binds **0.0.0.0** (all interfaces) — see
[Security posture](#security-posture). Pass `KOWITODB_API_KEY`,
`KOWITODB_TLS_CERT`/`KOWITODB_TLS_KEY` (mount the PEM files), and embedding env
vars with `-e`/`--env-file`, or override the default `CMD` to add flags.

The stock image is built without the `lance` feature. For a Lance image, add
`--features lance` to the final `cargo build` line in the Dockerfile and set
`KOWITODB_STORAGE=lance`.

## Continuous integration

[`.github/workflows/ci.yml`](../.github/workflows/ci.yml) runs on pushes to
`main` and on pull requests:

- **Rust:** `cargo fmt --all --check`, `cargo clippy --workspace --all-targets
  --locked -- -D warnings` (warnings are errors), and
  `cargo test --workspace --locked`, with `protoc` installed and the build cache
  warmed. `--locked` makes a stale `Cargo.lock` fail CI.
- **SDKs:** builds and vets the Go SDK (`go build ./... && go vet ./...`),
  type-checks the TypeScript SDK (`npm ci && npx tsc --noEmit`), and installs
  the Python SDK and imports it.

Releases are handled by `bump-version.yml` and `publish.yml` — see
[RELEASING.md](../RELEASING.md).

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
{index-path}/hnsw.bin    HNSW vector-index snapshot (persistent)
{index-path}/sessions/   agent conversation sessions (persistent sled store)
```

What persists and what does not:

- **Persistent on disk:** the object store (sled, or a Lance dataset if used) —
  including embeddings and version history — the Tantivy full-text index, and
  **agent memory**: `RecordTurn` sessions are written to a sled store at
  `{index-path}/sessions` and reloaded on startup, so `GetSession` and
  `active_agent_sessions` survive restarts.
- **Snapshotted, else rebuilt:** the HNSW vector index is checkpointed to
  `{index-path}/hnsw.bin` (periodically and on graceful shutdown) and loaded on
  startup; if the snapshot is missing or not from a clean checkpoint (e.g. after
  a hard kill), it is rebuilt from the stored embeddings.
- **In-memory, rebuilt from storage on startup:** the metadata index, the time
  index, and the graph index. `serve` (and the `ask`/`sql`/`stats` CLI
  commands) call `KowitoDBEngine::open()`, which loads/rebuilds these before
  serving — so all search modes work immediately after a restart, with no
  re-ingestion required.
- **Not persistent:** the plan cache (ephemeral) and the brute-force vector
  index (not on the live `ask` path).

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

> **There is no authentication unless you set `--api-key` / `KOWITODB_API_KEY`.**
> Anyone who can reach the gRPC port can read, write, and delete everything.
> The **Docker image binds `0.0.0.0:50051`** (and `0.0.0.0:9090` for metrics),
> so `docker run -p 50051:50051 …` exposes an unauthenticated database on every
> interface of the host unless you pass `-e KOWITODB_API_KEY=…` (and ideally
> TLS), or publish the port on loopback only (`-p 127.0.0.1:50051:50051`).

- **Auth is off by default.** Set `--api-key` (env `KOWITODB_API_KEY`) to require
  a Bearer / `x-api-key` token on every gRPC call. When unset, the server accepts
  any client that can reach the port. The SDKs send the key with `api_key=`
  (Python), `apiKey` (TypeScript), or `WithAPIKey` (Go) — see
  [SDKS.md](SDKS.md#authentication-deadlines-and-tls).
- **TLS is off by default.** Set `--tls-cert` and `--tls-key` to terminate TLS in
  the server itself. When unset, the server speaks plaintext gRPC and SDK clients
  connect with insecure channels.
- **The health-check and reflection gRPC services are always on and
  unauthenticated by design**, so liveness probes and tooling work without
  credentials. They expose service metadata (reflection) but not your data; keep
  the endpoint off the public internet regardless.
- **Default bind is loopback** (`127.0.0.1:50051`) for the binary — but not for
  the Docker image, whose default `CMD` binds `0.0.0.0`. Keep it on loopback
  unless you have a network boundary you trust.
- **The metrics endpoint** (`--metrics-addr`, `/metrics` + `/healthz`) is plain
  HTTP and unauthenticated; it exposes operational counters, not data, but keep
  it on a private network.

Recommended hardening:

1. Turn on `--api-key` and TLS for any non-loopback deployment; rotate the key
   out of band.
2. Keep KowitoDB on a private network / inside the cluster; never expose the
   port to the public internet. A reverse proxy or service mesh (Envoy/Linkerd)
   can add mTLS and richer authz in front if you need more than a static key.
3. Restrict ingress with security groups / network policies to known clients.
4. Run the process as an unprivileged user with write access only to the data
   directory (the Docker image runs as uid 10001), and keep TLS key files
   readable only by that user.
