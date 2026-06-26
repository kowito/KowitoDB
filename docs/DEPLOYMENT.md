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
exist, opens the engine, and serves the `KowitoDB` gRPC service on `--addr`.

## Configuration

All configuration is via CLI flags and one environment variable. There is no
config file.

### CLI flags (`serve`)

| Flag | Short | Default | Description |
| --- | --- | --- | --- |
| `--addr` | `-a` | `127.0.0.1:50051` | Socket address to bind the gRPC server. Use `0.0.0.0:50051` to accept remote connections. |
| `--storage-path` | `-s` | `./data/storage` | Directory for the sled object store. |
| `--index-path` | `-i` | `./data/index` | Directory for the Tantivy full-text index (created under `{index-path}/tantivy/`). |

The default bind address is loopback-only (`127.0.0.1`). Change it deliberately,
and see [Security posture](#security-posture) first.

### Environment variables

| Variable | Effect |
| --- | --- |
| `RUST_LOG` | Sets the `tracing-subscriber` `EnvFilter`. Defaults to `info` when unset. Examples: `RUST_LOG=info`, `RUST_LOG=kowitodb=debug,warn`. |

There are no other environment variables. In particular, there is **no** env
var for the bind port (use `--addr`), no auth/secret configuration, and no env
var that switches the storage backend (that is a build-time feature plus a
code-level constructor choice).

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

A multi-stage build that compiles a release binary and ships it on a slim
runtime image. `protoc` is not required at build time — the proto is compiled by
`tonic-build` using the vendored compiler via the `prost` toolchain.

```dockerfile
# ---- builder ----
FROM rust:1-bookworm AS builder
WORKDIR /src

# Cache dependencies first.
COPY Cargo.toml Cargo.lock ./
COPY kowitodb-core/Cargo.toml      kowitodb-core/
COPY kowitodb-storage/Cargo.toml   kowitodb-storage/
COPY kowitodb-index/Cargo.toml     kowitodb-index/
COPY kowitodb-planner/Cargo.toml   kowitodb-planner/
COPY kowitodb-sql/Cargo.toml       kowitodb-sql/
COPY kowitodb-server/Cargo.toml    kowitodb-server/
COPY kowitodb/Cargo.toml           kowitodb/

# Bring in the full source and build.
COPY . .
RUN cargo build --release --bin kowitodb

# ---- runtime ----
FROM debian:bookworm-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /src/target/release/kowitodb /usr/local/bin/kowitodb

# Persist data on a mounted volume.
VOLUME ["/var/lib/kowitodb"]
ENV RUST_LOG=info
EXPOSE 50051

ENTRYPOINT ["kowitodb", "serve", \
  "--addr", "0.0.0.0:50051", \
  "--storage-path", "/var/lib/kowitodb/storage", \
  "--index-path", "/var/lib/kowitodb/index"]
```

Build and run:

```bash
docker build -t kowitodb:0.1.0 .
docker run --rm -p 50051:50051 \
  -v kowitodb-data:/var/lib/kowitodb \
  kowitodb:0.1.0
```

For the Lance backend you would need a server entry point that calls
`new_with_lance` and a build with `--features lance`; the stock `kowitodb serve`
binary does not select Lance.

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

- **Persistent:** the object store (sled, or a Lance dataset if used) and the
  Tantivy full-text index.
- **Not persistent (in-memory, rebuilt only by re-inserting objects):** the
  HNSW vector index, the brute-force vector index, the metadata index, the time
  index, and the graph index; plus the plan cache and agent memory.

This means that after a restart the object store still holds every object and
full-text search still works, but vector/metadata/time/graph search return
nothing until those objects are re-inserted. This is a real operational
constraint, not a tuning knob — see
[OPERATIONS.md → Index persistence](OPERATIONS.md#index-persistence-and-restarts)
for how to handle it (e.g. a re-ingestion pass on startup).

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
  cumulative ask latency, average ask latency, and uptime. These are **not**
  exposed as a Prometheus endpoint; some counters leak into the `Stats` RPC's
  `index_size_bytes` field (the server adds `snapshot.ask_count` to it). Treat
  the `Stats` RPC and logs as the current observability surface.
- **Health checks.** There is **no** gRPC health-checking service and **no**
  gRPC reflection registered. For liveness, use a TCP connect check against
  `--addr`, or have a sidecar issue a cheap `Stats` RPC.

See [OPERATIONS.md](OPERATIONS.md) for interpreting the metrics surfaced by
`Stats`.

## Security posture

Read this before binding to anything other than loopback.

- **No authentication.** The gRPC server accepts any client that can reach the
  port. There is no API key, token, or mTLS check anywhere in the request path.
- **No TLS.** The server is configured for plaintext gRPC. SDK clients connect
  with insecure channels by default. (The Go SDK lets you pass
  `grpc.DialOption`s for credentials, but the server does not terminate TLS, so
  you would need a TLS-terminating proxy in front.)
- **Default bind is loopback** (`127.0.0.1:50051`). Keep it that way unless you
  have a network boundary you trust.

Recommended hardening until first-class auth/TLS land:

1. Keep KowitoDB on a private network / inside the cluster; never expose the
   port to the public internet.
2. Terminate TLS and enforce authentication at a reverse proxy or service mesh
   (e.g. Envoy/Linkerd) in front of the server.
3. Restrict ingress with security groups / network policies to known clients.
4. Run the process as an unprivileged user with write access only to the data
   directory.
