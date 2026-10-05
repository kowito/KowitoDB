# syntax=docker/dockerfile:1

# ---- Build stage ----
FROM rust:1-bookworm AS builder
WORKDIR /app

# A system protoc (fallback only: kowitodb-server's build.rs uses the vendored
# protoc from `protoc-bin-vendored` when PROTOC is unset).
RUN apt-get update \
    && apt-get install -y --no-install-recommends protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

# Cache dependency builds: copy manifests first, then build all third-party
# dependencies against empty placeholder crates so that layer is reused across
# source-only changes.
COPY Cargo.toml Cargo.lock ./
COPY kowitodb/Cargo.toml kowitodb/
COPY kowitodb-core/Cargo.toml kowitodb-core/
COPY kowitodb-storage/Cargo.toml kowitodb-storage/
COPY kowitodb-index/Cargo.toml kowitodb-index/
COPY kowitodb-planner/Cargo.toml kowitodb-planner/
COPY kowitodb-sql/Cargo.toml kowitodb-sql/
COPY kowitodb-server/Cargo.toml kowitodb-server/
COPY kowitodb-server/proto kowitodb-server/proto/
COPY kowitodb-server/build.rs kowitodb-server/

# Placeholder sources: the binary crate gets an empty main, the library crates
# an empty lib.rs (matching the real crates' target layout).
RUN mkdir -p kowitodb/src \
    && echo 'fn main() {}' > kowitodb/src/main.rs \
    && for crate in kowitodb-core kowitodb-storage kowitodb-index \
         kowitodb-planner kowitodb-sql kowitodb-server; do \
         mkdir -p "$crate/src" && touch "$crate/src/lib.rs"; \
       done

# Build only the dependencies (cached until a manifest, Cargo.lock or the proto
# changes), then throw away EVERYTHING produced for the placeholder workspace
# crates — artifacts (deps/kowitodb-* bins and deps/libkowitodb_*.rlib), build
# script outputs and fingerprints — plus the placeholder sources themselves.
# Without this, cargo can consider the empty placeholder crates fresh (COPY
# keeps the host's older mtimes) and link the real binary against them.
RUN cargo build --release --locked -p kowitodb \
    && rm -rf kowitodb*/src \
        target/release/kowitodb target/release/kowitodb.d \
        target/release/deps/kowitodb* target/release/deps/libkowitodb* \
        target/release/.fingerprint/kowitodb* \
        target/release/build/kowitodb*

# Now copy the real source and build the binary. Only this layer rebuilds on
# source changes. Touch every workspace source file as well, so the rebuild
# never depends on host mtimes.
COPY . .
RUN find kowitodb*/src -name '*.rs' -exec touch {} + \
    && cargo build --release --locked -p kowitodb

# ---- Runtime stage ----
FROM debian:bookworm-slim

# curl is only used by the HEALTHCHECK below.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 kowitodb \
    && useradd --system --uid 10001 --gid kowitodb --home-dir /data \
        --no-create-home --shell /usr/sbin/nologin kowitodb \
    && mkdir -p /data \
    && chown kowitodb:kowitodb /data

WORKDIR /app
COPY --from=builder /app/target/release/kowitodb /usr/local/bin/kowitodb

ENV RUST_LOG=info
# 50051 = gRPC, 9090 = Prometheus /metrics + /healthz
EXPOSE 50051 9090
# Declared after the chown so a fresh named volume inherits kowitodb:kowitodb.
# For a bind mount, make the host directory writable by uid 10001 (or run with
# `--user "$(id -u):$(id -g)"`).
VOLUME ["/data"]

# Run unprivileged (numeric so `runAsNonRoot` policies can verify it).
USER 10001:10001

# The server shuts down gracefully (flushing state) on SIGTERM.
STOPSIGNAL SIGTERM

# Probes the /healthz endpoint served on --metrics-addr (set by the default CMD).
# If you override CMD without `--metrics-addr 0.0.0.0:9090` (or the
# KOWITODB_METRICS_ADDR env var), also pass `--no-healthcheck`.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:9090/healthz || exit 1

# NOTE: the server binds 0.0.0.0 here and has NO authentication unless an API
# key is configured — set KOWITODB_API_KEY (e.g. `docker run -e
# KOWITODB_API_KEY=...`) before exposing the port beyond a trusted network.
ENTRYPOINT ["kowitodb"]
CMD ["serve", \
     "--addr", "0.0.0.0:50051", \
     "--storage-path", "/data/storage", \
     "--index-path", "/data/index", \
     "--metrics-addr", "0.0.0.0:9090"]
