# syntax=docker/dockerfile:1

# ---- Build stage ----
FROM rust:1-bookworm AS builder
WORKDIR /app

# protoc is required by tonic-build to compile the gRPC definitions.
RUN apt-get update \
    && apt-get install -y --no-install-recommends protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

# Cache dependency builds: copy manifests first, then build dependencies with a
# dummy lib.rs so they're cached across source-only changes.
COPY Cargo.toml Cargo.lock ./
COPY kowitodb/Cargo.toml kowitodb/
COPY kowitodb-core/Cargo.toml kowitodb-core/
COPY kowitodb-storage/Cargo.toml kowitodb-storage/
COPY kowitodb-index/Cargo.toml kowitodb-index/
COPY kowitodb-planner/Cargo.toml kowitodb-planner/
COPY kowitodb-sql/Cargo.toml kowitodb-sql/
COPY kowitodb-server/Cargo.toml kowitodb-server/

# Create dummy source files so `cargo build` can compile just the dependencies.
RUN mkdir -p kowitodb/src kowitodb-core/src kowitodb-storage/src \
    kowitodb-index/src kowitodb-planner/src kowitodb-sql/src \
    kowitodb-server/src
RUN for crate in kowitodb kowitodb-core kowitodb-storage kowitodb-index \
    kowitodb-planner kowitodb-sql kowitodb-server; do \
      echo 'fn main() {}' > "$crate/src/main.rs"; \
      touch "$crate/src/lib.rs"; \
    done
COPY kowitodb-server/proto kowitodb-server/proto/
COPY kowitodb-server/build.rs kowitodb-server/

# Build only the dependencies (this layer is cached until Cargo.toml or proto changes).
RUN cargo build --release -p kowitodb && rm -rf target/release/deps/kowitodb*

# Now copy the real source and build the binary. Only this layer rebuilds on
# source changes.
COPY . .
RUN touch kowitodb-server/src/lib.rs kowitodb/src/main.rs
RUN cargo build --release -p kowitodb

# ---- Runtime stage ----
FROM debian:bookworm-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/target/release/kowitodb /usr/local/bin/kowitodb

ENV RUST_LOG=info
# 50051 = gRPC, 9090 = Prometheus /metrics + /healthz
EXPOSE 50051 9090
VOLUME ["/data"]

ENTRYPOINT ["kowitodb"]
CMD ["serve", \
     "--addr", "0.0.0.0:50051", \
     "--storage-path", "/data/storage", \
     "--index-path", "/data/index", \
     "--metrics-addr", "0.0.0.0:9090"]
