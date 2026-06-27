# syntax=docker/dockerfile:1

# ---- Build stage ----
FROM rust:1-bookworm AS builder
WORKDIR /app

# protoc is required by tonic-build to compile the gRPC definitions.
RUN apt-get update \
    && apt-get install -y --no-install-recommends protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

COPY . .
# Build the optimized release binary (uses the tuned [profile.release]).
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
