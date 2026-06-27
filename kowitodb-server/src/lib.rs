mod cluster;
mod config;
mod db;
mod embedding;
#[cfg(feature = "local-embeddings")]
mod local_embedding;
mod memory;
mod metrics;
mod openai;
mod rerank;
mod service;

pub use cluster::{Cluster, ClusterService};
pub use config::ServerConfig;
pub use db::KowitoDBEngine;
pub use embedding::{EmbeddingClient, EmbeddingResult, ProxyEmbeddingClient};
#[cfg(feature = "local-embeddings")]
pub use local_embedding::LocalEmbeddingClient;
pub use memory::{AgentMemory, AgentSession, ConversationTurn, TurnRole};
pub use metrics::{MetricsCollector, ServerMetrics};
pub use openai::{OpenAiConfig, OpenAiEmbeddingClient};
pub use service::KowitoDBService;

pub mod proto {
    tonic::include_proto!("kowitodb");

    /// Encoded protobuf file descriptor set, emitted by `build.rs`, used to
    /// serve gRPC reflection.
    pub const FILE_DESCRIPTOR_SET: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/kowitodb_descriptor.bin"));
}

use std::net::SocketAddr;
use std::sync::Arc;

use axum::{routing::get, Router};
use tonic::transport::{Identity, Server, ServerTlsConfig};
use tonic::{Request, Status};
use tracing::{info, warn};

use proto::kowito_db_server::KowitoDbServer;

/// Start the gRPC server bound to `addr` with default (dev) configuration:
/// plaintext, no auth, no metrics endpoint.
pub async fn serve(engine: KowitoDBEngine, addr: SocketAddr) -> anyhow::Result<()> {
    serve_with_config(engine, addr, ServerConfig::default()).await
}

/// Start the gRPC server with explicit production configuration (auth, TLS,
/// metrics endpoint). See [`ServerConfig`].
// The interceptor closure inherits `check_auth`'s large `Result<_, Status>`.
#[allow(clippy::result_large_err)]
pub async fn serve_with_config(
    engine: KowitoDBEngine,
    addr: SocketAddr,
    config: ServerConfig,
) -> anyhow::Result<()> {
    let metrics = Arc::new(MetricsCollector::new());
    let max_results = config.max_results.unwrap_or(config::DEFAULT_MAX_RESULTS);
    let engine = Arc::new(engine);
    let service = KowitoDBService::new(engine.clone(), metrics.clone(), max_results);

    // Periodically persist the vector index so it survives restarts without a
    // full rebuild from storage.
    {
        let engine = engine.clone();
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(std::time::Duration::from_secs(300));
            tick.tick().await; // consume the immediate first tick
            loop {
                tick.tick().await;
                if let Err(e) = engine.checkpoint() {
                    warn!("Periodic vector-index checkpoint failed: {}", e);
                }
            }
        });
    }

    // Optional Prometheus/health HTTP endpoint on a separate port.
    if let Some(metrics_addr) = config.metrics_addr {
        spawn_metrics_server(metrics_addr, metrics.clone()).await?;
    }

    // gRPC health reporting + reflection (always on; unauthenticated so probes
    // and tooling work without credentials).
    let (mut health_reporter, health_service) = tonic_health::server::health_reporter();
    health_reporter
        .set_serving::<KowitoDbServer<KowitoDBService>>()
        .await;
    let reflection = tonic_reflection::server::Builder::configure()
        .register_encoded_file_descriptor_set(proto::FILE_DESCRIPTOR_SET)
        .build_v1alpha()?;

    // Wrap the main service with an API-key interceptor when configured.
    let api_key = config.api_key.clone();
    if api_key.is_some() {
        info!("API-key authentication enabled");
    } else {
        warn!("No API key set — gRPC endpoint is unauthenticated (set KOWITODB_API_KEY)");
    }
    let main_service = KowitoDbServer::with_interceptor(service, move |req: Request<()>| {
        check_auth(&api_key, req)
    });

    let mut builder = Server::builder();
    if config.tls_enabled() {
        let cert = std::fs::read(config.tls_cert.as_ref().unwrap())?;
        let key = std::fs::read(config.tls_key.as_ref().unwrap())?;
        let identity = Identity::from_pem(cert, key);
        builder = builder.tls_config(ServerTlsConfig::new().identity(identity))?;
        info!("TLS enabled");
    }

    info!(
        "KowitoDB server starting on {} ({})",
        addr,
        if config.tls_enabled() {
            "https"
        } else {
            "http"
        }
    );

    // Graceful shutdown on Ctrl-C / SIGINT so the final index checkpoint runs.
    let shutdown = async {
        let _ = tokio::signal::ctrl_c().await;
        info!("Shutdown signal received; draining");
    };

    builder
        .add_service(health_service)
        .add_service(reflection)
        .add_service(main_service)
        .serve_with_shutdown(addr, shutdown)
        .await?;

    // Persist the vector index one last time on the way out.
    match engine.checkpoint() {
        Ok(()) => info!("Final vector-index checkpoint written"),
        Err(e) => warn!("Final vector-index checkpoint failed: {}", e),
    }

    Ok(())
}

/// Start a cluster **gateway**: a coordinator that fronts `peers` data nodes and
/// speaks the exact same `KowitoDB` gRPC API, so clients/SDKs are unchanged.
/// Writes are partitioned (and optionally replicated); reads scatter-gather.
pub async fn serve_gateway(
    addr: SocketAddr,
    peers: Vec<String>,
    replication_factor: usize,
    write_quorum: usize,
) -> anyhow::Result<()> {
    let cluster = Arc::new(Cluster::connect(&peers, replication_factor, write_quorum).await?);
    info!(
        "KowitoDB gateway: {} data node(s), replication_factor={}, write_quorum={}",
        cluster.node_count(),
        replication_factor,
        write_quorum
    );

    // Heartbeat: probe data nodes periodically so down nodes are skipped on reads
    // and recovered automatically, without waiting for a request to discover them.
    {
        let cluster = cluster.clone();
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(std::time::Duration::from_secs(5));
            tick.tick().await; // consume the immediate first tick
            loop {
                tick.tick().await;
                cluster.heartbeat_once().await;
            }
        });
    }

    let service = ClusterService::new(cluster);

    let (mut health_reporter, health_service) = tonic_health::server::health_reporter();
    health_reporter
        .set_serving::<KowitoDbServer<ClusterService>>()
        .await;
    let reflection = tonic_reflection::server::Builder::configure()
        .register_encoded_file_descriptor_set(proto::FILE_DESCRIPTOR_SET)
        .build_v1alpha()?;

    info!("KowitoDB gateway listening on {}", addr);
    Server::builder()
        .add_service(health_service)
        .add_service(reflection)
        .add_service(KowitoDbServer::new(service))
        .serve(addr)
        .await?;
    Ok(())
}

/// Validate the API key on an incoming request. A no-op when no key is set.
// tonic interceptors must return `Result<_, Status>`; Status is intentionally large.
#[allow(clippy::result_large_err)]
fn check_auth(api_key: &Option<String>, req: Request<()>) -> Result<Request<()>, Status> {
    let Some(expected) = api_key else {
        return Ok(req);
    };

    let md = req.metadata();
    let presented = md
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .map(|v| v.strip_prefix("Bearer ").unwrap_or(v))
        .or_else(|| md.get("x-api-key").and_then(|v| v.to_str().ok()));

    match presented {
        Some(key) if key == expected => Ok(req),
        _ => Err(Status::unauthenticated("invalid or missing API key")),
    }
}

/// Spawn the HTTP server exposing `/metrics` (Prometheus) and `/healthz`.
async fn spawn_metrics_server(
    addr: SocketAddr,
    metrics: Arc<MetricsCollector>,
) -> anyhow::Result<()> {
    let app = Router::new()
        .route(
            "/metrics",
            get({
                let metrics = metrics.clone();
                move || {
                    let metrics = metrics.clone();
                    async move { metrics.to_prometheus() }
                }
            }),
        )
        .route("/healthz", get(|| async { "ok" }));

    let listener = tokio::net::TcpListener::bind(addr).await?;
    info!("Metrics endpoint on http://{}/metrics", addr);
    tokio::spawn(async move {
        if let Err(e) = axum::serve(listener, app).await {
            warn!("Metrics server stopped: {}", e);
        }
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req_with(header: &'static str, value: &str) -> Request<()> {
        let mut req = Request::new(());
        req.metadata_mut().insert(header, value.parse().unwrap());
        req
    }

    #[test]
    fn auth_disabled_allows_all() {
        assert!(check_auth(&None, Request::new(())).is_ok());
    }

    #[test]
    fn auth_accepts_bearer_and_x_api_key() {
        let key = Some("secret".to_string());
        assert!(check_auth(&key, req_with("authorization", "Bearer secret")).is_ok());
        assert!(check_auth(&key, req_with("x-api-key", "secret")).is_ok());
    }

    #[test]
    fn auth_rejects_missing_and_wrong() {
        let key = Some("secret".to_string());
        assert!(check_auth(&key, Request::new(())).is_err());
        assert!(check_auth(&key, req_with("authorization", "Bearer nope")).is_err());
        assert!(check_auth(&key, req_with("x-api-key", "nope")).is_err());
    }
}
