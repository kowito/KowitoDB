mod db;
mod embedding;
mod memory;
mod metrics;
mod openai;
mod service;

pub use db::KowitoDBEngine;
pub use embedding::{EmbeddingClient, EmbeddingResult, ProxyEmbeddingClient};
pub use memory::{AgentMemory, AgentSession, ConversationTurn, TurnRole};
pub use metrics::{MetricsCollector, ServerMetrics};
pub use openai::{OpenAiConfig, OpenAiEmbeddingClient};
pub use service::KowitoDBService;

pub mod proto {
    tonic::include_proto!("kowitodb");
}

use std::net::SocketAddr;
use std::sync::Arc;
use tonic::transport::Server;
use tracing::info;

/// Start the gRPC server bound to `addr`.
pub async fn serve(engine: KowitoDBEngine, addr: SocketAddr) -> anyhow::Result<()> {
    let metrics = Arc::new(MetricsCollector::new());
    let service = KowitoDBService::new(engine, metrics);

    info!("KowitoDB server starting on {}", addr);

    Server::builder()
        .add_service(proto::kowito_db_server::KowitoDbServer::new(service))
        .serve(addr)
        .await?;

    Ok(())
}
