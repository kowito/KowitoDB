mod db;
mod embedding;
mod memory;
mod openai;
mod service;

pub use db::KowitoDBEngine;
pub use embedding::{EmbeddingClient, EmbeddingResult, ProxyEmbeddingClient};
pub use memory::{AgentMemory, AgentSession, ConversationTurn, TurnRole};
pub use openai::{OpenAiConfig, OpenAiEmbeddingClient};
pub use service::KowitoDBService;

// Re-export generated proto code
pub mod proto {
    tonic::include_proto!("kowitodb");
}

use std::net::SocketAddr;
use tonic::transport::Server;
use tracing::info;

/// Start the gRPC server bound to `addr`.
pub async fn serve(engine: KowitoDBEngine, addr: SocketAddr) -> anyhow::Result<()> {
    let service = KowitoDBService::new(engine);

    info!("KowitoDB server starting on {}", addr);

    Server::builder()
        .add_service(proto::kowito_db_server::KowitoDbServer::new(service))
        .serve(addr)
        .await?;

    Ok(())
}
