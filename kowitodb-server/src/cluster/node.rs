//! Cluster — remote data-node client (RemoteNode) + auth interceptor (split from the former cluster.rs god-file).
#![allow(clippy::result_large_err)]

use super::*;

/// Injects the gateway's API key as a Bearer token on every outbound call to a
/// data node, so the gateway can authenticate to nodes that require a key.
#[derive(Clone)]
pub struct AuthInterceptor {
    token: Option<tonic::metadata::MetadataValue<tonic::metadata::Ascii>>,
}

impl tonic::service::Interceptor for AuthInterceptor {
    fn call(&mut self, mut req: tonic::Request<()>) -> Result<tonic::Request<()>, Status> {
        if let Some(t) = &self.token {
            req.metadata_mut().insert("authorization", t.clone());
        }
        Ok(req)
    }
}

/// A remote data node reached over gRPC.
pub struct RemoteNode {
    addr: String,
    client:
        KowitoDbClient<tonic::service::interceptor::InterceptedService<Channel, AuthInterceptor>>,
}

impl RemoteNode {
    /// Connect to a peer address (`host:port` or a full URL), presenting
    /// `api_key` (if set) as a Bearer token on every call.
    pub async fn connect(addr: impl Into<String>, api_key: Option<&str>) -> anyhow::Result<Self> {
        let addr = addr.into();
        let endpoint = if addr.starts_with("http") {
            addr.clone()
        } else {
            format!("http://{addr}")
        };
        let channel = Channel::from_shared(endpoint)?.connect().await?;
        let token = match api_key {
            Some(k) => Some(
                format!("Bearer {k}")
                    .parse()
                    .map_err(|_| anyhow::anyhow!("API key contains invalid header characters"))?,
            ),
            None => None,
        };
        let client = KowitoDbClient::with_interceptor(channel, AuthInterceptor { token });
        Ok(Self { addr, client })
    }

    pub fn addr(&self) -> &str {
        &self.addr
    }
}

#[tonic::async_trait]
impl ClusterNode for RemoteNode {
    async fn insert(&self, req: proto::InsertRequest) -> Result<proto::InsertResponse, Status> {
        self.client
            .clone()
            .insert(req)
            .await
            .map(|r| r.into_inner())
    }
    async fn batch_insert(
        &self,
        req: proto::BatchInsertRequest,
    ) -> Result<proto::BatchInsertResponse, Status> {
        self.client
            .clone()
            .batch_insert(req)
            .await
            .map(|r| r.into_inner())
    }
    async fn remember(
        &self,
        req: proto::RememberRequest,
    ) -> Result<proto::RememberResponse, Status> {
        self.client
            .clone()
            .remember(req)
            .await
            .map(|r| r.into_inner())
    }
    async fn get(&self, req: proto::GetRequest) -> Result<proto::GetResponse, Status> {
        self.client.clone().get(req).await.map(|r| r.into_inner())
    }
    async fn update(&self, req: proto::UpdateRequest) -> Result<proto::UpdateResponse, Status> {
        self.client
            .clone()
            .update(req)
            .await
            .map(|r| r.into_inner())
    }
    async fn delete(&self, req: proto::DeleteRequest) -> Result<proto::DeleteResponse, Status> {
        self.client
            .clone()
            .delete(req)
            .await
            .map(|r| r.into_inner())
    }
    async fn list(&self, req: proto::ListRequest) -> Result<proto::ListResponse, Status> {
        self.client.clone().list(req).await.map(|r| r.into_inner())
    }
    async fn search(&self, req: proto::SearchRequest) -> Result<proto::SearchResponse, Status> {
        self.client
            .clone()
            .search(req)
            .await
            .map(|r| r.into_inner())
    }
    async fn ask(&self, req: proto::AskRequest) -> Result<proto::AskResponse, Status> {
        self.client.clone().ask(req).await.map(|r| r.into_inner())
    }
    async fn sql(&self, req: proto::SqlRequest) -> Result<proto::SqlResponse, Status> {
        self.client.clone().sql(req).await.map(|r| r.into_inner())
    }
    async fn stats(&self, req: proto::StatsRequest) -> Result<proto::StatsResponse, Status> {
        self.client.clone().stats(req).await.map(|r| r.into_inner())
    }
    async fn record_turn(
        &self,
        req: proto::RecordTurnRequest,
    ) -> Result<proto::RecordTurnResponse, Status> {
        self.client
            .clone()
            .record_turn(req)
            .await
            .map(|r| r.into_inner())
    }
    async fn get_session(
        &self,
        req: proto::GetSessionRequest,
    ) -> Result<proto::GetSessionResponse, Status> {
        self.client
            .clone()
            .get_session(req)
            .await
            .map(|r| r.into_inner())
    }
}
