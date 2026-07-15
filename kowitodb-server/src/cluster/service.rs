//! Cluster — the ClusterService gRPC surface (gateway) (split from the former cluster.rs god-file).
#![allow(clippy::result_large_err)]

use super::*;

/// gRPC service that exposes a [`Cluster`] under the standard `KowitoDB` API —
/// i.e. the gateway speaks the exact same protocol as a single node, so clients
/// and SDKs are unchanged.
pub struct ClusterService {
    cluster: Arc<Cluster>,
}

impl ClusterService {
    pub fn new(cluster: Arc<Cluster>) -> Self {
        Self { cluster }
    }
}

#[tonic::async_trait]
impl crate::proto::kowito_db_server::KowitoDb for ClusterService {
    async fn insert(
        &self,
        request: Request<proto::InsertRequest>,
    ) -> Result<Response<proto::InsertResponse>, Status> {
        self.cluster
            .insert(request.into_inner())
            .await
            .map(Response::new)
    }
    async fn batch_insert(
        &self,
        request: Request<proto::BatchInsertRequest>,
    ) -> Result<Response<proto::BatchInsertResponse>, Status> {
        self.cluster
            .batch_insert(request.into_inner())
            .await
            .map(Response::new)
    }
    async fn remember(
        &self,
        request: Request<proto::RememberRequest>,
    ) -> Result<Response<proto::RememberResponse>, Status> {
        self.cluster
            .remember(request.into_inner())
            .await
            .map(Response::new)
    }
    async fn get(
        &self,
        request: Request<proto::GetRequest>,
    ) -> Result<Response<proto::GetResponse>, Status> {
        self.cluster
            .get(request.into_inner())
            .await
            .map(Response::new)
    }
    async fn update(
        &self,
        request: Request<proto::UpdateRequest>,
    ) -> Result<Response<proto::UpdateResponse>, Status> {
        self.cluster
            .update(request.into_inner())
            .await
            .map(Response::new)
    }
    async fn delete(
        &self,
        request: Request<proto::DeleteRequest>,
    ) -> Result<Response<proto::DeleteResponse>, Status> {
        self.cluster
            .delete(request.into_inner())
            .await
            .map(Response::new)
    }
    async fn list(
        &self,
        request: Request<proto::ListRequest>,
    ) -> Result<Response<proto::ListResponse>, Status> {
        self.cluster
            .list(request.into_inner())
            .await
            .map(Response::new)
    }
    async fn search(
        &self,
        request: Request<proto::SearchRequest>,
    ) -> Result<Response<proto::SearchResponse>, Status> {
        self.cluster
            .search(request.into_inner())
            .await
            .map(Response::new)
    }
    async fn ask(
        &self,
        request: Request<proto::AskRequest>,
    ) -> Result<Response<proto::AskResponse>, Status> {
        self.cluster
            .ask(request.into_inner())
            .await
            .map(Response::new)
    }
    async fn sql(
        &self,
        request: Request<proto::SqlRequest>,
    ) -> Result<Response<proto::SqlResponse>, Status> {
        self.cluster
            .sql(request.into_inner())
            .await
            .map(Response::new)
    }
    async fn record_turn(
        &self,
        request: Request<proto::RecordTurnRequest>,
    ) -> Result<Response<proto::RecordTurnResponse>, Status> {
        self.cluster
            .record_turn(request.into_inner())
            .await
            .map(Response::new)
    }
    async fn get_session(
        &self,
        request: Request<proto::GetSessionRequest>,
    ) -> Result<Response<proto::GetSessionResponse>, Status> {
        self.cluster
            .get_session(request.into_inner())
            .await
            .map(Response::new)
    }
    async fn stats(
        &self,
        request: Request<proto::StatsRequest>,
    ) -> Result<Response<proto::StatsResponse>, Status> {
        self.cluster
            .stats(request.into_inner())
            .await
            .map(Response::new)
    }
}
