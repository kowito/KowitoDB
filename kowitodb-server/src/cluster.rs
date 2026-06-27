//! Distributed cluster coordinator ("gateway" mode).
//!
//! Turns N KowitoDB data nodes into one logical database via a shared-nothing,
//! scatter-gather design:
//!
//! - **Writes** are partitioned by object id (consistent `id % N`) and optionally
//!   replicated to `R` consecutive nodes (best-effort: succeeds if any replica
//!   accepts). The gateway assigns the id up front so it can route before the id
//!   would otherwise be server-generated.
//! - **Id-keyed ops** (get/update/delete) route to the owning replica set.
//! - **Reads** (search/ask/sql/list/stats) scatter to every node in parallel and
//!   merge: search/ask de-duplicate by id (keeping the best score) and re-rank;
//!   stats/list aggregate.
//! - **Agent sessions** partition by `session_id`.
//!
//! This provides real horizontal distribution. It is **not** a consensus-backed,
//! strongly-consistent cluster: there is no Raft, no quorum, no automatic
//! rebalancing on membership change, and no failure recovery/repair. Those are
//! the production-HA follow-ups. Replication here is best-effort write fan-out
//! with read-side de-duplication.

// gRPC handlers return `Result<_, Status>`; tonic's Status is intentionally large.
#![allow(clippy::result_large_err)]

use std::collections::HashMap;
use std::sync::Arc;

use kowitodb_core::ObjectId;
use tonic::transport::Channel;
use tonic::{Request, Response, Status};
use tracing::{info, warn};

use crate::proto;
use crate::proto::kowito_db_client::KowitoDbClient;

/// One node in the cluster — a data node over gRPC, or a test double.
#[tonic::async_trait]
pub trait ClusterNode: Send + Sync {
    async fn insert(&self, req: proto::InsertRequest) -> Result<proto::InsertResponse, Status>;
    async fn batch_insert(
        &self,
        req: proto::BatchInsertRequest,
    ) -> Result<proto::BatchInsertResponse, Status>;
    async fn remember(
        &self,
        req: proto::RememberRequest,
    ) -> Result<proto::RememberResponse, Status>;
    async fn get(&self, req: proto::GetRequest) -> Result<proto::GetResponse, Status>;
    async fn update(&self, req: proto::UpdateRequest) -> Result<proto::UpdateResponse, Status>;
    async fn delete(&self, req: proto::DeleteRequest) -> Result<proto::DeleteResponse, Status>;
    async fn list(&self, req: proto::ListRequest) -> Result<proto::ListResponse, Status>;
    async fn search(&self, req: proto::SearchRequest) -> Result<proto::SearchResponse, Status>;
    async fn ask(&self, req: proto::AskRequest) -> Result<proto::AskResponse, Status>;
    async fn sql(&self, req: proto::SqlRequest) -> Result<proto::SqlResponse, Status>;
    async fn stats(&self, req: proto::StatsRequest) -> Result<proto::StatsResponse, Status>;
    async fn record_turn(
        &self,
        req: proto::RecordTurnRequest,
    ) -> Result<proto::RecordTurnResponse, Status>;
    async fn get_session(
        &self,
        req: proto::GetSessionRequest,
    ) -> Result<proto::GetSessionResponse, Status>;
}

/// A remote data node reached over gRPC.
pub struct RemoteNode {
    addr: String,
    client: KowitoDbClient<Channel>,
}

impl RemoteNode {
    /// Connect to a peer address (`host:port` or a full URL).
    pub async fn connect(addr: impl Into<String>) -> anyhow::Result<Self> {
        let addr = addr.into();
        let endpoint = if addr.starts_with("http") {
            addr.clone()
        } else {
            format!("http://{addr}")
        };
        let client = KowitoDbClient::connect(endpoint).await?;
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

/// The distributed coordinator over a set of data nodes.
pub struct Cluster {
    nodes: Vec<Arc<dyn ClusterNode>>,
    replication_factor: usize,
}

impl Cluster {
    /// Build a cluster from already-constructed nodes.
    pub fn new(nodes: Vec<Arc<dyn ClusterNode>>, replication_factor: usize) -> Self {
        let n = nodes.len().max(1);
        Self {
            nodes,
            replication_factor: replication_factor.clamp(1, n),
        }
    }

    /// Connect to peer data nodes over gRPC.
    pub async fn connect(peers: &[String], replication_factor: usize) -> anyhow::Result<Self> {
        if peers.is_empty() {
            anyhow::bail!("a cluster needs at least one peer node");
        }
        let mut nodes: Vec<Arc<dyn ClusterNode>> = Vec::with_capacity(peers.len());
        for peer in peers {
            let node = RemoteNode::connect(peer.clone()).await?;
            info!("Cluster: connected to data node {}", node.addr());
            nodes.push(Arc::new(node));
        }
        Ok(Self::new(nodes, replication_factor))
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    fn replicas(&self, owner: usize) -> Vec<usize> {
        let n = self.nodes.len();
        (0..self.replication_factor)
            .map(|i| (owner + i) % n)
            .collect()
    }

    fn replicas_for_id(&self, id: ObjectId) -> Vec<usize> {
        self.replicas((id.as_u128() % self.nodes.len() as u128) as usize)
    }

    fn replicas_for_key(&self, key: &str) -> Vec<usize> {
        let h = key.bytes().fold(1469598103934665603u64, |a, b| {
            (a ^ b as u64).wrapping_mul(1099511628211)
        });
        self.replicas((h % self.nodes.len() as u64) as usize)
    }

    // ---- Writes (partitioned + replicated) ----

    pub async fn insert(
        &self,
        mut req: proto::InsertRequest,
    ) -> Result<proto::InsertResponse, Status> {
        let id = parse_or_new_id(req.id.as_deref());
        req.id = Some(id.to_string());
        self.write_to_replicas(&self.replicas_for_id(id), |n| {
            let r = req.clone();
            async move { n.insert(r).await.map(|_| ()) }
        })
        .await?;
        Ok(proto::InsertResponse { id: id.to_string() })
    }

    pub async fn remember(
        &self,
        mut req: proto::RememberRequest,
    ) -> Result<proto::RememberResponse, Status> {
        let id = parse_or_new_id(req.id.as_deref());
        req.id = Some(id.to_string());
        self.write_to_replicas(&self.replicas_for_id(id), |n| {
            let r = req.clone();
            async move { n.remember(r).await.map(|_| ()) }
        })
        .await?;
        Ok(proto::RememberResponse { id: id.to_string() })
    }

    pub async fn batch_insert(
        &self,
        req: proto::BatchInsertRequest,
    ) -> Result<proto::BatchInsertResponse, Status> {
        // Assign ids, then group each item into every replica node's sub-batch.
        let mut ids = Vec::with_capacity(req.items.len());
        let mut groups: HashMap<usize, Vec<proto::InsertRequest>> = HashMap::new();
        for mut item in req.items {
            let id = parse_or_new_id(item.id.as_deref());
            item.id = Some(id.to_string());
            ids.push(id.to_string());
            for &node in &self.replicas_for_id(id) {
                groups.entry(node).or_default().push(item.clone());
            }
        }
        for (node, items) in groups {
            let req = proto::BatchInsertRequest { items };
            if let Err(e) = self.nodes[node].batch_insert(req).await {
                warn!("batch_insert on node {node} failed: {e}");
            }
        }
        Ok(proto::BatchInsertResponse { ids })
    }

    /// Run `f` on each replica; succeed if at least one accepts.
    async fn write_to_replicas<F, Fut>(&self, replicas: &[usize], f: F) -> Result<(), Status>
    where
        F: Fn(Arc<dyn ClusterNode>) -> Fut,
        Fut: std::future::Future<Output = Result<(), Status>>,
    {
        let mut accepted = false;
        let mut last_err = None;
        for &i in replicas {
            match f(self.nodes[i].clone()).await {
                Ok(()) => accepted = true,
                Err(e) => {
                    warn!("write to replica {i} failed: {e}");
                    last_err = Some(e);
                }
            }
        }
        if accepted {
            Ok(())
        } else {
            Err(last_err.unwrap_or_else(|| Status::internal("no replicas available")))
        }
    }

    // ---- Id-keyed ops ----

    pub async fn get(&self, req: proto::GetRequest) -> Result<proto::GetResponse, Status> {
        let id = parse_id(&req.id)?;
        for &i in &self.replicas_for_id(id) {
            if let Ok(resp) = self.nodes[i].get(req.clone()).await {
                if resp.object.is_some() {
                    return Ok(resp);
                }
            }
        }
        Ok(proto::GetResponse { object: None })
    }

    pub async fn update(&self, req: proto::UpdateRequest) -> Result<proto::UpdateResponse, Status> {
        let id = parse_id(&req.id)?;
        let mut out = proto::UpdateResponse {
            updated: false,
            version: 0,
        };
        for &i in &self.replicas_for_id(id) {
            if let Ok(resp) = self.nodes[i].update(req.clone()).await {
                if resp.updated {
                    out = resp;
                }
            }
        }
        Ok(out)
    }

    pub async fn delete(&self, req: proto::DeleteRequest) -> Result<proto::DeleteResponse, Status> {
        let id = parse_id(&req.id)?;
        let mut existed = false;
        for &i in &self.replicas_for_id(id) {
            if let Ok(resp) = self.nodes[i].delete(req.clone()).await {
                existed |= resp.existed;
            }
        }
        Ok(proto::DeleteResponse { existed })
    }

    // ---- Scatter-gather reads ----

    pub async fn search(&self, req: proto::SearchRequest) -> Result<proto::SearchResponse, Status> {
        let top_k = req.top_k.max(1) as usize;
        let responses = self
            .scatter(|n| {
                let r = req.clone();
                async move { n.search(r).await }
            })
            .await;

        let mut by_id: HashMap<String, proto::SearchResult> = HashMap::new();
        for resp in responses {
            for r in resp.results {
                by_id
                    .entry(r.id.clone())
                    .and_modify(|e| {
                        if r.score > e.score {
                            *e = r.clone();
                        }
                    })
                    .or_insert(r);
            }
        }
        let mut merged: Vec<_> = by_id.into_values().collect();
        merged.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        merged.truncate(top_k);

        Ok(proto::SearchResponse {
            total_found: merged.len() as i32,
            results: merged,
            plan_explanation: format!("distributed scatter-gather over {} nodes", self.nodes.len()),
        })
    }

    pub async fn ask(&self, req: proto::AskRequest) -> Result<proto::AskResponse, Status> {
        let max_results = req.max_results.max(1) as usize;
        let detected_intent = String::new();
        let responses = self
            .scatter(|n| {
                let r = req.clone();
                async move { n.ask(r).await }
            })
            .await;

        let mut intent = detected_intent;
        let mut by_id: HashMap<String, proto::AskResult> = HashMap::new();
        for resp in responses {
            if intent.is_empty() {
                intent = resp.detected_intent;
            }
            for r in resp.results {
                by_id
                    .entry(r.id.clone())
                    .and_modify(|e| {
                        if r.relevance_score > e.relevance_score {
                            *e = r.clone();
                        }
                    })
                    .or_insert(r);
            }
        }
        let mut merged: Vec<_> = by_id.into_values().collect();
        merged.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        merged.truncate(max_results);

        Ok(proto::AskResponse {
            results: merged,
            plan_explanation: format!("distributed scatter-gather over {} nodes", self.nodes.len()),
            detected_intent: intent,
        })
    }

    pub async fn sql(&self, req: proto::SqlRequest) -> Result<proto::SqlResponse, Status> {
        // Per-node SQL over each partition; rows are concatenated. NOTE:
        // cross-shard aggregates (COUNT/AVG/...) are per-shard partials, not
        // globally combined — a distributed-SQL planner is future work.
        let responses = self
            .scatter(|n| {
                let r = req.clone();
                async move { n.sql(r).await }
            })
            .await;
        let rows = responses.into_iter().flat_map(|r| r.rows).collect();
        Ok(proto::SqlResponse { rows })
    }

    pub async fn list(&self, req: proto::ListRequest) -> Result<proto::ListResponse, Status> {
        let offset = req.offset as usize;
        let limit = if req.limit == 0 {
            100
        } else {
            req.limit as usize
        };
        // Over-fetch (offset+limit) from each node, merge by id, then page.
        let per_node = proto::ListRequest {
            offset: 0,
            limit: (offset + limit) as u32,
        };
        let responses = self
            .scatter(|n| {
                let r = per_node;
                async move { n.list(r).await }
            })
            .await;

        let total: u64 = responses.iter().map(|r| r.total).sum();
        let mut by_id: HashMap<String, proto::KnowledgeObject> = HashMap::new();
        for resp in responses {
            for obj in resp.objects {
                by_id.entry(obj.id.clone()).or_insert(obj);
            }
        }
        let mut objects: Vec<_> = by_id.into_values().collect();
        objects.sort_by(|a, b| a.id.cmp(&b.id)); // stable global order
        let objects = objects.into_iter().skip(offset).take(limit).collect();
        Ok(proto::ListResponse { objects, total })
    }

    pub async fn stats(&self, req: proto::StatsRequest) -> Result<proto::StatsResponse, Status> {
        let responses = self
            .scatter(|n| {
                let r = req;
                async move { n.stats(r).await }
            })
            .await;

        let mut out = proto::StatsResponse::default();
        let count = responses.len().max(1) as f64;
        let mut hit_rate_sum = 0.0;
        for resp in &responses {
            out.total_objects += resp.total_objects;
            out.vector_count += resp.vector_count;
            out.index_size_bytes += resp.index_size_bytes;
            out.graph_nodes += resp.graph_nodes;
            out.graph_edges += resp.graph_edges;
            out.active_agent_sessions += resp.active_agent_sessions;
            out.total_cost_usd += resp.total_cost_usd;
            out.cache_entries += resp.cache_entries;
            hit_rate_sum += resp.cache_hit_rate;
        }
        out.cache_hit_rate = hit_rate_sum / count;
        Ok(out)
    }

    // ---- Session-keyed ops ----

    pub async fn record_turn(
        &self,
        req: proto::RecordTurnRequest,
    ) -> Result<proto::RecordTurnResponse, Status> {
        let replicas = self.replicas_for_key(&req.session_id);
        let mut out = proto::RecordTurnResponse { turn_count: 0 };
        for &i in &replicas {
            if let Ok(resp) = self.nodes[i].record_turn(req.clone()).await {
                out = resp;
            }
        }
        Ok(out)
    }

    pub async fn get_session(
        &self,
        req: proto::GetSessionRequest,
    ) -> Result<proto::GetSessionResponse, Status> {
        for &i in &self.replicas_for_key(&req.session_id) {
            if let Ok(resp) = self.nodes[i].get_session(req.clone()).await {
                if resp.found {
                    return Ok(resp);
                }
            }
        }
        Ok(proto::GetSessionResponse {
            found: false,
            turns: Vec::new(),
        })
    }

    /// Run `f` against every node in parallel; drop errored nodes.
    async fn scatter<F, Fut, T>(&self, f: F) -> Vec<T>
    where
        F: Fn(Arc<dyn ClusterNode>) -> Fut,
        Fut: std::future::Future<Output = Result<T, Status>>,
    {
        let futures = self.nodes.iter().map(|n| f(n.clone()));
        futures::future::join_all(futures)
            .await
            .into_iter()
            .filter_map(|r| r.ok())
            .collect()
    }
}

/// gRPC service that exposes a [`Cluster`] under the standard `KowitoDB` API —
/// i.e. the gateway speaks the exact same protocol as a single node, so clients
/// and SDKs are unchanged.
pub struct ClusterService {
    cluster: Arc<Cluster>,
}

impl ClusterService {
    pub fn new(cluster: Cluster) -> Self {
        Self {
            cluster: Arc::new(cluster),
        }
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

fn parse_id(s: &str) -> Result<ObjectId, Status> {
    ObjectId::parse_str(s).map_err(|_| Status::invalid_argument("invalid object id"))
}

fn parse_or_new_id(s: Option<&str>) -> ObjectId {
    s.and_then(|s| ObjectId::parse_str(s).ok())
        .unwrap_or_else(uuid::Uuid::new_v4)
}

#[cfg(test)]
mod tests {
    use super::*;
    use parking_lot::Mutex;

    /// In-memory data node: stores inserts, "search" matches by content substring.
    #[derive(Default)]
    struct MockNode {
        objects: Mutex<HashMap<String, (String, f32)>>, // id -> (content, score)
    }

    #[tonic::async_trait]
    impl ClusterNode for MockNode {
        async fn insert(&self, req: proto::InsertRequest) -> Result<proto::InsertResponse, Status> {
            let id = req.id.clone().unwrap();
            self.objects
                .lock()
                .insert(id.clone(), (req.content, req.importance.max(0.1)));
            Ok(proto::InsertResponse { id })
        }
        async fn search(&self, req: proto::SearchRequest) -> Result<proto::SearchResponse, Status> {
            let results: Vec<_> = self
                .objects
                .lock()
                .iter()
                .filter(|(_, (content, _))| content.contains(&req.query))
                .map(|(id, (content, score))| proto::SearchResult {
                    id: id.clone(),
                    content: content.clone(),
                    score: *score,
                    metadata: Default::default(),
                })
                .collect();
            Ok(proto::SearchResponse {
                total_found: results.len() as i32,
                results,
                plan_explanation: String::new(),
            })
        }
        async fn get(&self, req: proto::GetRequest) -> Result<proto::GetResponse, Status> {
            let object =
                self.objects
                    .lock()
                    .get(&req.id)
                    .map(|(content, _)| proto::KnowledgeObject {
                        id: req.id.clone(),
                        content: content.clone(),
                        ..Default::default()
                    });
            Ok(proto::GetResponse { object })
        }
        async fn delete(&self, req: proto::DeleteRequest) -> Result<proto::DeleteResponse, Status> {
            let existed = self.objects.lock().remove(&req.id).is_some();
            Ok(proto::DeleteResponse { existed })
        }
        async fn stats(&self, _req: proto::StatsRequest) -> Result<proto::StatsResponse, Status> {
            Ok(proto::StatsResponse {
                total_objects: self.objects.lock().len() as u64,
                ..Default::default()
            })
        }
        // Unused by these tests:
        async fn batch_insert(
            &self,
            req: proto::BatchInsertRequest,
        ) -> Result<proto::BatchInsertResponse, Status> {
            let mut ids = Vec::new();
            for item in req.items {
                ids.push(self.insert(item).await?.id);
            }
            Ok(proto::BatchInsertResponse { ids })
        }
        async fn remember(
            &self,
            _req: proto::RememberRequest,
        ) -> Result<proto::RememberResponse, Status> {
            Ok(proto::RememberResponse::default())
        }
        async fn update(
            &self,
            _req: proto::UpdateRequest,
        ) -> Result<proto::UpdateResponse, Status> {
            Ok(proto::UpdateResponse::default())
        }
        async fn list(&self, _req: proto::ListRequest) -> Result<proto::ListResponse, Status> {
            Ok(proto::ListResponse::default())
        }
        async fn ask(&self, _req: proto::AskRequest) -> Result<proto::AskResponse, Status> {
            Ok(proto::AskResponse::default())
        }
        async fn sql(&self, _req: proto::SqlRequest) -> Result<proto::SqlResponse, Status> {
            Ok(proto::SqlResponse::default())
        }
        async fn record_turn(
            &self,
            _req: proto::RecordTurnRequest,
        ) -> Result<proto::RecordTurnResponse, Status> {
            Ok(proto::RecordTurnResponse::default())
        }
        async fn get_session(
            &self,
            _req: proto::GetSessionRequest,
        ) -> Result<proto::GetSessionResponse, Status> {
            Ok(proto::GetSessionResponse::default())
        }
    }

    fn cluster(n: usize, rf: usize) -> (Cluster, Vec<Arc<MockNode>>) {
        let mocks: Vec<Arc<MockNode>> = (0..n).map(|_| Arc::new(MockNode::default())).collect();
        let nodes: Vec<Arc<dyn ClusterNode>> = mocks.iter().map(|m| m.clone() as _).collect();
        (Cluster::new(nodes, rf), mocks)
    }

    #[tokio::test]
    async fn test_write_partitioned_and_read_scatter_gather() {
        let (cluster, mocks) = cluster(3, 1);

        // Insert 30 objects; each should land on exactly one node.
        for i in 0..30 {
            cluster
                .insert(proto::InsertRequest {
                    content: format!("doc number {i} about widgets"),
                    ..Default::default()
                })
                .await
                .unwrap();
        }
        let placed: usize = mocks.iter().map(|m| m.objects.lock().len()).sum();
        assert_eq!(placed, 30, "every object stored exactly once (rf=1)");
        // Distribution actually spread across nodes.
        assert!(mocks.iter().all(|m| !m.objects.lock().is_empty()));

        // A scatter-gather search finds matches from all shards, merged.
        let resp = cluster
            .search(proto::SearchRequest {
                query: "widgets".into(),
                top_k: 50,
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(resp.results.len(), 30, "all shards' matches are merged");

        // Stats aggregate across the cluster.
        let stats = cluster.stats(proto::StatsRequest {}).await.unwrap();
        assert_eq!(stats.total_objects, 30);
    }

    #[tokio::test]
    async fn test_replication_and_dedup() {
        let (cluster, mocks) = cluster(3, 2); // replicate to 2 nodes

        let resp = cluster
            .insert(proto::InsertRequest {
                content: "replicated widget".into(),
                ..Default::default()
            })
            .await
            .unwrap();

        // Stored on exactly 2 replicas.
        let copies: usize = mocks.iter().map(|m| m.objects.lock().len()).sum();
        assert_eq!(copies, 2, "rf=2 → two physical copies");

        // Search still returns the object once (de-duplicated by id).
        let search = cluster
            .search(proto::SearchRequest {
                query: "widget".into(),
                top_k: 10,
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(search.results.len(), 1, "replicas de-duplicated on read");
        assert_eq!(search.results[0].id, resp.id);

        // Get routes to a replica and finds it; delete removes from all replicas.
        assert!(cluster
            .get(proto::GetRequest {
                id: resp.id.clone()
            })
            .await
            .unwrap()
            .object
            .is_some());
        assert!(
            cluster
                .delete(proto::DeleteRequest {
                    id: resp.id.clone()
                })
                .await
                .unwrap()
                .existed
        );
        assert_eq!(
            mocks.iter().map(|m| m.objects.lock().len()).sum::<usize>(),
            0
        );
    }
}
