//! Distributed cluster coordinator ("gateway" mode).
//!
//! Turns N KowitoDB data nodes into one logical database via a shared-nothing,
//! scatter-gather design:
//!
//! - **Writes** are partitioned by object id (consistent `id % N`) and optionally
//!   replicated to `R` consecutive nodes. A write succeeds once `write_quorum`
//!   replicas ack (tunable durability). The gateway assigns the id up front so it
//!   can route before the id would otherwise be server-generated.
//! - **Id-keyed ops** (get/update/delete) route to the owning replica set.
//! - **Reads** (search/ask/sql/list/stats) scatter to every node in parallel and
//!   merge: search/ask de-duplicate by id (keeping the best score) and re-rank;
//!   stats/list aggregate. Partial node failure is tolerated; a total outage
//!   surfaces as an error (vs. an empty "no matches" result).
//! - **Agent sessions** partition by `session_id`.
//!
//! `get` performs **read-repair + last-write-wins reconciliation**: it returns
//! the freshest copy (latest `updated_at`) across replicas and heals any replica
//! that is missing, staler, or content-divergent; a **heartbeat** proactively
//! tracks node health.
//!
//! `rebalance()` relocates objects to their correct owners after a membership
//! change, and `sql` combines scalar aggregates (COUNT/SUM/MIN/MAX) across shards.
//!
//! This provides real horizontal distribution with tunable durability, health
//! tracking, read-repair, read-time reconciliation, and rebalancing. It is
//! **not** a consensus-backed, strongly-consistent cluster: there is no Raft, so
//! reads are not linearizable and concurrent conflicting writes resolve by
//! last-write-wins. Consensus is a deliberate non-goal (see ROADMAP).

// gRPC handlers return `Result<_, Status>`; tonic's Status is intentionally large.
#![allow(clippy::result_large_err)]

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use kowitodb_core::ObjectId;
use tonic::transport::Channel;
use tonic::{Request, Response, Status};
use tracing::{debug, info, warn};

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

/// The distributed coordinator over a set of data nodes.
pub struct Cluster {
    nodes: Vec<Arc<dyn ClusterNode>>,
    /// Proactively-tracked health per node (aligned with `nodes`). Reads skip
    /// nodes marked unhealthy; the heartbeat and per-request outcomes update it.
    health: Vec<AtomicBool>,
    replication_factor: usize,
    /// Minimum replica acks required for a write to succeed (durability).
    write_quorum: usize,
}

impl Cluster {
    /// Build a cluster from already-constructed nodes (write_quorum = 1).
    pub fn new(nodes: Vec<Arc<dyn ClusterNode>>, replication_factor: usize) -> Self {
        let n = nodes.len().max(1);
        let health = (0..nodes.len()).map(|_| AtomicBool::new(true)).collect();
        Self {
            nodes,
            health,
            replication_factor: replication_factor.clamp(1, n),
            write_quorum: 1,
        }
    }

    fn is_healthy(&self, i: usize) -> bool {
        self.health[i].load(Ordering::Relaxed)
    }

    /// Update a node's health, logging up/down transitions.
    fn set_health(&self, i: usize, healthy: bool) {
        let prev = self.health[i].swap(healthy, Ordering::Relaxed);
        if prev != healthy {
            if healthy {
                info!("Cluster: node {i} recovered (healthy)");
            } else {
                warn!("Cluster: node {i} marked unhealthy");
            }
        }
    }

    /// Number of nodes currently considered healthy.
    pub fn healthy_count(&self) -> usize {
        (0..self.nodes.len())
            .filter(|&i| self.is_healthy(i))
            .count()
    }

    /// Probe every node once (cheap `stats` call) and update health. Run
    /// periodically by the gateway so down nodes are detected and recovered
    /// without waiting for a request to hit them.
    pub async fn heartbeat_once(&self) {
        for i in 0..self.nodes.len() {
            let ok = self.nodes[i].stats(proto::StatsRequest {}).await.is_ok();
            self.set_health(i, ok);
        }
    }

    /// Require `w` replica acks per write (clamped to the replication factor).
    /// `w >= ceil((R+1)/2)` gives majority-quorum durability.
    pub fn with_write_quorum(mut self, w: usize) -> Self {
        self.write_quorum = w.clamp(1, self.replication_factor);
        self
    }

    /// Connect to peer data nodes over gRPC.
    pub async fn connect(
        peers: &[String],
        replication_factor: usize,
        write_quorum: usize,
        api_key: Option<String>,
    ) -> anyhow::Result<Self> {
        if peers.is_empty() {
            anyhow::bail!("a cluster needs at least one peer node");
        }
        let mut nodes: Vec<Arc<dyn ClusterNode>> = Vec::with_capacity(peers.len());
        for peer in peers {
            let node = RemoteNode::connect(peer.clone(), api_key.as_deref()).await?;
            info!("Cluster: connected to data node {}", node.addr());
            nodes.push(Arc::new(node));
        }
        Ok(Self::new(nodes, replication_factor).with_write_quorum(write_quorum))
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
        // Assign ids and record each id's replica set, then group items into
        // each node's sub-batch.
        let mut ids = Vec::with_capacity(req.items.len());
        let mut replica_sets: Vec<(String, Vec<usize>)> = Vec::with_capacity(req.items.len());
        let mut groups: HashMap<usize, Vec<proto::InsertRequest>> = HashMap::new();
        for mut item in req.items {
            let id = parse_or_new_id(item.id.as_deref());
            item.id = Some(id.to_string());
            let replicas = self.replicas_for_id(id);
            for &node in &replicas {
                groups.entry(node).or_default().push(item.clone());
            }
            ids.push(id.to_string());
            replica_sets.push((id.to_string(), replicas));
        }

        // Send each node's sub-batch and track which nodes acked.
        let mut failed: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for (node, items) in groups {
            let req = proto::BatchInsertRequest { items };
            match self.nodes[node].batch_insert(req).await {
                Ok(_) => self.set_health(node, true),
                Err(e) => {
                    self.set_health(node, false);
                    warn!("batch_insert on node {node} failed: {e}");
                    failed.insert(node);
                }
            }
        }

        // Enforce the write quorum per object (same durability contract as the
        // single `insert`): every item must have reached `write_quorum` replicas.
        for (id, replicas) in &replica_sets {
            let quorum = self.write_quorum.clamp(1, replicas.len().max(1));
            let acks = replicas.iter().filter(|r| !failed.contains(r)).count();
            if acks < quorum {
                return Err(Status::unavailable(format!(
                    "batch_insert: write quorum not met for {id} ({acks}/{quorum} acks)"
                )));
            }
        }
        Ok(proto::BatchInsertResponse { ids })
    }

    /// Run `f` on each replica; succeed once `write_quorum` replicas ack.
    async fn write_to_replicas<F, Fut>(&self, replicas: &[usize], f: F) -> Result<(), Status>
    where
        F: Fn(Arc<dyn ClusterNode>) -> Fut,
        Fut: std::future::Future<Output = Result<(), Status>>,
    {
        let quorum = self.write_quorum.clamp(1, replicas.len().max(1));
        let mut acks = 0usize;
        let mut last_err = None;
        for &i in replicas {
            match f(self.nodes[i].clone()).await {
                Ok(()) => {
                    self.set_health(i, true);
                    acks += 1;
                }
                Err(e) => {
                    self.set_health(i, false);
                    warn!("write to replica {i} failed: {e}");
                    last_err = Some(e);
                }
            }
        }
        if acks >= quorum {
            Ok(())
        } else {
            Err(last_err.unwrap_or_else(|| {
                Status::unavailable(format!("write quorum not met: {acks}/{quorum} acks"))
            }))
        }
    }

    // ---- Id-keyed ops ----

    /// Read an object, with **read-repair + last-write-wins reconciliation**:
    /// query all healthy replicas, return the freshest copy (latest
    /// `updated_at`), and write that copy back to any replica that is missing it
    /// *or holds a staler/divergent copy* — so the cluster converges on the most
    /// recent version after a partial write or a divergence.
    pub async fn get(&self, req: proto::GetRequest) -> Result<proto::GetResponse, Status> {
        let id = parse_id(&req.id)?;
        let replicas: Vec<usize> = self
            .replicas_for_id(id)
            .into_iter()
            .filter(|&i| self.is_healthy(i))
            .collect();

        let futures = replicas.iter().map(|&i| {
            let node = self.nodes[i].clone();
            let req = req.clone();
            async move { (i, node.get(req).await) }
        });
        let outcomes = futures::future::join_all(futures).await;

        // Gather each replica's copy (if any) and the set that responded.
        let mut copies: Vec<(usize, proto::KnowledgeObject)> = Vec::new();
        let mut responded: Vec<usize> = Vec::new();
        for (i, outcome) in outcomes {
            match outcome {
                Ok(resp) => {
                    self.set_health(i, true);
                    responded.push(i);
                    if let Some(obj) = resp.object {
                        copies.push((i, obj));
                    }
                }
                Err(_) => self.set_health(i, false),
            }
        }

        // Last-write-wins: the freshest copy by `updated_at` (RFC3339 sorts
        // lexicographically; an empty timestamp is treated as oldest).
        let winner = copies
            .iter()
            .max_by(|a, b| a.1.updated_at.cmp(&b.1.updated_at))
            .map(|(_, o)| o.clone());

        if let Some(obj) = &winner {
            // Repair any responding replica whose copy is missing, staler, or
            // divergent in content from the winner.
            let stale: Vec<usize> = responded
                .iter()
                .copied()
                .filter(|i| match copies.iter().find(|(ci, _)| ci == i) {
                    Some((_, o)) => o.updated_at < obj.updated_at || o.content != obj.content,
                    None => true,
                })
                .collect();
            if !stale.is_empty() {
                let repair = knowledge_to_insert_req(obj);
                for i in stale {
                    if self.nodes[i].insert(repair.clone()).await.is_ok() {
                        debug!("read-repair: reconciled {} on node {i}", obj.id);
                    }
                }
            }
        }
        Ok(proto::GetResponse { object: winner })
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
            .await?;

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
            .await?;

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
        // Per-node SQL over each partition. A top-level scalar aggregate
        // (COUNT/SUM/MIN/MAX, no GROUP BY) is *combined* across shards into one
        // global row; everything else is concatenated. (AVG and GROUP BY aren't
        // mergeable from partials alone and are concatenated — see
        // `merge_sql_aggregate`.)
        let responses = self
            .scatter(|n| {
                let r = req.clone();
                async move { n.sql(r).await }
            })
            .await?;
        let rows = match merge_sql_aggregate(&req.query, &responses) {
            Some(merged) => merged,
            None => responses.into_iter().flat_map(|r| r.rows).collect(),
        };
        Ok(proto::SqlResponse { rows })
    }

    /// Rebalance object placement to match the current partitioning — call after
    /// a **membership change** (nodes added/removed) so each object lives on the
    /// node(s) that now own its id. Misplaced objects are copied to their correct
    /// owner replica set and removed from nodes that no longer own them. Returns
    /// the number of objects relocated. Best-effort and idempotent: re-running on
    /// a balanced cluster moves nothing.
    pub async fn rebalance(&self) -> Result<usize, Status> {
        let mut moved = 0usize;
        for (i, node) in self.nodes.iter().enumerate() {
            if !self.is_healthy(i) {
                continue;
            }
            let listed = match node
                .list(proto::ListRequest {
                    offset: 0,
                    limit: u32::MAX,
                })
                .await
            {
                Ok(r) => r,
                Err(_) => {
                    self.set_health(i, false);
                    continue;
                }
            };
            for obj in listed.objects {
                let Ok(id) = parse_id(&obj.id) else { continue };
                let owners = self.replicas_for_id(id);
                if owners.contains(&i) {
                    continue; // correctly placed on this node
                }
                // Relocate: write to the correct owners, then drop from this node
                // — but ONLY if at least one owner write succeeded, so a transient
                // owner failure can never delete the last copy (data loss).
                let insert = knowledge_to_insert_req(&obj);
                let mut acks = 0usize;
                for &o in &owners {
                    if self.nodes[o].insert(insert.clone()).await.is_ok() {
                        acks += 1;
                    }
                }
                if acks == 0 {
                    debug!(
                        "rebalance: no owner accepted {}; keeping source copy",
                        obj.id
                    );
                    continue;
                }
                if self.nodes[i]
                    .delete(proto::DeleteRequest { id: obj.id.clone() })
                    .await
                    .is_ok()
                {
                    moved += 1;
                    debug!("rebalance: moved {} off node {i}", obj.id);
                }
            }
        }
        info!("rebalance complete: {moved} object(s) relocated");
        Ok(moved)
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
            .await?;

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
            .await?;

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

    /// Run `f` against every **healthy** node in parallel, updating health from
    /// the outcomes. Tolerates partial failure (drops errored nodes), but errors
    /// if no healthy node responds — so callers can tell "no matches" (empty Ok)
    /// from "cluster unavailable" (Err).
    async fn scatter<F, Fut, T>(&self, f: F) -> Result<Vec<T>, Status>
    where
        F: Fn(Arc<dyn ClusterNode>) -> Fut,
        Fut: std::future::Future<Output = Result<T, Status>>,
    {
        let candidates: Vec<usize> = (0..self.nodes.len())
            .filter(|&i| self.is_healthy(i))
            .collect();
        if candidates.is_empty() && !self.nodes.is_empty() {
            return Err(Status::unavailable("no healthy cluster nodes"));
        }

        let futures = candidates.iter().map(|&i| {
            let fut = f(self.nodes[i].clone());
            async move { (i, fut.await) }
        });
        let outcomes = futures::future::join_all(futures).await;
        let attempted = outcomes.len();

        let mut oks = Vec::with_capacity(attempted);
        let mut last_err = None;
        for (i, outcome) in outcomes {
            match outcome {
                Ok(v) => {
                    self.set_health(i, true);
                    oks.push(v);
                }
                Err(e) => {
                    self.set_health(i, false);
                    last_err = Some(e);
                }
            }
        }
        if oks.is_empty() && attempted > 0 {
            return Err(last_err.unwrap_or_else(|| Status::unavailable("all cluster nodes failed")));
        }
        Ok(oks)
    }
}

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

fn parse_id(s: &str) -> Result<ObjectId, Status> {
    ObjectId::parse_str(s).map_err(|_| Status::invalid_argument("invalid object id"))
}

/// Combine per-shard partials of a top-level scalar aggregate into one global
/// row. Returns `None` (caller concatenates) unless the query is a single
/// COUNT/SUM/MIN/MAX with no GROUP BY and every shard returned exactly one
/// single-column row. AVG can't be merged from partials alone, so it is not
/// combined here.
fn merge_sql_aggregate(
    query: &str,
    responses: &[proto::SqlResponse],
) -> Option<Vec<proto::SqlRow>> {
    let q = query.to_lowercase();
    if q.contains("group by") {
        return None;
    }
    let op = ["count(", "sum(", "min(", "max("]
        .into_iter()
        .find(|kw| q.contains(*kw))?;
    // AVG present alongside disqualifies a clean single-aggregate merge.
    if q.contains("avg(") {
        return None;
    }

    let mut col_name: Option<String> = None;
    let mut values: Vec<f64> = Vec::new();
    for resp in responses {
        // Each shard must return exactly one single-column row to be mergeable.
        let [row] = resp.rows.as_slice() else {
            return None;
        };
        if row.columns.len() != 1 {
            return None;
        }
        let (k, v) = row.columns.iter().next().unwrap();
        col_name.get_or_insert_with(|| k.clone());
        values.push(v.parse::<f64>().ok()?);
    }
    if values.is_empty() {
        return None;
    }

    let combined = match op {
        "count(" | "sum(" => values.iter().sum(),
        "min(" => values.iter().copied().fold(f64::INFINITY, f64::min),
        "max(" => values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        _ => return None,
    };
    // Integer-valued results (counts) print without a trailing ".0".
    let val = if combined.fract() == 0.0 {
        format!("{}", combined as i64)
    } else {
        format!("{combined}")
    };
    let mut columns = HashMap::new();
    columns.insert(col_name?, val);
    Some(vec![proto::SqlRow { columns }])
}

/// Reconstruct an `InsertRequest` from a fetched object, to replay it onto a
/// replica during read-repair (the id is preserved).
fn knowledge_to_insert_req(obj: &proto::KnowledgeObject) -> proto::InsertRequest {
    proto::InsertRequest {
        id: Some(obj.id.clone()),
        content: obj.content.clone(),
        embeddings: obj.embeddings.clone(),
        metadata: obj.metadata.clone(),
        keywords: obj.keywords.clone(),
        relationships: obj
            .relationships
            .iter()
            .map(|r| proto::RelationshipInput {
                relation_type: r.relation_type.clone(),
                target_id: r.target_id.clone(),
                weight: r.weight,
            })
            .collect(),
        importance: obj.importance,
    }
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
    /// `fail` makes every call return Unavailable (simulating a downed node).
    #[derive(Default)]
    struct MockNode {
        // id -> (content, score, updated_at)
        objects: Mutex<HashMap<String, (String, f32, String)>>,
        fail: std::sync::atomic::AtomicBool,
    }

    impl MockNode {
        fn set_fail(&self, f: bool) {
            self.fail.store(f, std::sync::atomic::Ordering::SeqCst);
        }
        /// Seed a copy with an explicit `updated_at` (for reconciliation tests).
        fn seed(&self, id: &str, content: &str, updated_at: &str) {
            self.objects.lock().insert(
                id.to_string(),
                (content.to_string(), 1.0, updated_at.to_string()),
            );
        }
        fn check(&self) -> Result<(), Status> {
            if self.fail.load(std::sync::atomic::Ordering::SeqCst) {
                Err(Status::unavailable("node down"))
            } else {
                Ok(())
            }
        }
    }

    #[tonic::async_trait]
    impl ClusterNode for MockNode {
        async fn insert(&self, req: proto::InsertRequest) -> Result<proto::InsertResponse, Status> {
            self.check()?;
            let id = req.id.clone().unwrap();
            self.objects.lock().insert(
                id.clone(),
                (req.content, req.importance.max(0.1), String::new()),
            );
            Ok(proto::InsertResponse { id })
        }
        async fn search(&self, req: proto::SearchRequest) -> Result<proto::SearchResponse, Status> {
            self.check()?;
            let results: Vec<_> = self
                .objects
                .lock()
                .iter()
                .filter(|(_, (content, _, _))| content.contains(&req.query))
                .map(|(id, (content, score, _))| proto::SearchResult {
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
            self.check()?;
            let object = self
                .objects
                .lock()
                .get(&req.id)
                .map(|(content, _, updated_at)| proto::KnowledgeObject {
                    id: req.id.clone(),
                    content: content.clone(),
                    updated_at: updated_at.clone(),
                    ..Default::default()
                });
            Ok(proto::GetResponse { object })
        }
        async fn delete(&self, req: proto::DeleteRequest) -> Result<proto::DeleteResponse, Status> {
            self.check()?;
            let existed = self.objects.lock().remove(&req.id).is_some();
            Ok(proto::DeleteResponse { existed })
        }
        async fn stats(&self, _req: proto::StatsRequest) -> Result<proto::StatsResponse, Status> {
            self.check()?;
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
            self.check()?;
            let objects: Vec<proto::KnowledgeObject> = self
                .objects
                .lock()
                .iter()
                .map(|(id, (content, _, updated_at))| proto::KnowledgeObject {
                    id: id.clone(),
                    content: content.clone(),
                    updated_at: updated_at.clone(),
                    ..Default::default()
                })
                .collect();
            Ok(proto::ListResponse {
                total: objects.len() as u64,
                objects,
            })
        }
        async fn ask(&self, _req: proto::AskRequest) -> Result<proto::AskResponse, Status> {
            Ok(proto::AskResponse::default())
        }
        async fn sql(&self, _req: proto::SqlRequest) -> Result<proto::SqlResponse, Status> {
            self.check()?;
            // Simulate this shard answering `SELECT COUNT(*)` with its partial.
            let mut columns = HashMap::new();
            columns.insert(
                "count(*)".to_string(),
                self.objects.lock().len().to_string(),
            );
            Ok(proto::SqlResponse {
                rows: vec![proto::SqlRow { columns }],
            })
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

    #[tokio::test]
    async fn test_write_quorum() {
        let mocks: Vec<Arc<MockNode>> = (0..3).map(|_| Arc::new(MockNode::default())).collect();
        let nodes: Vec<Arc<dyn ClusterNode>> = mocks.iter().map(|m| m.clone() as _).collect();
        let cluster = Cluster::new(nodes, 3).with_write_quorum(2); // rf=3, W=2

        let mk = |c: &str| proto::InsertRequest {
            content: c.into(),
            ..Default::default()
        };

        // All replicas healthy → write succeeds.
        assert!(cluster.insert(mk("a")).await.is_ok());

        // Two replicas down → only 1 ack < quorum(2) → write fails.
        mocks[1].set_fail(true);
        mocks[2].set_fail(true);
        assert!(
            cluster.insert(mk("b")).await.is_err(),
            "write must fail when the quorum is not met"
        );

        // One replica back → 2 acks ≥ quorum → write succeeds again.
        mocks[2].set_fail(false);
        assert!(cluster.insert(mk("c")).await.is_ok());
    }

    #[tokio::test]
    async fn test_reads_tolerate_partial_failure() {
        let (cluster, mocks) = cluster(3, 1);
        for i in 0..9 {
            cluster
                .insert(proto::InsertRequest {
                    content: format!("widget {i}"),
                    ..Default::default()
                })
                .await
                .unwrap();
        }
        let query = || proto::SearchRequest {
            query: "widget".into(),
            top_k: 50,
            ..Default::default()
        };

        // One node down → search still returns results from the live nodes.
        mocks[0].set_fail(true);
        assert!(
            !cluster.search(query()).await.unwrap().results.is_empty(),
            "a single node failure must be tolerated"
        );

        // Every node down → search errors (distinguishes an outage from "no
        // matches", which is an empty Ok).
        for m in &mocks {
            m.set_fail(true);
        }
        assert!(
            cluster.search(query()).await.is_err(),
            "total outage must surface as an error"
        );
    }

    #[tokio::test]
    async fn test_health_gating_and_recovery() {
        let (cluster, mocks) = cluster(3, 1);
        // Pin ids 0..6 → nodes 0,1,2,0,1,2 (2 objects per node) for determinism.
        for k in 0..6u128 {
            cluster
                .insert(proto::InsertRequest {
                    content: format!("widget {k}"),
                    id: Some(uuid::Uuid::from_u128(k).to_string()),
                    ..Default::default()
                })
                .await
                .unwrap();
        }
        let query = || proto::SearchRequest {
            query: "widget".into(),
            top_k: 50,
            ..Default::default()
        };

        assert_eq!(cluster.search(query()).await.unwrap().results.len(), 6);
        assert_eq!(cluster.healthy_count(), 3);

        // Mark node 0 unhealthy → its 2 objects are skipped on reads.
        cluster.set_health(0, false);
        assert_eq!(cluster.healthy_count(), 2);
        assert_eq!(cluster.search(query()).await.unwrap().results.len(), 4);

        // Heartbeat probes the (healthy) mocks → node 0 recovers → all visible.
        cluster.heartbeat_once().await;
        assert_eq!(cluster.healthy_count(), 3);
        assert_eq!(cluster.search(query()).await.unwrap().results.len(), 6);

        // A genuinely-down node is detected by the heartbeat and recovered on fix.
        mocks[1].set_fail(true);
        cluster.heartbeat_once().await;
        assert!(!cluster.is_healthy(1));
        mocks[1].set_fail(false);
        cluster.heartbeat_once().await;
        assert!(cluster.is_healthy(1));
    }

    #[tokio::test]
    async fn test_read_repair() {
        // 3 replicas (rf=3=n), write_quorum=1 → a write can land on just 1 node.
        let mocks: Vec<Arc<MockNode>> = (0..3).map(|_| Arc::new(MockNode::default())).collect();
        let nodes: Vec<Arc<dyn ClusterNode>> = mocks.iter().map(|m| m.clone() as _).collect();
        let cluster = Cluster::new(nodes, 3); // write_quorum defaults to 1
        let id = uuid::Uuid::from_u128(0).to_string();
        let has = |m: &Arc<MockNode>| m.objects.lock().contains_key(&id);

        // Two replicas down during the write → only node 0 stores it (quorum 1 ok).
        mocks[1].set_fail(true);
        mocks[2].set_fail(true);
        cluster
            .insert(proto::InsertRequest {
                content: "durable".into(),
                id: Some(id.clone()),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(mocks.iter().filter(|m| has(m)).count(), 1, "only one copy");

        // Heal the nodes; the heartbeat restores their health.
        mocks[1].set_fail(false);
        mocks[2].set_fail(false);
        cluster.heartbeat_once().await;

        // A read finds it on node 0, sees nodes 1 & 2 missing it → read-repair.
        assert!(cluster
            .get(proto::GetRequest { id: id.clone() })
            .await
            .unwrap()
            .object
            .is_some());
        assert_eq!(
            mocks.iter().filter(|m| has(m)).count(),
            3,
            "read-repair converged all replicas"
        );
    }

    #[tokio::test]
    async fn test_rebalance_relocates_misplaced_objects() {
        let mocks: Vec<Arc<MockNode>> = (0..3).map(|_| Arc::new(MockNode::default())).collect();
        let nodes: Vec<Arc<dyn ClusterNode>> = mocks.iter().map(|m| m.clone() as _).collect();
        let cluster = Cluster::new(nodes, 1); // rf=1: each id owned by one node

        // id 0 → owner node 0. Seed it on the WRONG node (1), as if a membership
        // change shifted ownership.
        let id = uuid::Uuid::from_u128(0).to_string();
        mocks[1].seed(&id, "misplaced", "");
        let owner = cluster.replicas_for_id(uuid::Uuid::from_u128(0))[0];
        assert_eq!(owner, 0, "id 0 is owned by node 0 under id % 3");

        let moved = cluster.rebalance().await.unwrap();
        assert_eq!(moved, 1, "the misplaced object is relocated");
        assert!(
            mocks[0].objects.lock().contains_key(&id),
            "now on the owner"
        );
        assert!(
            !mocks[1].objects.lock().contains_key(&id),
            "removed from wrong node"
        );

        // Idempotent: a second pass moves nothing.
        assert_eq!(cluster.rebalance().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_distributed_aggregate_count_is_combined() {
        let (cluster, _mocks) = cluster(3, 1);
        // 7 objects spread across 3 shards by id.
        for k in 0..7u128 {
            cluster
                .insert(proto::InsertRequest {
                    content: format!("row {k}"),
                    id: Some(uuid::Uuid::from_u128(k).to_string()),
                    ..Default::default()
                })
                .await
                .unwrap();
        }
        let resp = cluster
            .sql(proto::SqlRequest {
                query: "SELECT COUNT(*) FROM knowledge".into(),
            })
            .await
            .unwrap();
        // The per-shard partial counts are summed into one global row.
        assert_eq!(resp.rows.len(), 1, "aggregate collapses to one row");
        assert_eq!(
            resp.rows[0].columns.get("count(*)").map(String::as_str),
            Some("7")
        );
    }

    #[tokio::test]
    async fn test_last_write_wins_reconciliation() {
        let mocks: Vec<Arc<MockNode>> = (0..3).map(|_| Arc::new(MockNode::default())).collect();
        let nodes: Vec<Arc<dyn ClusterNode>> = mocks.iter().map(|m| m.clone() as _).collect();
        let cluster = Cluster::new(nodes, 3);
        let id = uuid::Uuid::from_u128(0).to_string();

        // Divergent copies: node 0 stale, node 1 freshest, node 2 missing.
        mocks[0].seed(&id, "old version", "2026-01-01T00:00:00Z");
        mocks[1].seed(&id, "new version", "2026-06-01T00:00:00Z");

        let got = cluster
            .get(proto::GetRequest { id: id.clone() })
            .await
            .unwrap()
            .object
            .expect("object present");
        assert_eq!(got.content, "new version", "returns the freshest copy");

        // All replicas converge on the freshest content.
        for m in &mocks {
            assert_eq!(
                m.objects.lock().get(&id).map(|(c, _, _)| c.clone()),
                Some("new version".to_string()),
                "every replica reconciled to the latest version"
            );
        }
    }
}
