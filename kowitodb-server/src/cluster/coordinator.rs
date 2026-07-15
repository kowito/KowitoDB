//! Cluster — the Cluster coordinator (scatter-gather, quorum, rebalance) (split from the former cluster.rs god-file).
#![allow(clippy::result_large_err)]

use super::node::RemoteNode;
use super::*;

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

    pub(crate) fn is_healthy(&self, i: usize) -> bool {
        self.health[i].load(Ordering::Relaxed)
    }

    /// Update a node's health, logging up/down transitions.
    pub(crate) fn set_health(&self, i: usize, healthy: bool) {
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

    pub(crate) fn replicas(&self, owner: usize) -> Vec<usize> {
        let n = self.nodes.len();
        (0..self.replication_factor)
            .map(|i| (owner + i) % n)
            .collect()
    }

    pub(crate) fn replicas_for_id(&self, id: ObjectId) -> Vec<usize> {
        self.replicas((id.as_u128() % self.nodes.len() as u128) as usize)
    }

    pub(crate) fn replicas_for_key(&self, key: &str) -> Vec<usize> {
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
    pub(crate) async fn write_to_replicas<F, Fut>(
        &self,
        replicas: &[usize],
        f: F,
    ) -> Result<(), Status>
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
    pub(crate) async fn scatter<F, Fut, T>(&self, f: F) -> Result<Vec<T>, Status>
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
