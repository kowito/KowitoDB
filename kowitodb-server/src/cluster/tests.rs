//! Cluster tests (split from the former cluster.rs god-file).

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
    async fn update(&self, _req: proto::UpdateRequest) -> Result<proto::UpdateResponse, Status> {
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
