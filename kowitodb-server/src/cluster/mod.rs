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

mod coordinator;
mod node;
mod service;
#[cfg(test)]
mod tests;

pub use coordinator::Cluster;
pub use service::ClusterService;

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
