use std::sync::Arc;
use std::time::Instant;

use tonic::{Request, Response, Status};
use tracing::info;

use crate::db::KowitoDBEngine;
use crate::metrics::MetricsCollector;
use crate::proto;
use crate::proto::kowito_db_server::KowitoDb;

pub struct KowitoDBService {
    engine: KowitoDBEngine,
    metrics: Arc<MetricsCollector>,
    /// Upper bound on results returned by Ask/Search.
    max_results: i32,
}

impl KowitoDBService {
    pub fn new(engine: KowitoDBEngine, metrics: Arc<MetricsCollector>, max_results: usize) -> Self {
        Self {
            engine,
            metrics,
            max_results: max_results.clamp(1, i32::MAX as usize) as i32,
        }
    }
}

#[tonic::async_trait]
impl KowitoDb for KowitoDBService {
    async fn insert(
        &self,
        request: Request<proto::InsertRequest>,
    ) -> Result<Response<proto::InsertResponse>, Status> {
        let req = request.into_inner();
        let obj = insert_req_to_obj(req);

        match self.engine.insert(obj).await {
            Ok(id) => {
                self.metrics.record_insert();
                info!("Insert: {}", id);
                Ok(Response::new(proto::InsertResponse { id: id.to_string() }))
            }
            Err(e) => {
                self.metrics.record_error();
                Err(Status::internal(e.to_string()))
            }
        }
    }

    async fn get(
        &self,
        request: Request<proto::GetRequest>,
    ) -> Result<Response<proto::GetResponse>, Status> {
        let req = request.into_inner();
        let id =
            uuid::Uuid::parse_str(&req.id).map_err(|_| Status::invalid_argument("Invalid UUID"))?;

        let obj = self.engine.get(id).await.map_err(|e| {
            self.metrics.record_error();
            Status::internal(e.to_string())
        })?;

        Ok(Response::new(proto::GetResponse {
            object: obj.map(knowledge_to_proto),
        }))
    }

    async fn batch_insert(
        &self,
        request: Request<proto::BatchInsertRequest>,
    ) -> Result<Response<proto::BatchInsertResponse>, Status> {
        let req = request.into_inner();
        let objects: Vec<_> = req.items.into_iter().map(insert_req_to_obj).collect();
        let count = objects.len();

        let ids = self.engine.batch_insert(objects).await.map_err(|e| {
            self.metrics.record_error();
            Status::internal(e.to_string())
        })?;

        for _ in 0..count {
            self.metrics.record_insert();
        }
        info!("BatchInsert: {} objects", count);
        Ok(Response::new(proto::BatchInsertResponse {
            ids: ids.into_iter().map(|id| id.to_string()).collect(),
        }))
    }

    async fn list(
        &self,
        request: Request<proto::ListRequest>,
    ) -> Result<Response<proto::ListResponse>, Status> {
        let req = request.into_inner();
        let limit = if req.limit == 0 {
            self.max_results as usize
        } else {
            req.limit as usize
        };

        let (objects, total) = self
            .engine
            .list(req.offset as usize, limit)
            .await
            .map_err(|e| {
                self.metrics.record_error();
                Status::internal(e.to_string())
            })?;

        Ok(Response::new(proto::ListResponse {
            objects: objects.into_iter().map(knowledge_to_proto).collect(),
            total: total as u64,
        }))
    }

    async fn delete(
        &self,
        request: Request<proto::DeleteRequest>,
    ) -> Result<Response<proto::DeleteResponse>, Status> {
        let req = request.into_inner();
        let id =
            uuid::Uuid::parse_str(&req.id).map_err(|_| Status::invalid_argument("Invalid UUID"))?;

        let existed = self.engine.delete(id).await.map_err(|e| {
            self.metrics.record_error();
            Status::internal(e.to_string())
        })?;

        Ok(Response::new(proto::DeleteResponse { existed }))
    }

    async fn search(
        &self,
        request: Request<proto::SearchRequest>,
    ) -> Result<Response<proto::SearchResponse>, Status> {
        let req = request.into_inner();
        let max_results = req.top_k.clamp(1, self.max_results) as usize;

        let start = Instant::now();
        let response = self
            .engine
            .ask_filtered(&req.query, max_results, None, &req.metadata_filter)
            .await
            .map_err(|e| {
                self.metrics.record_error();
                Status::internal(e.to_string())
            })?;
        self.metrics.record_ask(start.elapsed());

        let results: Vec<proto::SearchResult> = response
            .results
            .into_iter()
            .map(|r| proto::SearchResult {
                id: r.id,
                content: r.content,
                score: r.relevance_score,
                metadata: r.metadata,
            })
            .collect();

        Ok(Response::new(proto::SearchResponse {
            total_found: results.len() as i32,
            results,
            plan_explanation: response.plan_explanation,
        }))
    }

    async fn ask(
        &self,
        request: Request<proto::AskRequest>,
    ) -> Result<Response<proto::AskResponse>, Status> {
        let req = request.into_inner();
        let max_results = req.max_results.clamp(1, self.max_results) as usize;
        let budget = (req.max_context_tokens > 0).then_some(req.max_context_tokens as usize);

        info!("ai.ask(): \"{}\"", req.question);

        let start = Instant::now();
        let response = self
            .engine
            .ask_filtered(&req.question, max_results, budget, &req.metadata_filter)
            .await
            .map_err(|e| {
                self.metrics.record_error();
                Status::internal(e.to_string())
            })?;
        self.metrics.record_ask(start.elapsed());

        Ok(Response::new(proto::AskResponse {
            results: response.results,
            plan_explanation: response.plan_explanation,
            detected_intent: response.detected_intent,
        }))
    }

    async fn remember(
        &self,
        request: Request<proto::RememberRequest>,
    ) -> Result<Response<proto::RememberResponse>, Status> {
        let req = request.into_inner();
        let mut obj = kowitodb_core::KnowledgeObject::new(req.content);

        for (model, vec_proto) in req.embeddings {
            obj.embeddings.insert(model, vec_proto.values);
        }
        for (k, v) in req.metadata {
            obj.metadata.insert(k, serde_json::Value::String(v));
        }
        obj.keywords = req.keywords;
        obj.importance = req.importance.clamp(0.0, 1.0);

        let id = self.engine.insert(obj).await.map_err(|e| {
            self.metrics.record_error();
            Status::internal(e.to_string())
        })?;

        self.metrics.record_remember();
        info!("ai.remember(): stored {}", id);
        Ok(Response::new(proto::RememberResponse {
            id: id.to_string(),
        }))
    }

    async fn stats(
        &self,
        _request: Request<proto::StatsRequest>,
    ) -> Result<Response<proto::StatsResponse>, Status> {
        let stats = self.engine.stats().await.map_err(|e| {
            self.metrics.record_error();
            Status::internal(e.to_string())
        })?;

        let (cache_entries, cache_hit_rate) = stats
            .cache_stats
            .as_ref()
            .map(|c| (c.entries as u64, c.hit_rate as f64))
            .unwrap_or((0, 0.0));

        Ok(Response::new(proto::StatsResponse {
            total_objects: stats.total_objects,
            vector_count: stats.vector_count,
            index_size_bytes: stats.index_size_bytes,
            graph_nodes: stats.graph_nodes,
            graph_edges: stats.graph_edges,
            active_agent_sessions: stats.active_agent_sessions,
            total_cost_usd: stats.total_cost_usd,
            cache_entries,
            cache_hit_rate,
        }))
    }

    async fn update(
        &self,
        request: Request<proto::UpdateRequest>,
    ) -> Result<Response<proto::UpdateResponse>, Status> {
        let req = request.into_inner();
        let id =
            uuid::Uuid::parse_str(&req.id).map_err(|_| Status::invalid_argument("Invalid UUID"))?;

        let version = self
            .engine
            .update(
                id,
                req.content,
                req.metadata,
                req.keywords,
                req.importance,
                req.change_description,
            )
            .await
            .map_err(|e| {
                self.metrics.record_error();
                Status::internal(e.to_string())
            })?;

        Ok(Response::new(match version {
            Some(v) => proto::UpdateResponse {
                updated: true,
                version: v as u32,
            },
            None => proto::UpdateResponse {
                updated: false,
                version: 0,
            },
        }))
    }

    async fn sql(
        &self,
        request: Request<proto::SqlRequest>,
    ) -> Result<Response<proto::SqlResponse>, Status> {
        let req = request.into_inner();
        let rows = self.engine.sql_select(&req.query).await.map_err(|e| {
            self.metrics.record_error();
            Status::invalid_argument(e.to_string())
        })?;
        self.metrics.record_sql();

        Ok(Response::new(proto::SqlResponse {
            rows: rows
                .into_iter()
                .map(|columns| proto::SqlRow { columns })
                .collect(),
        }))
    }

    async fn record_turn(
        &self,
        request: Request<proto::RecordTurnRequest>,
    ) -> Result<Response<proto::RecordTurnResponse>, Status> {
        use crate::memory::TurnRole;
        let req = request.into_inner();
        let role = match req.role.to_lowercase().as_str() {
            "assistant" => TurnRole::Assistant,
            "system" => TurnRole::System,
            "observation" => TurnRole::Observation,
            _ => TurnRole::User,
        };

        let mut session = self.engine.agent_memory.get_or_create(&req.session_id);
        session.add_turn(role, req.content);
        let turn_count = session.turn_count() as u32;
        self.engine.agent_memory.save(session);

        Ok(Response::new(proto::RecordTurnResponse { turn_count }))
    }

    async fn get_session(
        &self,
        request: Request<proto::GetSessionRequest>,
    ) -> Result<Response<proto::GetSessionResponse>, Status> {
        let req = request.into_inner();
        Ok(Response::new(
            match self.engine.agent_memory.get(&req.session_id) {
                Some(session) => proto::GetSessionResponse {
                    found: true,
                    turns: session
                        .turns
                        .into_iter()
                        .map(|t| proto::ConversationTurnProto {
                            role: format!("{:?}", t.role).to_lowercase(),
                            content: t.content,
                            timestamp: t.timestamp.to_rfc3339(),
                        })
                        .collect(),
                },
                None => proto::GetSessionResponse {
                    found: false,
                    turns: Vec::new(),
                },
            },
        ))
    }
}

/// Build a `KnowledgeObject` from an `InsertRequest` (shared by Insert/BatchInsert).
fn insert_req_to_obj(req: proto::InsertRequest) -> kowitodb_core::KnowledgeObject {
    let mut obj = kowitodb_core::KnowledgeObject::new(req.content);

    for (model, vec_proto) in req.embeddings {
        obj.embeddings.insert(model, vec_proto.values);
    }
    for (k, v) in req.metadata {
        obj.metadata.insert(k, serde_json::Value::String(v));
    }
    obj.keywords = req.keywords;
    for rel in req.relationships {
        if let Ok(target_id) = uuid::Uuid::parse_str(&rel.target_id) {
            obj.relationships.push(kowitodb_core::Relationship {
                relation_type: rel.relation_type,
                target_id,
                weight: rel.weight,
            });
        }
    }
    if req.importance > 0.0 {
        obj.importance = req.importance.clamp(0.0, 1.0);
    }
    obj
}

/// Convert a `KnowledgeObject` to its proto form (shared by Get/List).
fn knowledge_to_proto(o: kowitodb_core::KnowledgeObject) -> proto::KnowledgeObject {
    proto::KnowledgeObject {
        id: o.id.to_string(),
        content: o.content,
        embeddings: o
            .embeddings
            .into_iter()
            .map(|(k, v)| (k, proto::EmbeddingVector { values: v }))
            .collect(),
        metadata: o
            .metadata
            .into_iter()
            .map(|(k, v)| (k, v.to_string()))
            .collect(),
        keywords: o.keywords,
        relationships: o
            .relationships
            .into_iter()
            .map(|r| proto::Relationship {
                relation_type: r.relation_type,
                target_id: r.target_id.to_string(),
                weight: r.weight,
            })
            .collect(),
        importance: o.importance,
        created_at: o.created_at.to_rfc3339(),
        updated_at: o.updated_at.to_rfc3339(),
    }
}
