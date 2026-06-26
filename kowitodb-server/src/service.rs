use tonic::{Request, Response, Status};
use tracing::info;

use crate::db::KowitoDBEngine;
use crate::proto;
use crate::proto::kowito_db_server::KowitoDb;

/// gRPC service wrapping the KowitoDB engine.
pub struct KowitoDBService {
    engine: KowitoDBEngine,
}

impl KowitoDBService {
    pub fn new(engine: KowitoDBEngine) -> Self {
        Self { engine }
    }
}

#[tonic::async_trait]
impl KowitoDb for KowitoDBService {
    async fn insert(
        &self,
        request: Request<proto::InsertRequest>,
    ) -> Result<Response<proto::InsertResponse>, Status> {
        let req = request.into_inner();
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

        let id = self
            .engine
            .insert(obj)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        info!("Insert request completed: {}", id);
        Ok(Response::new(proto::InsertResponse { id: id.to_string() }))
    }

    async fn get(
        &self,
        request: Request<proto::GetRequest>,
    ) -> Result<Response<proto::GetResponse>, Status> {
        let req = request.into_inner();
        let id =
            uuid::Uuid::parse_str(&req.id).map_err(|_| Status::invalid_argument("Invalid UUID"))?;

        let obj = self
            .engine
            .get(id)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(match obj {
            Some(o) => proto::GetResponse {
                object: Some(proto::KnowledgeObject {
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
                }),
            },
            None => proto::GetResponse { object: None },
        }))
    }

    async fn delete(
        &self,
        request: Request<proto::DeleteRequest>,
    ) -> Result<Response<proto::DeleteResponse>, Status> {
        let req = request.into_inner();
        let id =
            uuid::Uuid::parse_str(&req.id).map_err(|_| Status::invalid_argument("Invalid UUID"))?;

        let existed = self
            .engine
            .delete(id)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(proto::DeleteResponse { existed }))
    }

    async fn search(
        &self,
        request: Request<proto::SearchRequest>,
    ) -> Result<Response<proto::SearchResponse>, Status> {
        let req = request.into_inner();
        let max_results = req.top_k.max(1).min(100) as usize;

        let response = self
            .engine
            .ask(&req.query, max_results)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

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

        let total = results.len() as i32;

        Ok(Response::new(proto::SearchResponse {
            results,
            plan_explanation: response.plan_explanation,
            total_found: total,
        }))
    }

    /// The core AI API: `ai.ask(question)`
    async fn ask(
        &self,
        request: Request<proto::AskRequest>,
    ) -> Result<Response<proto::AskResponse>, Status> {
        let req = request.into_inner();
        let max_results = req.max_results.max(1).min(100) as usize;

        info!("ai.ask(): \"{}\"", req.question);

        let response = self
            .engine
            .ask(&req.question, max_results)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(proto::AskResponse {
            results: response.results,
            plan_explanation: response.plan_explanation,
            detected_intent: response.detected_intent,
        }))
    }

    /// The `ai.remember()` API: simplified insert with auto-embedding.
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

        let id = self
            .engine
            .insert(obj)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        info!("ai.remember(): stored {}", id);
        Ok(Response::new(proto::RememberResponse {
            id: id.to_string(),
        }))
    }

    async fn stats(
        &self,
        _request: Request<proto::StatsRequest>,
    ) -> Result<Response<proto::StatsResponse>, Status> {
        let stats = self
            .engine
            .stats()
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(proto::StatsResponse {
            total_objects: stats.total_objects,
            vector_count: stats.vector_count,
            index_size_bytes: stats.index_size_bytes,
        }))
    }
}
