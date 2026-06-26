use std::collections::HashMap;
use std::sync::Arc;

use kowitodb_core::Result as KResult;
use kowitodb_core::{Embedding, KnowledgeObject, ObjectId};
use kowitodb_index::{
    FullTextIndex, GraphIndex, IndexResult, IndexSource, MetadataIndex, TimeIndex, VectorIndex,
};
use kowitodb_planner::{
    cache::QueryCache, context::ContextOptimizer, cost::CostTracker, reranker::Reranker,
    DetectedIntent, ExecutionPlan, QueryPlanner, RankedResult,
};
use kowitodb_storage::{StorageBackend, StorageEngine, StoredObject};
use tracing::{debug, info};

use crate::embedding::{EmbeddingClient, ProxyEmbeddingClient};
use crate::memory::AgentMemory;
use crate::proto;

/// Core engine wiring storage, all 6 indexes, query planner, and all optimizers.
pub struct KowitoDBEngine {
    pub storage: Arc<StorageEngine>,
    pub vector_index: Arc<VectorIndex>,
    pub fulltext_index: Arc<FullTextIndex>,
    pub metadata_index: Arc<MetadataIndex>,
    pub time_index: Arc<TimeIndex>,
    pub graph_index: Arc<GraphIndex>,
    pub planner: Arc<QueryPlanner>,
    pub reranker: Arc<Reranker>,
    pub context_optimizer: Arc<ContextOptimizer>,
    pub cost_tracker: Arc<CostTracker>,
    pub agent_memory: Arc<AgentMemory>,
    pub embedding_client: Arc<dyn EmbeddingClient>,
    pub plan_cache: Arc<QueryCache<(DetectedIntent, ExecutionPlan)>>,
    default_model: String,
}

impl KowitoDBEngine {
    pub fn new(
        storage_path: impl AsRef<std::path::Path>,
        index_path: impl AsRef<std::path::Path>,
    ) -> KResult<Self> {
        let storage = StorageEngine::open(storage_path)?;
        let fulltext_index = FullTextIndex::open(index_path)?;
        let vector_index = VectorIndex::new();
        let metadata_index = MetadataIndex::new();
        let time_index = TimeIndex::new();
        let graph_index = GraphIndex::new();
        let planner = QueryPlanner::new();
        let reranker = Reranker::new();
        let context_optimizer = ContextOptimizer::new(4096);
        let cost_tracker = CostTracker::new();
        let agent_memory = AgentMemory::new();
        let embedding_client: Arc<dyn EmbeddingClient> =
            Arc::new(ProxyEmbeddingClient::new("proxy-text-embedding", 128));
        let plan_cache: QueryCache<(DetectedIntent, ExecutionPlan)> = QueryCache::new(300, 1000);

        info!("KowitoDB engine initialized with all subsystems");
        Ok(Self {
            storage: Arc::new(storage),
            vector_index: Arc::new(vector_index),
            fulltext_index: Arc::new(fulltext_index),
            metadata_index: Arc::new(metadata_index),
            time_index: Arc::new(time_index),
            graph_index: Arc::new(graph_index),
            planner: Arc::new(planner),
            reranker: Arc::new(reranker),
            context_optimizer: Arc::new(context_optimizer),
            cost_tracker: Arc::new(cost_tracker),
            agent_memory: Arc::new(agent_memory),
            embedding_client,
            plan_cache: Arc::new(plan_cache),
            default_model: "default".to_string(),
        })
    }

    /// Insert a knowledge object into storage and all 6 indexes.
    pub async fn insert(&self, obj: KnowledgeObject) -> KResult<ObjectId> {
        let id = obj.id;
        debug!("Inserting knowledge object {}", id);

        // Index vectors (auto-embed if needed)
        for (model, embedding) in &obj.embeddings {
            self.vector_index.insert(id, model, embedding.clone())?;
        }
        if obj.embeddings.is_empty() && !obj.content.is_empty() {
            if let Ok(result) = self.embedding_client.embed(&obj.content).await {
                self.vector_index.insert(id, &result.model, result.vector)?;
                self.cost_tracker.record_embedding_calls(1);
            }
        }

        // Full-text index
        self.fulltext_index.insert(
            id,
            &obj.content,
            &obj.keywords,
            &serde_json::to_string(&obj.metadata).unwrap_or_default(),
        )?;

        // Metadata index
        for (key, value) in &obj.metadata {
            let val_str = match value {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            self.metadata_index.insert(id, key, &val_str);
        }

        // Time index
        self.time_index
            .insert(id, obj.created_at.timestamp_millis());

        // Graph index (relationships)
        if !obj.relationships.is_empty() {
            self.graph_index
                .insert_relationships(id, &obj.relationships);
        }

        // Persist to storage
        let stored = obj_to_stored(&obj)?;
        self.storage.put(stored).await?;

        // Invalidate caches
        self.plan_cache.clear();

        info!(
            "Inserted {}: {} (vectors={}, keywords={}, rels={})",
            id,
            &obj.content[..obj.content.len().min(80)],
            obj.embeddings.len(),
            obj.keywords.len(),
            obj.relationships.len(),
        );
        Ok(id)
    }

    /// Retrieve by ID.
    pub async fn get(&self, id: ObjectId) -> KResult<Option<KnowledgeObject>> {
        match self.storage.get(id).await? {
            Some(stored) => Ok(Some(stored_to_obj(&stored)?)),
            None => Ok(None),
        }
    }

    /// Delete from all indexes and storage.
    pub async fn delete(&self, id: ObjectId) -> KResult<bool> {
        self.vector_index.remove(id);
        let _ = self.fulltext_index.remove(id);
        self.metadata_index.remove_object(id);
        self.time_index.remove(id);
        self.graph_index.remove_object(id);

        let existed = self.storage.delete(id).await?;
        if existed {
            self.plan_cache.clear();
            info!("Deleted {}", id);
        }
        Ok(existed)
    }

    /// The core `ai.ask()` method — full pipeline.
    pub async fn ask(&self, question: &str, max_results: usize) -> KResult<AskResponse> {
        // Check plan cache
        let (intent, plan) = if let Some(cached) = self.plan_cache.get(question) {
            debug!("Plan cache hit for: {}", question);
            cached
        } else {
            let (intent, plan) = self.planner.plan(question);
            self.plan_cache
                .insert(question.to_string(), (intent.clone(), plan.clone()));
            (intent, plan)
        };

        // Execute plan against all indexes
        let raw_results = self.execute_plan(&plan, &intent).await?;
        self.cost_tracker.record_index_lookups(raw_results.len());

        // Graph traversal for entity/relationship queries
        let graph_results = self.execute_graph_traversal(&raw_results, &intent).await?;
        let all_results: Vec<IndexResult> = raw_results.into_iter().chain(graph_results).collect();

        // Rerank merged results (multi-source fusion + boosting)
        let ranked = self.reranker.rerank_simple(&all_results);

        // Assemble optimized context
        let assembled = self.assemble_context(&ranked);
        self.cost_tracker
            .record_llm_input_tokens(assembled.total_tokens);

        // Limit to requested max_results
        let limited: Vec<RankedResult> = ranked.into_iter().take(max_results).collect();

        Ok(AskResponse::from_parts(
            limited,
            plan.explain(),
            format!("{:?}", intent.intent),
            assembled,
        ))
    }

    /// Execute the planned retrieval steps.
    async fn execute_plan(
        &self,
        plan: &ExecutionPlan,
        intent: &DetectedIntent,
    ) -> KResult<Vec<IndexResult>> {
        let mut all_results: Vec<IndexResult> = Vec::new();
        let question = &plan.question;
        let keywords = &intent.entities.keywords;
        let dates = &intent.entities.dates;
        let metadata_filters = &intent.entities.metadata_filters;

        // Generate embedding
        let query_embedding: Option<Embedding> =
            self.embedding_client.embed(question).await.ok().map(|r| {
                self.cost_tracker.record_embedding_calls(1);
                r.vector
            });

        for step in &plan.steps {
            match step.step_type {
                kowitodb_planner::PlanStepType::VectorSearch => {
                    if let Some(ref emb) = query_embedding {
                        let results = self.vector_index.search(
                            emb,
                            &self.default_model,
                            step.limit.unwrap_or(20),
                        )?;
                        if !results.is_empty() {
                            let ids: Vec<_> = results.iter().map(|(id, _)| *id).collect();
                            let scores: Vec<_> = results.iter().map(|(_, s)| *s).collect();
                            all_results.push(IndexResult::new(ids, scores, IndexSource::Vector));
                        }
                    }
                }
                kowitodb_planner::PlanStepType::KeywordSearch => {
                    let query_str = if !keywords.is_empty() {
                        keywords.join(" ")
                    } else {
                        question.clone()
                    };
                    if !query_str.is_empty() {
                        if let Ok(results) = self
                            .fulltext_index
                            .search(&query_str, step.limit.unwrap_or(20))
                        {
                            if !results.is_empty() {
                                let ids: Vec<_> = results.iter().map(|(id, _)| *id).collect();
                                let scores: Vec<_> = results.iter().map(|(_, s)| *s).collect();
                                all_results.push(IndexResult::new(
                                    ids,
                                    scores,
                                    IndexSource::FullText,
                                ));
                            }
                        }
                    }
                }
                kowitodb_planner::PlanStepType::TimeFilter => {
                    if !dates.is_empty() {
                        let now_ms = chrono::Utc::now().timestamp_millis();
                        let ids = self.time_index.before(now_ms);
                        if !ids.is_empty() {
                            let scores = vec![1.0; ids.len()];
                            all_results.push(IndexResult::new(ids, scores, IndexSource::Time));
                        }
                    }
                }
                kowitodb_planner::PlanStepType::MetadataFilter => {
                    for (key, value) in metadata_filters {
                        let ids = self.metadata_index.query_exact(key, value);
                        if !ids.is_empty() {
                            let scores = vec![1.0; ids.len()];
                            all_results.push(IndexResult::new(ids, scores, IndexSource::Metadata));
                        }
                    }
                }
                _ => {
                    debug!("Deferred plan step: {:?}", step.step_type);
                }
            }
        }

        Ok(all_results)
    }

    /// Graph traversal for entity-heavy or relationship queries.
    async fn execute_graph_traversal(
        &self,
        raw_results: &[IndexResult],
        intent: &DetectedIntent,
    ) -> KResult<Vec<IndexResult>> {
        let mut seeds: Vec<ObjectId> = Vec::new();
        for result in raw_results {
            seeds.extend(&result.ids);
        }
        seeds.sort();
        seeds.dedup();

        if seeds.is_empty() {
            return Ok(Vec::new());
        }

        let max_depth = if matches!(intent.intent, kowitodb_planner::Intent::EntitySearch)
            || !intent.entities.named.is_empty()
        {
            2
        } else {
            1
        };

        let scored = self.graph_index.scored_traverse(&seeds, max_depth, None);
        if scored.is_empty() {
            return Ok(Vec::new());
        }

        // Exclude seeds (already in other results)
        let seed_set: std::collections::HashSet<ObjectId> = seeds.into_iter().collect();
        let new_nodes: Vec<_> = scored
            .into_iter()
            .filter(|(id, _)| !seed_set.contains(id))
            .collect();

        if new_nodes.is_empty() {
            return Ok(Vec::new());
        }

        let ids: Vec<_> = new_nodes.iter().map(|(id, _)| *id).collect();
        let scores: Vec<_> = new_nodes.iter().map(|(_, s)| *s).collect();

        debug!(
            "Graph: {} new nodes discovered via traversal",
            new_nodes.len()
        );

        Ok(vec![IndexResult::new(ids, scores, IndexSource::Graph)])
    }

    /// Assemble ranked results into token-optimized context.
    fn assemble_context(&self, ranked: &[RankedResult]) -> kowitodb_planner::AssembledContext {
        let content_lookup = |id: ObjectId| -> Option<String> {
            // Content loading from storage is async; for MVP return placeholder.
            // In production, this uses a blocking content cache or async context.
            Some(format!("[Object {}] — content loaded on demand", id))
        };
        self.context_optimizer.assemble(ranked, &content_lookup)
    }

    /// Comprehensive database stats.
    pub async fn stats(&self) -> KResult<StatsResponse> {
        Ok(StatsResponse {
            total_objects: self.storage.count().await? as u64,
            vector_count: self.vector_index.len() as u64,
            graph_nodes: self.graph_index.node_count() as u64,
            graph_edges: self.graph_index.edge_count() as u64,
            index_size_bytes: 0,
            cache_stats: Some(self.plan_cache.stats()),
            total_cost_usd: self.cost_tracker.total_cost(),
            active_agent_sessions: self.agent_memory.session_count() as u64,
        })
    }
}

// ---- Response types ----

#[derive(Debug, Clone)]
pub struct AskResponse {
    pub results: Vec<proto::AskResult>,
    pub plan_explanation: String,
    pub detected_intent: String,
    pub total_tokens: usize,
    pub compression_ratio: f32,
}

impl AskResponse {
    fn from_parts(
        ranked: Vec<RankedResult>,
        plan: String,
        intent: String,
        ctx: kowitodb_planner::AssembledContext,
    ) -> Self {
        let results: Vec<proto::AskResult> = ranked
            .into_iter()
            .map(|r| {
                let mut meta = HashMap::new();
                let sources_str = r
                    .sources
                    .iter()
                    .map(|s| format!("{:?}", s).to_lowercase())
                    .collect::<Vec<_>>()
                    .join(",");
                meta.insert("sources".to_string(), sources_str);

                proto::AskResult {
                    id: r.id.to_string(),
                    content: format!("<Object {}>", r.id),
                    relevance_score: r.score,
                    metadata: meta,
                    retrieval_source: r
                        .sources
                        .first()
                        .map(|s| format!("{:?}", s).to_lowercase())
                        .unwrap_or_default(),
                }
            })
            .collect();

        AskResponse {
            results,
            plan_explanation: plan,
            detected_intent: intent,
            total_tokens: ctx.total_tokens,
            compression_ratio: ctx.stats.compression_ratio,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StatsResponse {
    pub total_objects: u64,
    pub vector_count: u64,
    pub graph_nodes: u64,
    pub graph_edges: u64,
    pub index_size_bytes: u64,
    pub cache_stats: Option<kowitodb_planner::CacheStats>,
    pub total_cost_usd: f64,
    pub active_agent_sessions: u64,
}

// ---- Ser/de helpers ----

fn obj_to_stored(obj: &KnowledgeObject) -> KResult<StoredObject> {
    Ok(StoredObject {
        id: obj.id,
        content: obj.content.clone(),
        metadata_json: serde_json::to_string(&obj.metadata)
            .map_err(|e| kowitodb_core::KowitoError::Serialization(e.to_string()))?,
        keywords_json: serde_json::to_string(&obj.keywords)
            .map_err(|e| kowitodb_core::KowitoError::Serialization(e.to_string()))?,
        relationships_json: serde_json::to_string(&obj.relationships)
            .map_err(|e| kowitodb_core::KowitoError::Serialization(e.to_string()))?,
        embeddings_json: serde_json::to_string(&obj.embeddings)
            .map_err(|e| kowitodb_core::KowitoError::Serialization(e.to_string()))?,
        importance: obj.importance,
        created_at: obj.created_at.to_rfc3339(),
        updated_at: obj.updated_at.to_rfc3339(),
    })
}

fn stored_to_obj(stored: &StoredObject) -> KResult<KnowledgeObject> {
    Ok(KnowledgeObject {
        id: stored.id,
        content: stored.content.clone(),
        embeddings: serde_json::from_str(&stored.embeddings_json)
            .map_err(|e| kowitodb_core::KowitoError::Serialization(e.to_string()))?,
        metadata: serde_json::from_str(&stored.metadata_json)
            .map_err(|e| kowitodb_core::KowitoError::Serialization(e.to_string()))?,
        keywords: serde_json::from_str(&stored.keywords_json)
            .map_err(|e| kowitodb_core::KowitoError::Serialization(e.to_string()))?,
        relationships: serde_json::from_str(&stored.relationships_json)
            .map_err(|e| kowitodb_core::KowitoError::Serialization(e.to_string()))?,
        importance: stored.importance,
        created_at: chrono::DateTime::parse_from_rfc3339(&stored.created_at)
            .map(|dt| dt.with_timezone(&chrono::Utc))
            .unwrap_or_else(|_| chrono::Utc::now()),
        updated_at: chrono::DateTime::parse_from_rfc3339(&stored.updated_at)
            .map(|dt| dt.with_timezone(&chrono::Utc))
            .unwrap_or_else(|_| chrono::Utc::now()),
        version_history: Vec::new(),
    })
}
