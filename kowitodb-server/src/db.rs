use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use kowitodb_core::Result as KResult;
use kowitodb_core::{Embedding, KnowledgeObject, ObjectId};
use kowitodb_index::{
    FullTextIndex, GraphIndex, HnswParams, IndexResult, IndexSource, MetadataIndex,
    ShardedHnswIndex, TimeIndex, VectorIndex,
};
use kowitodb_planner::{
    cache::QueryCache, context::ContextOptimizer, cost::CostTracker, reranker::Reranker,
    DetectedIntent, ExecutionPlan, QueryPlanner, RankedResult,
};
use kowitodb_storage::{StorageBackend, StorageEngine, StorageFilter, StoredObject};
use lru::LruCache;
use parking_lot::Mutex;
use std::num::NonZeroUsize;
use tracing::{debug, info};

use crate::embedding::{EmbeddingClient, ProxyEmbeddingClient};
use crate::memory::{AgentMemory, TurnRole};
use crate::openai::{OpenAiConfig, OpenAiEmbeddingClient};
use crate::proto;

/// Maximum number of object contents held in the in-memory LRU cache. On a
/// miss, content is reloaded from storage, so this only bounds memory use.
const CONTENT_CACHE_CAP: usize = 10_000;

/// Number of HNSW shards for the vector index — scales build/query with cores.
fn vector_shard_count() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .clamp(1, 16)
}

/// Whether to int8-quantize stored vectors (4× less memory), via
/// `KOWITODB_VECTOR_QUANTIZE=1`. Off by default.
fn vector_quantize_enabled() -> bool {
    std::env::var("KOWITODB_VECTOR_QUANTIZE")
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE"))
        .unwrap_or(false)
}

/// Vector-index parameters built from the environment.
fn vector_index_params() -> HnswParams {
    HnswParams {
        quantize: vector_quantize_enabled(),
        ..Default::default()
    }
}

/// Retrieval confidence below this triggers a corrective (broadened) pass.
const CONFIDENCE_THRESHOLD: f32 = 0.35;

/// Whether the CRAG-style corrective gate is enabled (default on; disable with
/// `KOWITODB_CORRECTIVE_RETRIEVAL=0`).
fn corrective_retrieval_enabled() -> bool {
    std::env::var("KOWITODB_CORRECTIVE_RETRIEVAL")
        .map(|v| !matches!(v.as_str(), "0" | "false" | "FALSE"))
        .unwrap_or(true)
}

/// Estimate retrieval confidence in [0, 1] from the ranked results.
///
/// The reranker normalizes the top score to 1.0, so confidence keys on *result
/// coverage* (did we find enough?) and *cross-source agreement* (do multiple
/// indexes agree?) rather than the absolute top score.
fn retrieval_confidence(ranked: &[RankedResult], requested: usize) -> f32 {
    if ranked.is_empty() {
        return 0.0;
    }
    let req = requested.max(1) as f32;
    let coverage = (ranked.len().min(requested) as f32) / req;
    let considered = ranked.iter().take(requested).count().max(1) as f32;
    let multi_source = ranked
        .iter()
        .take(requested)
        .filter(|r| r.sources.len() > 1)
        .count() as f32
        / considered;
    0.7 * coverage + 0.3 * multi_source
}

/// Whether Contextual Retrieval augmentation is enabled (default on; disable
/// with `KOWITODB_CONTEXTUAL_RETRIEVAL=0`).
fn contextual_retrieval_enabled() -> bool {
    std::env::var("KOWITODB_CONTEXTUAL_RETRIEVAL")
        .map(|v| !matches!(v.as_str(), "0" | "false" | "FALSE"))
        .unwrap_or(true)
}

/// Build the text to embed / full-text index: a deterministic context preamble
/// (sorted metadata + keywords) prepended to the content. Returns the original
/// content unchanged when disabled or when there is nothing to add.
///
/// This is the first-cut, no-LLM form of Anthropic's Contextual Retrieval — the
/// context comes from the object's structured fields rather than a generative
/// model. The stored/returned content is never modified.
fn contextualize_for_index(obj: &KnowledgeObject) -> String {
    if !contextual_retrieval_enabled() {
        return obj.content.clone();
    }

    let mut context = String::new();
    let mut metadata: Vec<_> = obj.metadata.iter().collect();
    metadata.sort_by(|a, b| a.0.cmp(b.0)); // deterministic ordering
    for (key, value) in metadata {
        let val = match value {
            serde_json::Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        context.push_str(&format!("{key}: {val}. "));
    }
    if !obj.keywords.is_empty() {
        context.push_str(&format!("Keywords: {}. ", obj.keywords.join(", ")));
    }

    if context.is_empty() {
        obj.content.clone()
    } else {
        format!("{context}\n{}", obj.content)
    }
}

/// Bounded LRU cache of object content keyed by id, used to avoid storage
/// round-trips for hot objects without growing without limit.
struct ContentCache(Mutex<LruCache<ObjectId, String>>);

impl ContentCache {
    fn new(capacity: usize) -> Self {
        let cap = NonZeroUsize::new(capacity).unwrap_or(NonZeroUsize::MIN);
        Self(Mutex::new(LruCache::new(cap)))
    }

    fn get(&self, id: &ObjectId) -> Option<String> {
        self.0.lock().get(id).cloned()
    }

    fn insert(&self, id: ObjectId, content: String) {
        self.0.lock().put(id, content);
    }

    fn remove(&self, id: &ObjectId) {
        self.0.lock().pop(id);
    }
}

/// A fully loaded result with content from storage.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct LoadedResult {
    pub id: ObjectId,
    pub content: String,
    pub relevance_score: f32,
    pub retrieval_sources: Vec<String>,
    pub metadata: HashMap<String, String>,
    pub importance: f32,
}

/// Core engine wiring storage, all 6 indexes, query planner, and all optimizers.
pub struct KowitoDBEngine {
    pub storage: Arc<dyn StorageBackend>,
    pub hnsw_index: Arc<ShardedHnswIndex>,
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
    content_cache: Arc<ContentCache>,
    /// Index directory; when set, the vector index is persisted here as a
    /// snapshot (`None` for in-memory engines).
    index_path: Option<std::path::PathBuf>,
    #[allow(dead_code)]
    default_model: String,
}

impl KowitoDBEngine {
    pub fn new(
        storage_path: impl AsRef<std::path::Path>,
        index_path: impl AsRef<std::path::Path>,
    ) -> KResult<Self> {
        let storage: Arc<dyn StorageBackend> = Arc::new(StorageEngine::open(storage_path)?);
        let index_ref = index_path.as_ref();
        let agent_memory = open_session_store(index_ref)?;
        let fulltext_index = FullTextIndex::open(index_ref)?;
        let engine = Self::assemble(
            storage,
            fulltext_index,
            agent_memory,
            Some(index_ref.to_path_buf()),
        );
        info!("KowitoDB engine initialized with all subsystems (sled storage)");
        Ok(engine)
    }

    /// Open a sled-backed engine and rebuild the in-memory indexes from the
    /// persisted object store. Prefer this over [`Self::new`] when serving an
    /// existing database: the sled/Lance store and the full-text index persist
    /// across restarts, but the vector/metadata/time/graph indexes start empty
    /// and must be repopulated for search to work immediately.
    pub async fn open(
        storage_path: impl AsRef<std::path::Path>,
        index_path: impl AsRef<std::path::Path>,
    ) -> KResult<Self> {
        let mut engine = Self::new(storage_path, index_path)?;
        engine.load_or_reindex().await?;
        Ok(engine)
    }

    /// Create an engine backed by a [Lance](https://lancedb.github.io/lance/)
    /// dataset instead of the default sled store. Requires the `lance` feature.
    #[cfg(feature = "lance")]
    pub async fn new_with_lance(
        lance_uri: impl Into<String>,
        index_path: impl AsRef<std::path::Path>,
    ) -> KResult<Self> {
        let storage: Arc<dyn StorageBackend> =
            Arc::new(kowitodb_storage::LanceStorage::open(lance_uri).await?);
        let index_ref = index_path.as_ref();
        let agent_memory = open_session_store(index_ref)?;
        let fulltext_index = FullTextIndex::open(index_ref)?;
        let mut engine = Self::assemble(
            storage,
            fulltext_index,
            agent_memory,
            Some(index_ref.to_path_buf()),
        );
        engine.load_or_reindex().await?;
        info!("KowitoDB engine initialized with all subsystems (Lance storage)");
        Ok(engine)
    }

    /// Create an in-memory engine for testing (no disk I/O).
    pub fn new_in_memory() -> KResult<Self> {
        let storage: Arc<dyn StorageBackend> = Arc::new(StorageEngine::new_in_memory()?);
        // For tests, use a temp directory for the fulltext index
        let tmp = std::env::temp_dir().join(format!("kowitodb-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&tmp).map_err(kowitodb_core::KowitoError::Io)?;
        let fulltext_index = FullTextIndex::open(&tmp)?;
        Ok(Self::assemble(
            storage,
            fulltext_index,
            AgentMemory::new(),
            None,
        ))
    }

    /// Assemble the full engine (all indexes, planner, optimizers) over a given
    /// storage backend, full-text index, and agent-memory store. Shared by every
    /// constructor.
    fn assemble(
        storage: Arc<dyn StorageBackend>,
        fulltext_index: FullTextIndex,
        agent_memory: AgentMemory,
        index_path: Option<std::path::PathBuf>,
    ) -> Self {
        let embedding_client = select_embedding_client();
        let plan_cache: QueryCache<(DetectedIntent, ExecutionPlan)> = QueryCache::new(300, 1000);

        Self {
            storage,
            hnsw_index: Arc::new(ShardedHnswIndex::new(
                vector_shard_count(),
                vector_index_params(),
            )),
            vector_index: Arc::new(VectorIndex::new()),
            fulltext_index: Arc::new(fulltext_index),
            metadata_index: Arc::new(MetadataIndex::new()),
            time_index: Arc::new(TimeIndex::new()),
            graph_index: Arc::new(GraphIndex::new()),
            planner: Arc::new(QueryPlanner::new()),
            reranker: Arc::new(Reranker::new()),
            context_optimizer: Arc::new(ContextOptimizer::new(4096)),
            cost_tracker: Arc::new(CostTracker::new()),
            agent_memory: Arc::new(agent_memory),
            embedding_client,
            plan_cache: Arc::new(plan_cache),
            content_cache: Arc::new(ContentCache::new(CONTENT_CACHE_CAP)),
            index_path,
            default_model: "default".to_string(),
        }
    }

    /// Path of the persisted vector-index snapshot, if this engine has an index
    /// directory.
    fn hnsw_snapshot_path(&self) -> Option<std::path::PathBuf> {
        self.index_path.as_ref().map(|p| p.join("hnsw.bin"))
    }

    /// Load the persisted vector index if a snapshot exists, then rebuild the
    /// remaining in-memory indexes from storage. If no snapshot is found the
    /// vector index is rebuilt from stored embeddings too.
    async fn load_or_reindex(&mut self) -> KResult<()> {
        let loaded = match self.hnsw_snapshot_path() {
            Some(path) => match ShardedHnswIndex::load(&path) {
                Ok(Some(index)) => {
                    info!("Loaded persisted vector index ({} vectors)", index.len());
                    self.hnsw_index = Arc::new(index);
                    true
                }
                Ok(None) => false,
                Err(e) => {
                    tracing::warn!("Could not load vector index snapshot ({e}); rebuilding");
                    false
                }
            },
            None => false,
        };
        self.reindex_from_storage(!loaded).await?;
        Ok(())
    }

    /// Persist the vector index to disk so it need not be rebuilt on restart.
    /// No-op for in-memory engines.
    pub fn checkpoint(&self) -> KResult<()> {
        if let Some(path) = self.hnsw_snapshot_path() {
            self.hnsw_index
                .save(&path)
                .map_err(kowitodb_core::KowitoError::Io)?;
            debug!(
                "Checkpointed vector index ({} vectors) to {:?}",
                self.hnsw_index.len(),
                path
            );
        }
        Ok(())
    }

    /// Rebuild the in-memory indexes (vector/metadata/time/graph) and content
    /// cache from the persisted object store. Returns the number of objects
    /// reindexed.
    ///
    /// The full-text index is intentionally skipped: it persists to disk and is
    /// already loaded on open, so re-inserting would duplicate documents.
    /// Embeddings are taken from storage — no embedding API calls are made.
    ///
    /// When `include_vectors` is false the HNSW index is left untouched (e.g. it
    /// was just loaded from a snapshot); the other indexes are still rebuilt.
    pub async fn reindex_from_storage(&self, include_vectors: bool) -> KResult<usize> {
        let objects = self.storage.search(StorageFilter::default()).await?;
        let count = objects.len();

        // Collect vectors so the sharded index can build them in parallel.
        let mut vectors: Vec<(ObjectId, Embedding)> = Vec::new();

        for stored in &objects {
            let obj = stored_to_obj(stored)?;
            self.content_cache.insert(obj.id, obj.content.clone());

            if include_vectors {
                for embedding in obj.embeddings.values() {
                    vectors.push((obj.id, embedding.clone()));
                }
            }
            for (key, value) in &obj.metadata {
                let val_str = match value {
                    serde_json::Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                self.metadata_index.insert(obj.id, key, &val_str);
            }
            self.time_index
                .insert(obj.id, obj.created_at.timestamp_millis());
            if !obj.relationships.is_empty() {
                self.graph_index
                    .insert_relationships(obj.id, &obj.relationships);
            }
        }

        if !vectors.is_empty() {
            self.hnsw_index.build_parallel(vectors);
        }

        if count > 0 {
            info!(
                "Reindexed {} objects from storage into in-memory indexes",
                count
            );
        }
        Ok(count)
    }

    /// Insert a knowledge object into storage and all 6 indexes.
    pub async fn insert(&self, mut obj: KnowledgeObject) -> KResult<ObjectId> {
        let id = obj.id;

        // Cache the *original* content for retrieval/display.
        self.content_cache.insert(id, obj.content.clone());

        // Contextual Retrieval (Anthropic, 2024): embed and full-text index a
        // context-augmented version of the text while storage returns the
        // original. The dense vector and BM25 index then capture structured
        // context (metadata/keywords), improving recall.
        let indexed_text = contextualize_for_index(&obj);

        // Index vectors (auto-embed if needed). The generated embedding is
        // written back onto the object so it is persisted to storage and can be
        // restored by reindex_from_storage() after a restart.
        for embedding in obj.embeddings.values() {
            self.hnsw_index.insert(id, embedding.clone());
        }
        if obj.embeddings.is_empty() && !obj.content.is_empty() {
            if let Ok(result) = self.embedding_client.embed(&indexed_text).await {
                self.hnsw_index.insert(id, result.vector.clone());
                obj.embeddings.insert(result.model, result.vector);
                self.cost_tracker.record_embedding_calls(1);
            }
        }

        // Full-text index (over the context-augmented text).
        self.fulltext_index.insert(
            id,
            &indexed_text,
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

        // Ensure fulltext index is searchable immediately
        let _ = self.fulltext_index.commit();

        self.plan_cache.clear();

        info!(
            "Inserted {}: {} (vecs={}, kws={}, rels={})",
            id,
            &obj.content[..obj.content.len().min(80)],
            obj.embeddings.len(),
            obj.keywords.len(),
            obj.relationships.len(),
        );
        Ok(id)
    }

    /// Insert many objects in one call, returning their ids in order.
    pub async fn batch_insert(&self, objects: Vec<KnowledgeObject>) -> KResult<Vec<ObjectId>> {
        let mut ids = Vec::with_capacity(objects.len());
        for obj in objects {
            ids.push(self.insert(obj).await?);
        }
        Ok(ids)
    }

    /// List stored objects with pagination. Returns the requested page (ordered
    /// by id for stable paging) and the total object count.
    pub async fn list(
        &self,
        offset: usize,
        limit: usize,
    ) -> KResult<(Vec<KnowledgeObject>, usize)> {
        let mut ids = self.storage.list_ids().await?;
        ids.sort();
        let total = ids.len();

        let mut objects = Vec::new();
        for id in ids.into_iter().skip(offset).take(limit) {
            if let Some(obj) = self.get(id).await? {
                objects.push(obj);
            }
        }
        Ok((objects, total))
    }

    /// Re-rank the top `window` results by stored ranking signals: an importance
    /// factor (`1 + IMPORTANCE_WEIGHT * importance`) and a recency factor
    /// (`1 + RECENCY_WEIGHT * exp(-age_days / HALF_LIFE)`). A uniform default
    /// importance (0.5) and equal ages leave the order unchanged.
    async fn apply_ranking_signals(
        &self,
        mut ranked: Vec<RankedResult>,
        window: usize,
    ) -> Vec<RankedResult> {
        const IMPORTANCE_WEIGHT: f32 = 0.5;
        const RECENCY_WEIGHT: f32 = 0.2;
        let window = window.min(ranked.len());
        for r in ranked.iter_mut().take(window) {
            if let Ok(Some(stored)) = self.storage.get(r.id).await {
                let importance_factor = 1.0 + IMPORTANCE_WEIGHT * stored.importance;
                let recency_factor = 1.0 + RECENCY_WEIGHT * recency_score(&stored.created_at);
                r.score *= importance_factor * recency_factor;
            }
        }
        ranked.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        ranked
    }

    /// Intersection of object ids matching every exact metadata pair (AND).
    fn metadata_allowed_set(&self, filter: &HashMap<String, String>) -> HashSet<ObjectId> {
        let mut allowed: Option<HashSet<ObjectId>> = None;
        for (key, value) in filter {
            let ids: HashSet<ObjectId> = self
                .metadata_index
                .query_exact(key, value)
                .into_iter()
                .collect();
            allowed = Some(match allowed {
                Some(acc) => acc.intersection(&ids).copied().collect(),
                None => ids,
            });
            if allowed.as_ref().is_some_and(|s| s.is_empty()) {
                break;
            }
        }
        allowed.unwrap_or_default()
    }

    /// Broadened retrieval used by the corrective gate: a wide vector + keyword
    /// sweep over the question, returned as raw index results to merge + re-rank.
    async fn corrective_retrieval(&self, question: &str) -> KResult<Vec<IndexResult>> {
        const WIDE: usize = 50;
        let mut out = Vec::new();

        if let Ok(results) = self.fulltext_index.search(question, WIDE) {
            if !results.is_empty() {
                let (ids, scores): (Vec<_>, Vec<_>) = results.into_iter().unzip();
                out.push(IndexResult::new(ids, scores, IndexSource::FullText));
            }
        }
        if let Ok(emb) = self.embedding_client.embed(question).await {
            self.cost_tracker.record_embedding_calls(1);
            let results = self.hnsw_index.search(&emb.vector, WIDE);
            if !results.is_empty() {
                let (ids, scores): (Vec<_>, Vec<_>) = results.into_iter().unzip();
                out.push(IndexResult::new(ids, scores, IndexSource::Vector));
            }
        }
        Ok(out)
    }

    /// Retrieve by ID (checks content cache first, then storage).
    pub async fn get(&self, id: ObjectId) -> KResult<Option<KnowledgeObject>> {
        match self.storage.get(id).await? {
            Some(stored) => {
                let obj = stored_to_obj(&stored)?;
                // Refresh content cache
                self.content_cache.insert(id, obj.content.clone());
                Ok(Some(obj))
            }
            None => Ok(None),
        }
    }

    /// Delete from all indexes and storage.
    pub async fn delete(&self, id: ObjectId) -> KResult<bool> {
        self.hnsw_index.remove(id);
        self.vector_index.remove(id);
        let _ = self.fulltext_index.remove(id);
        self.metadata_index.remove_object(id);
        self.time_index.remove(id);
        self.graph_index.remove_object(id);
        self.content_cache.remove(&id);

        let existed = self.storage.delete(id).await?;
        if existed {
            self.plan_cache.clear();
            info!("Deleted {}", id);
        }
        Ok(existed)
    }

    /// Update an existing object in place (id preserved), recording a version
    /// history entry. Returns the new version count, or `None` if not found.
    ///
    /// Changing the content clears the stored embedding so it is regenerated on
    /// re-insert, keeping the vector index accurate.
    pub async fn update(
        &self,
        id: ObjectId,
        content: Option<String>,
        metadata: HashMap<String, String>,
        keywords: Vec<String>,
        importance: Option<f32>,
        change_description: Option<String>,
    ) -> KResult<Option<usize>> {
        let Some(mut obj) = self.get(id).await? else {
            return Ok(None);
        };

        let content_changed = match content {
            Some(c) => {
                let changed = c != obj.content;
                obj.content = c;
                changed
            }
            None => false,
        };
        for (k, v) in metadata {
            obj.metadata.insert(k, serde_json::Value::String(v));
        }
        if !keywords.is_empty() {
            obj.keywords = keywords;
        }
        if let Some(imp) = importance {
            obj.importance = imp.clamp(0.0, 1.0);
        }
        obj.record_version(change_description);
        if content_changed {
            obj.embeddings.clear();
        }
        let version = obj.version_history.len();

        // Re-index: drop stale entries, then re-insert under the same id.
        self.delete(id).await?;
        self.insert(obj).await?;
        Ok(Some(version))
    }

    /// The core `ai.ask()` method — full pipeline with real content.
    pub async fn ask(&self, question: &str, max_results: usize) -> KResult<AskResponse> {
        self.ask_filtered(question, max_results, None, &HashMap::new())
            .await
    }

    /// `ai.ask()` with an optional context-token budget that honors a request's
    /// `max_context_tokens`. A `None` (or zero) budget uses the engine default.
    pub async fn ask_with_budget(
        &self,
        question: &str,
        max_results: usize,
        max_context_tokens: Option<usize>,
    ) -> KResult<AskResponse> {
        self.ask_filtered(question, max_results, max_context_tokens, &HashMap::new())
            .await
    }

    /// `ai.ask()` constrained to objects matching every `metadata_filter` pair
    /// (exact match, ANDed). An empty filter retrieves without constraint.
    pub async fn ask_filtered(
        &self,
        question: &str,
        max_results: usize,
        max_context_tokens: Option<usize>,
        metadata_filter: &HashMap<String, String>,
    ) -> KResult<AskResponse> {
        // Check plan cache
        let (intent, plan) = if let Some(cached) = self.plan_cache.get(question) {
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

        // Graph traversal
        let graph_results = self.execute_graph_traversal(&raw_results, &intent).await?;
        let mut all_results: Vec<IndexResult> =
            raw_results.into_iter().chain(graph_results).collect();

        // Rerank
        let mut ranked = self.reranker.rerank_simple(&all_results);

        // CRAG-style corrective gate: when retrieval confidence is low (few
        // results / little cross-source agreement), broaden the search across
        // vector + keyword and re-rank. Exploits the integrated indexes.
        if corrective_retrieval_enabled()
            && retrieval_confidence(&ranked, max_results) < CONFIDENCE_THRESHOLD
        {
            let corrective = self.corrective_retrieval(question).await?;
            if !corrective.is_empty() {
                self.cost_tracker
                    .record_index_lookups(corrective.iter().map(|r| r.ids.len()).sum());
                all_results.extend(corrective);
                ranked = self.reranker.rerank_simple(&all_results);
                debug!("Corrective retrieval engaged for low-confidence query");
            }
        }

        // Apply metadata filter (via the metadata index) before limiting, so the
        // result count reflects the constraint.
        let ranked: Vec<RankedResult> = if metadata_filter.is_empty() {
            ranked
        } else {
            let allowed = self.metadata_allowed_set(metadata_filter);
            ranked
                .into_iter()
                .filter(|r| allowed.contains(&r.id))
                .collect()
        };

        // Boost results by stored `importance` (priority) and recency (newer
        // knowledge), so high-priority and fresh items surface. Applied over a
        // candidate window so an item just below the cut can rise.
        let ranked = self
            .apply_ranking_signals(ranked, max_results.saturating_mul(3))
            .await;

        // Limit + load real content
        let limited: Vec<RankedResult> = ranked.into_iter().take(max_results).collect();
        let loaded = self.load_results(&limited).await;

        // Assemble optimized context from loaded content
        let assembled = self.assemble_context_from_loaded(&loaded, max_context_tokens);
        self.cost_tracker
            .record_llm_input_tokens(assembled.total_tokens);

        Ok(AskResponse::from_loaded(
            loaded,
            plan.explain(),
            format!("{:?}", intent.intent),
            assembled,
        ))
    }

    /// Load real content for ranked results from cache + storage.
    async fn load_results(&self, ranked: &[RankedResult]) -> Vec<LoadedResult> {
        let mut loaded = Vec::with_capacity(ranked.len());

        for r in ranked {
            // Try content cache first
            let content = if let Some(cached) = self.content_cache.get(&r.id) {
                cached
            } else {
                // Fall back to storage
                match self.storage.get(r.id).await {
                    Ok(Some(stored)) => {
                        let val = stored.content.clone();
                        self.content_cache.insert(r.id, val.clone());
                        val
                    }
                    _ => format!("<Object {}>", r.id),
                }
            };

            let sources: Vec<String> = r
                .sources
                .iter()
                .map(|s| format!("{:?}", s).to_lowercase())
                .collect();

            let mut metadata = HashMap::new();
            metadata.insert("sources".to_string(), sources.join(","));

            loaded.push(LoadedResult {
                id: r.id,
                content,
                relevance_score: r.score,
                retrieval_sources: sources,
                metadata,
                importance: 0.5,
            });
        }

        loaded
    }

    /// Assemble context from already-loaded results.
    fn assemble_context_from_loaded(
        &self,
        loaded: &[LoadedResult],
        max_tokens: Option<usize>,
    ) -> kowitodb_planner::AssembledContext {
        // Convert LoadedResult -> RankedResult for the optimizer
        let ranked: Vec<RankedResult> = loaded
            .iter()
            .map(|l| RankedResult {
                id: l.id,
                score: l.relevance_score,
                sources: l
                    .retrieval_sources
                    .iter()
                    .map(|s| match s.as_str() {
                        "vector" => IndexSource::Vector,
                        "fulltext" => IndexSource::FullText,
                        "graph" => IndexSource::Graph,
                        "metadata" => IndexSource::Metadata,
                        "time" => IndexSource::Time,
                        _ => IndexSource::Vector,
                    })
                    .collect(),
                source_scores: HashMap::new(),
            })
            .collect();

        let content_lookup = |id: ObjectId| -> Option<String> {
            loaded
                .iter()
                .find(|l| l.id == id)
                .map(|l| l.content.clone())
        };

        self.context_optimizer
            .assemble_with_budget(&ranked, &content_lookup, max_tokens)
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

        let query_embedding: Option<Embedding> =
            self.embedding_client.embed(question).await.ok().map(|r| {
                self.cost_tracker.record_embedding_calls(1);
                r.vector
            });

        for step in &plan.steps {
            match step.step_type {
                kowitodb_planner::PlanStepType::VectorSearch => {
                    if let Some(ref emb) = query_embedding {
                        let results = self.hnsw_index.search(emb, step.limit.unwrap_or(20));
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

    /// Graph traversal for entity-heavy queries.
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

        // Bidirectional: follows both "references" and "referenced by" edges
        let scored = self
            .graph_index
            .scored_bidirectional_traverse(&seeds, max_depth, None);
        if scored.is_empty() {
            return Ok(Vec::new());
        }

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

        Ok(vec![IndexResult::new(ids, scores, IndexSource::Graph)])
    }

    /// Execute a SQL query against knowledge objects.
    ///
    /// Maps SQL WHERE clauses to the metadata, keyword, and time indexes.
    /// Results are loaded from storage with real content.
    pub async fn sql_query(&self, sql: &str) -> KResult<Vec<LoadedResult>> {
        let stmt = kowitodb_sql::parse_sql(sql)
            .map_err(|e| kowitodb_core::KowitoError::Planner(e.to_string()))?;

        let (where_clauses, limit) = match stmt {
            kowitodb_sql::SqlStatement::Select {
                where_clauses,
                limit,
                ..
            } => (where_clauses, limit),
        };

        let mut candidate_sets: Vec<Vec<ObjectId>> = Vec::new();

        for clause in &where_clauses {
            match clause {
                kowitodb_sql::WhereClause::MetadataEquals { key, value } => {
                    let ids = self.metadata_index.query_exact(key, value);
                    if !ids.is_empty() {
                        candidate_sets.push(ids);
                    }
                }
                kowitodb_sql::WhereClause::MetadataContains { key, substring } => {
                    let ids = self.metadata_index.query_contains(key, substring);
                    if !ids.is_empty() {
                        candidate_sets.push(ids);
                    }
                }
                kowitodb_sql::WhereClause::KeywordContains { substring } => {
                    // Use full-text search for keyword contains
                    if let Ok(results) = self.fulltext_index.search(substring, 100) {
                        if !results.is_empty() {
                            candidate_sets.push(results.into_iter().map(|(id, _)| id).collect());
                        }
                    }
                }
                kowitodb_sql::WhereClause::ContentContains { substring } => {
                    if let Ok(results) = self.fulltext_index.search(substring, 100) {
                        if !results.is_empty() {
                            candidate_sets.push(results.into_iter().map(|(id, _)| id).collect());
                        }
                    }
                }
                kowitodb_sql::WhereClause::CreatedAfter { timestamp } => {
                    // Parse timestamp to milliseconds
                    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(timestamp) {
                        let ids = self.time_index.after(dt.timestamp_millis());
                        if !ids.is_empty() {
                            candidate_sets.push(ids);
                        }
                    }
                }
                kowitodb_sql::WhereClause::CreatedBefore { timestamp } => {
                    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(timestamp) {
                        let ids = self.time_index.before(dt.timestamp_millis());
                        if !ids.is_empty() {
                            candidate_sets.push(ids);
                        }
                    }
                }
                _ => {
                    debug!("SQL clause not yet routed to index: {:?}", clause);
                }
            }
        }

        // Intersect candidate sets (AND semantics)
        let final_ids: Vec<ObjectId> = if candidate_sets.is_empty() {
            // No WHERE clauses: get all objects
            self.storage.list_ids().await?
        } else if candidate_sets.len() == 1 {
            candidate_sets.into_iter().next().unwrap()
        } else {
            // Intersect all sets
            let mut sets: Vec<std::collections::HashSet<ObjectId>> = candidate_sets
                .into_iter()
                .map(|v| v.into_iter().collect())
                .collect();
            let (first, rest) = sets.split_at_mut(1);
            first[0].retain(|id| rest.iter().all(|s| s.contains(id)));
            first[0].iter().copied().collect()
        };

        // Apply limit
        let final_ids: Vec<ObjectId> = if let Some(lim) = limit {
            final_ids.into_iter().take(lim).collect()
        } else {
            final_ids
        };

        // Load content for results
        let mut loaded = Vec::with_capacity(final_ids.len());
        for id in &final_ids {
            let content = if let Some(cached) = self.content_cache.get(id) {
                cached
            } else if let Ok(Some(stored)) = self.storage.get(*id).await {
                let val = stored.content.clone();
                self.content_cache.insert(*id, val.clone());
                val
            } else {
                format!("<Object {}>", id)
            };

            loaded.push(LoadedResult {
                id: *id,
                content,
                relevance_score: 1.0,
                retrieval_sources: vec!["sql".to_string()],
                metadata: HashMap::new(),
                importance: 0.5,
            });
        }

        Ok(loaded)
    }

    /// Execute arbitrary SQL through the DataFusion engine.
    ///
    /// Unlike [`Self::sql_query`] (which routes a small parsed subset to the
    /// native indexes), this runs the full DataFusion planner over the
    /// `knowledge` table provider — supporting projections, `ORDER BY`,
    /// `GROUP BY`, aggregates, and complex predicates. Rows are returned as
    /// ordered column-name → stringified-value maps.
    pub async fn sql_select(&self, sql: &str) -> KResult<Vec<HashMap<String, String>>> {
        let ctx = kowitodb_sql::SqlContext::new(self.storage.clone())
            .map_err(|e| kowitodb_core::KowitoError::Planner(e.to_string()))?;
        ctx.query_rows(sql)
            .await
            .map_err(|e| kowitodb_core::KowitoError::Planner(e.to_string()))
    }

    /// Record an agent conversation turn AND promote it into searchable,
    /// graph-able knowledge (Mem0-style episodic memory). Returns the session's
    /// turn count.
    ///
    /// The memory object has a stable id derived from `(session_id, content)`, so
    /// re-recording the same turn is idempotent (no duplicate memory). `system`
    /// turns are not promoted. This makes past conversation retrievable via
    /// `ai.ask()` and linkable in the graph alongside ingested knowledge.
    pub async fn remember_turn(
        &self,
        session_id: &str,
        role: &str,
        content: String,
    ) -> KResult<u32> {
        let turn_role = match role.to_lowercase().as_str() {
            "assistant" => TurnRole::Assistant,
            "system" => TurnRole::System,
            "observation" => TurnRole::Observation,
            _ => TurnRole::User,
        };

        let mut session = self.agent_memory.get_or_create(session_id);
        session.add_turn(turn_role.clone(), content.clone());
        let count = session.turn_count() as u32;
        self.agent_memory.save(session);

        // Promote to searchable knowledge (idempotent by stable id), linked in
        // the graph to the existing knowledge the turn mentions.
        if !matches!(turn_role, TurnRole::System) && !content.trim().is_empty() {
            let mem_id = stable_memory_id(session_id, &content);
            if self.get(mem_id).await?.is_none() {
                let related = self.find_related_objects(&content, 3);
                let mut obj = KnowledgeObject::new(content)
                    .with_metadata("session_id", session_id)
                    .with_metadata("role", role)
                    .with_metadata("kind", "memory");
                obj.id = mem_id;
                for target in related {
                    obj = obj.with_relationship("mentions", target);
                }
                self.insert(obj).await?;
            }
        }
        Ok(count)
    }

    /// Up to `k` existing knowledge objects related to `text` (via the full-text
    /// index) — used to link a new memory to the entities it mentions.
    fn find_related_objects(&self, text: &str, k: usize) -> Vec<ObjectId> {
        match self.fulltext_index.search(text, k) {
            Ok(results) => results.into_iter().map(|(id, _)| id).collect(),
            Err(_) => Vec::new(),
        }
    }

    /// Comprehensive database stats.
    pub async fn stats(&self) -> KResult<StatsResponse> {
        Ok(StatsResponse {
            total_objects: self.storage.count().await? as u64,
            vector_count: self.hnsw_index.len() as u64,
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
    fn from_loaded(
        loaded: Vec<LoadedResult>,
        plan: String,
        intent: String,
        ctx: kowitodb_planner::AssembledContext,
    ) -> Self {
        let results: Vec<proto::AskResult> = loaded
            .into_iter()
            .map(|l| proto::AskResult {
                id: l.id.to_string(),
                content: l.content,
                relevance_score: l.relevance_score,
                metadata: l.metadata,
                retrieval_source: l.retrieval_sources.first().cloned().unwrap_or_default(),
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

/// The deterministic dev embedding fallback (no network, not semantic).
fn proxy_embedding_client() -> Arc<dyn EmbeddingClient> {
    Arc::new(ProxyEmbeddingClient::new("proxy-text-embedding", 128))
}

/// Select the embedding client from `KOWITODB_EMBEDDING_PROVIDER`:
/// `local` (Candle on-device), `openai`/`ollama` (HTTP), else the dev proxy.
fn select_embedding_client() -> Arc<dyn EmbeddingClient> {
    let provider = std::env::var("KOWITODB_EMBEDDING_PROVIDER")
        .unwrap_or_default()
        .to_lowercase();

    if provider == "local" {
        return local_embedding_client();
    }

    match OpenAiConfig::from_env() {
        Some(cfg) => {
            info!(
                "Embeddings: OpenAI-compatible provider (model={})",
                cfg.model
            );
            Arc::new(OpenAiEmbeddingClient::new(cfg))
        }
        None => {
            info!("Embeddings: deterministic proxy (set KOWITODB_EMBEDDING_PROVIDER=local for a real on-device model)");
            proxy_embedding_client()
        }
    }
}

#[cfg(feature = "local-embeddings")]
fn local_embedding_client() -> Arc<dyn EmbeddingClient> {
    let model = std::env::var("KOWITODB_EMBEDDING_MODEL")
        .unwrap_or_else(|_| crate::local_embedding::DEFAULT_LOCAL_MODEL.to_string());
    match crate::local_embedding::LocalEmbeddingClient::load(&model) {
        Ok(client) => Arc::new(client),
        Err(e) => {
            tracing::error!("Failed to load local embedding model ({e}); using the proxy instead");
            proxy_embedding_client()
        }
    }
}

#[cfg(not(feature = "local-embeddings"))]
fn local_embedding_client() -> Arc<dyn EmbeddingClient> {
    tracing::warn!(
        "KOWITODB_EMBEDDING_PROVIDER=local but this binary was built without the \
         local-embeddings feature; using the proxy"
    );
    proxy_embedding_client()
}

/// Recency score in [0, 1] from an RFC3339 timestamp: 1.0 for "now", decaying
/// with an ~30-day half-life. Returns 0 for unparseable timestamps.
fn recency_score(created_at: &str) -> f32 {
    const HALF_LIFE_DAYS: f32 = 30.0;
    match chrono::DateTime::parse_from_rfc3339(created_at) {
        Ok(dt) => {
            let age_days = (chrono::Utc::now() - dt.with_timezone(&chrono::Utc))
                .num_days()
                .max(0) as f32;
            (-age_days / HALF_LIFE_DAYS).exp()
        }
        Err(_) => 0.0,
    }
}

/// Deterministic memory id from `(session_id, content)`, so the same turn maps
/// to the same knowledge object (idempotent promotion).
fn stable_memory_id(session_id: &str, content: &str) -> ObjectId {
    use std::hash::{Hash, Hasher};
    let mut hi = std::collections::hash_map::DefaultHasher::new();
    session_id.hash(&mut hi);
    content.hash(&mut hi);
    0xA5u8.hash(&mut hi);
    let mut lo = std::collections::hash_map::DefaultHasher::new();
    content.hash(&mut lo);
    session_id.hash(&mut lo);
    0x5Au8.hash(&mut lo);
    let bits = ((hi.finish() as u128) << 64) | (lo.finish() as u128);
    uuid::Uuid::from_u128(bits)
}

/// Open the persistent agent-session store under `{index_path}/sessions`.
fn open_session_store(index_path: &std::path::Path) -> KResult<AgentMemory> {
    let sessions_path = index_path.join("sessions");
    AgentMemory::open(&sessions_path)
        .map_err(|e| kowitodb_core::KowitoError::Storage(format!("agent session store: {e}")))
}

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
        version_history_json: serde_json::to_string(&obj.version_history)
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
        version_history: serde_json::from_str(&stored.version_history_json).unwrap_or_default(),
        importance: stored.importance,
        created_at: chrono::DateTime::parse_from_rfc3339(&stored.created_at)
            .map(|dt| dt.with_timezone(&chrono::Utc))
            .unwrap_or_else(|_| chrono::Utc::now()),
        updated_at: chrono::DateTime::parse_from_rfc3339(&stored.updated_at)
            .map(|dt| dt.with_timezone(&chrono::Utc))
            .unwrap_or_else(|_| chrono::Utc::now()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_insert_and_ask_end_to_end() {
        let engine = KowitoDBEngine::new_in_memory().unwrap();

        // Insert enterprise customer knowledge
        let acme = KnowledgeObject::new(
            "Acme Corp renewed their enterprise license in March 2024 after raising Series A funding of $15M."
        )
        .with_keywords(vec!["acme".into(), "renewal".into(), "series a".into(), "enterprise".into()])
        .with_metadata("company", "Acme Corp")
        .with_metadata("stage", "series_a")
        .with_metadata("renewed", "true")
        .with_importance(0.9);

        let globex = KnowledgeObject::new(
            "Globex Inc. received Series B funding of $30M in January 2024 and upgraded to enterprise tier."
        )
        .with_keywords(vec!["globex".into(), "series b".into(), "enterprise".into(), "funding".into()])
        .with_metadata("company", "Globex Inc.")
        .with_metadata("stage", "series_b")
        .with_metadata("renewed", "true");

        let initech = KnowledgeObject::new(
            "Initech went through Series A in 2023 but churned in December 2024 due to budget cuts."
        )
        .with_keywords(vec!["initech".into(), "series a".into(), "churn".into()])
        .with_metadata("company", "Initech")
        .with_metadata("stage", "series_a")
        .with_metadata("renewed", "false");

        // Insert all
        engine.insert(acme).await.unwrap();
        engine.insert(globex).await.unwrap();
        engine.insert(initech).await.unwrap();

        // Ask a natural language question
        let response = engine
            .ask("Which enterprise customers renewed after Series A?", 5)
            .await
            .unwrap();

        // Verify we got results
        assert!(!response.results.is_empty(), "Should have results");
        println!(
            "Intent: {}, Results: {}",
            response.detected_intent,
            response.results.len()
        );

        // Results should contain real content, not placeholders
        for r in &response.results {
            assert!(
                !r.content.starts_with('<'),
                "Content should be real, got: {}",
                r.content
            );
            assert!(r.content.len() > 10, "Content too short: {}", r.content);
        }

        // Plan should be explained
        assert!(!response.plan_explanation.is_empty());
        println!("Plan:\n{}", response.plan_explanation);
    }

    #[tokio::test]
    async fn test_insert_get_delete_roundtrip() {
        let engine = KowitoDBEngine::new_in_memory().unwrap();

        let obj = KnowledgeObject::new("Test content for roundtrip")
            .with_keywords(vec!["test".into()])
            .with_metadata("key", "value");

        let id = engine.insert(obj).await.unwrap();

        // Get it back
        let retrieved = engine.get(id).await.unwrap().expect("Object should exist");
        assert_eq!(retrieved.content, "Test content for roundtrip");
        assert_eq!(retrieved.keywords, vec!["test"]);

        // Delete
        let existed = engine.delete(id).await.unwrap();
        assert!(existed);

        // Should be gone
        let gone = engine.get(id).await.unwrap();
        assert!(gone.is_none());
    }

    #[tokio::test]
    async fn test_graph_traversal_via_insert() {
        let engine = KowitoDBEngine::new_in_memory().unwrap();

        let openai = KnowledgeObject::new("OpenAI is an AI research lab").with_keywords(vec![
            "openai".into(),
            "ai".into(),
            "research".into(),
        ]);

        let ms = KnowledgeObject::new("Microsoft invested $10B in OpenAI")
            .with_keywords(vec![
                "microsoft".into(),
                "investment".into(),
                "openai".into(),
            ])
            .with_relationship("invested_in", openai.id);

        engine.insert(openai).await.unwrap();
        engine.insert(ms).await.unwrap();

        // Ask about companies connected to OpenAI
        let response = engine
            .ask("Which companies invested in OpenAI?", 5)
            .await
            .unwrap();

        println!(
            "Graph query results: {} (intent: {})",
            response.results.len(),
            response.detected_intent
        );
        // Should find Microsoft via graph traversal
        assert!(!response.results.is_empty());
    }

    #[tokio::test]
    async fn test_reindex_rebuilds_in_memory_indexes_after_restart() {
        let base = std::env::temp_dir().join(format!("kowitodb-restart-{}", uuid::Uuid::new_v4()));
        let storage_path = base.join("storage");
        let index_path = base.join("index");
        std::fs::create_dir_all(&storage_path).unwrap();
        std::fs::create_dir_all(&index_path).unwrap();

        // First session: insert objects, then drop the engine (simulated shutdown).
        {
            let engine = KowitoDBEngine::open(&storage_path, &index_path)
                .await
                .unwrap();
            engine
                .insert(
                    KnowledgeObject::new("Acme renewed their enterprise contract")
                        .with_keywords(vec!["acme".into(), "enterprise".into()])
                        .with_metadata("company", "Acme"),
                )
                .await
                .unwrap();
            engine
                .insert(
                    KnowledgeObject::new("Globex churned last quarter")
                        .with_metadata("company", "Globex"),
                )
                .await
                .unwrap();
            assert_eq!(engine.stats().await.unwrap().vector_count, 2);
        }

        // Second session over the same paths. Without reindex the in-memory
        // indexes would be empty; open() must repopulate them from storage.
        let engine = KowitoDBEngine::open(&storage_path, &index_path)
            .await
            .unwrap();
        let stats = engine.stats().await.unwrap();
        assert_eq!(stats.total_objects, 2);
        assert_eq!(
            stats.vector_count, 2,
            "vector index must be rebuilt from persisted embeddings on restart"
        );

        // Metadata index rebuilt.
        assert_eq!(
            engine.metadata_index.query_exact("company", "Acme").len(),
            1
        );

        // End-to-end ask works after restart.
        let resp = engine.ask("enterprise contract", 5).await.unwrap();
        assert!(!resp.results.is_empty());

        let _ = std::fs::remove_dir_all(&base);
    }

    #[tokio::test]
    async fn test_vector_index_checkpoint_and_reload() {
        let base = std::env::temp_dir().join(format!("kowitodb-ckpt-{}", uuid::Uuid::new_v4()));
        let storage_path = base.join("storage");
        let index_path = base.join("index");
        std::fs::create_dir_all(&storage_path).unwrap();
        std::fs::create_dir_all(&index_path).unwrap();

        {
            let engine = KowitoDBEngine::open(&storage_path, &index_path)
                .await
                .unwrap();
            for i in 0..3 {
                engine
                    .insert(KnowledgeObject::new(format!("document {i}")))
                    .await
                    .unwrap();
            }
            assert_eq!(engine.stats().await.unwrap().vector_count, 3);
            engine.checkpoint().unwrap();
        }

        // The checkpoint wrote a snapshot.
        assert!(
            index_path.join("hnsw.bin").exists(),
            "checkpoint must write hnsw.bin"
        );

        // Reopen: the snapshot is loaded and vectors are intact + searchable.
        let engine = KowitoDBEngine::open(&storage_path, &index_path)
            .await
            .unwrap();
        assert_eq!(engine.stats().await.unwrap().vector_count, 3);
        assert!(!engine.ask("document", 5).await.unwrap().results.is_empty());

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn test_retrieval_confidence() {
        // Empty → zero confidence (triggers corrective).
        assert_eq!(retrieval_confidence(&[], 5), 0.0);

        let mk = |sources: Vec<IndexSource>| RankedResult {
            id: uuid::Uuid::new_v4(),
            score: 1.0,
            sources,
            source_scores: HashMap::new(),
        };

        // One result, single source, 5 requested → low confidence.
        let sparse = vec![mk(vec![IndexSource::Vector])];
        assert!(retrieval_confidence(&sparse, 5) < CONFIDENCE_THRESHOLD);

        // Full coverage with cross-source agreement → high confidence.
        let strong: Vec<_> = (0..5)
            .map(|_| mk(vec![IndexSource::Vector, IndexSource::FullText]))
            .collect();
        assert!(retrieval_confidence(&strong, 5) > CONFIDENCE_THRESHOLD);
    }

    #[test]
    fn test_contextualize_for_index() {
        let obj = KnowledgeObject::new("Quarterly results were strong.")
            .with_metadata("company", "Acme")
            .with_keywords(vec!["renewal".into()]);
        let text = contextualize_for_index(&obj);
        assert!(text.contains("company: Acme"));
        assert!(text.contains("Keywords: renewal"));
        assert!(text.contains("Quarterly results were strong."));
        // The object's stored content is never modified.
        assert_eq!(obj.content, "Quarterly results were strong.");
    }

    #[tokio::test]
    async fn test_contextual_retrieval_finds_metadata_terms() {
        let engine = KowitoDBEngine::new_in_memory().unwrap();
        // Content does NOT mention "Acme"; only metadata does.
        let id = engine
            .insert(
                KnowledgeObject::new("Quarterly results were strong and the team grew.")
                    .with_metadata("company", "Acme"),
            )
            .await
            .unwrap();

        // The metadata term is findable because it was folded into the embedded /
        // full-text-indexed text (Contextual Retrieval).
        let resp = engine.ask("Acme", 5).await.unwrap();
        assert!(
            resp.results.iter().any(|r| r.id == id.to_string()),
            "contextual retrieval should make metadata-only terms findable"
        );

        // Stored content remains the original (un-augmented).
        let stored = engine.get(id).await.unwrap().unwrap();
        assert_eq!(
            stored.content,
            "Quarterly results were strong and the team grew."
        );
    }

    #[tokio::test]
    async fn test_memory_promoted_to_searchable_knowledge() {
        let engine = KowitoDBEngine::new_in_memory().unwrap();

        let n = engine
            .remember_turn("sess-1", "user", "I love hiking in the mountains".into())
            .await
            .unwrap();
        assert_eq!(n, 1);
        // Recorded in agent memory.
        assert_eq!(engine.agent_memory.get("sess-1").unwrap().turn_count(), 1);

        // Retrievable as knowledge via ai.ask().
        let resp = engine.ask("hiking", 5).await.unwrap();
        assert!(
            resp.results.iter().any(|r| r.content.contains("hiking")),
            "promoted memory should be retrievable"
        );

        // Idempotent: re-recording the same turn does not duplicate the memory.
        engine
            .remember_turn("sess-1", "user", "I love hiking in the mountains".into())
            .await
            .unwrap();
        let (objects, _) = engine.list(0, 100).await.unwrap();
        let memories = objects
            .iter()
            .filter(|o| o.metadata.get("kind").and_then(|v| v.as_str()) == Some("memory"))
            .count();
        assert_eq!(memories, 1, "duplicate memory must be deduped by stable id");

        // System turns are recorded but not promoted to knowledge.
        engine
            .remember_turn("sess-1", "system", "you are a helpful assistant".into())
            .await
            .unwrap();
        let (objects, _) = engine.list(0, 100).await.unwrap();
        assert!(
            !objects
                .iter()
                .any(|o| o.content.contains("helpful assistant")),
            "system turns are not promoted"
        );
    }

    #[tokio::test]
    async fn test_memory_links_to_related_knowledge() {
        let engine = KowitoDBEngine::new_in_memory().unwrap();

        // An existing knowledge entity.
        let acme_id = engine
            .insert(
                KnowledgeObject::new("Acme Corp raised a Series A funding round")
                    .with_keywords(vec!["acme".into()]),
            )
            .await
            .unwrap();

        // A turn that mentions it → the promoted memory links to it in the graph.
        engine
            .remember_turn("s1", "user", "I met with Acme about the renewal".into())
            .await
            .unwrap();

        let mem_id = stable_memory_id("s1", "I met with Acme about the renewal");
        let memory = engine.get(mem_id).await.unwrap().unwrap();
        assert!(
            memory
                .relationships
                .iter()
                .any(|r| r.target_id == acme_id && r.relation_type == "mentions"),
            "memory should be graph-linked to the Acme entity it mentions"
        );
    }

    #[tokio::test]
    async fn test_update_and_versioning() {
        let engine = KowitoDBEngine::new_in_memory().unwrap();
        let id = engine
            .insert(KnowledgeObject::new("original content").with_metadata("k", "v1"))
            .await
            .unwrap();

        // First update: change content + metadata + importance.
        let v = engine
            .update(
                id,
                Some("updated content".into()),
                HashMap::from([("k".to_string(), "v2".to_string())]),
                vec![],
                Some(0.9),
                Some("edit 1".into()),
            )
            .await
            .unwrap();
        assert_eq!(v, Some(1));

        let obj = engine.get(id).await.unwrap().unwrap();
        assert_eq!(obj.content, "updated content");
        assert_eq!(obj.metadata.get("k").and_then(|x| x.as_str()), Some("v2"));
        assert!((obj.importance - 0.9).abs() < 1e-6);
        assert_eq!(obj.version_history.len(), 1);

        // Second update accumulates history — proving versions persist across
        // storage round-trips.
        let v2 = engine
            .update(
                id,
                None,
                HashMap::new(),
                vec![],
                None,
                Some("edit 2".into()),
            )
            .await
            .unwrap();
        assert_eq!(v2, Some(2));
        assert_eq!(
            engine.get(id).await.unwrap().unwrap().version_history.len(),
            2
        );

        // Updating a missing object returns None.
        let missing = engine
            .update(
                uuid::Uuid::new_v4(),
                Some("x".into()),
                HashMap::new(),
                vec![],
                None,
                None,
            )
            .await
            .unwrap();
        assert_eq!(missing, None);
    }

    #[tokio::test]
    async fn test_batch_insert_and_list_pagination() {
        let engine = KowitoDBEngine::new_in_memory().unwrap();
        let objs: Vec<_> = (0..5)
            .map(|i| {
                KnowledgeObject::new(format!("document number {i}")).with_metadata("kind", "note")
            })
            .collect();
        let ids = engine.batch_insert(objs).await.unwrap();
        assert_eq!(ids.len(), 5);

        let (page, total) = engine.list(0, 2).await.unwrap();
        assert_eq!(total, 5);
        assert_eq!(page.len(), 2);

        let (last, total) = engine.list(4, 10).await.unwrap();
        assert_eq!(total, 5);
        assert_eq!(last.len(), 1);

        // Offset past the end yields an empty page but the correct total.
        let (none, total) = engine.list(99, 10).await.unwrap();
        assert_eq!(total, 5);
        assert!(none.is_empty());
    }

    #[tokio::test]
    async fn test_importance_weighted_ranking() {
        let engine = KowitoDBEngine::new_in_memory().unwrap();
        let low = engine
            .insert(
                KnowledgeObject::new("enterprise widget alpha")
                    .with_keywords(vec!["enterprise".into()])
                    .with_importance(0.1),
            )
            .await
            .unwrap();
        let high = engine
            .insert(
                KnowledgeObject::new("enterprise widget beta")
                    .with_keywords(vec!["enterprise".into()])
                    .with_importance(0.9),
            )
            .await
            .unwrap();

        let resp = engine.ask("enterprise widget", 5).await.unwrap();
        let pos = |id: ObjectId| resp.results.iter().position(|r| r.id == id.to_string());
        let (hp, lp) = (pos(high), pos(low));
        assert!(hp.is_some() && lp.is_some(), "both should be retrieved");
        assert!(
            hp.unwrap() < lp.unwrap(),
            "higher-importance object should rank above the lower-importance one"
        );
    }

    #[tokio::test]
    async fn test_recency_weighted_ranking() {
        let engine = KowitoDBEngine::new_in_memory().unwrap();
        // Same importance, same match terms; only age differs.
        let mut old = KnowledgeObject::new("enterprise widget historical")
            .with_keywords(vec!["enterprise".into()]);
        old.created_at = chrono::Utc::now() - chrono::Duration::days(120);
        let old_id = engine.insert(old).await.unwrap();
        let new_id = engine
            .insert(
                KnowledgeObject::new("enterprise widget current")
                    .with_keywords(vec!["enterprise".into()]),
            )
            .await
            .unwrap();

        let resp = engine.ask("enterprise widget", 5).await.unwrap();
        let pos = |id: ObjectId| resp.results.iter().position(|r| r.id == id.to_string());
        assert!(
            pos(new_id).unwrap() < pos(old_id).unwrap(),
            "more recent object should rank above the older one"
        );
    }

    #[tokio::test]
    async fn test_ask_with_metadata_filter() {
        let engine = KowitoDBEngine::new_in_memory().unwrap();
        engine
            .insert(
                KnowledgeObject::new("Acme enterprise renewal closed")
                    .with_metadata("company", "Acme")
                    .with_keywords(vec!["enterprise".into(), "renewal".into()]),
            )
            .await
            .unwrap();
        engine
            .insert(
                KnowledgeObject::new("Globex enterprise renewal closed")
                    .with_metadata("company", "Globex")
                    .with_keywords(vec!["enterprise".into(), "renewal".into()]),
            )
            .await
            .unwrap();

        let filter = HashMap::from([("company".to_string(), "Acme".to_string())]);
        let resp = engine
            .ask_filtered("enterprise renewal", 10, None, &filter)
            .await
            .unwrap();

        assert!(!resp.results.is_empty(), "filter excluded everything");
        for r in &resp.results {
            assert!(
                r.content.contains("Acme") && !r.content.contains("Globex"),
                "metadata filter leaked a non-matching object: {}",
                r.content
            );
        }
    }

    #[tokio::test]
    async fn test_ask_honors_context_token_budget() {
        let engine = KowitoDBEngine::new_in_memory().unwrap();
        for i in 0..5 {
            engine
                .insert(KnowledgeObject::new(format!(
                    "Document number {i} about enterprise renewals and funding rounds \
                     with enough words to consume a non-trivial number of tokens each."
                )))
                .await
                .unwrap();
        }

        // A tiny budget should yield a smaller assembled context than a large one.
        let small = engine
            .ask_with_budget("enterprise renewals", 5, Some(20))
            .await
            .unwrap();
        let large = engine
            .ask_with_budget("enterprise renewals", 5, Some(4096))
            .await
            .unwrap();
        assert!(small.total_tokens <= large.total_tokens);
        assert!(small.total_tokens <= 100, "small budget was not honored");
    }

    #[tokio::test]
    async fn test_sql_select_datafusion_aggregate_and_order() {
        let engine = KowitoDBEngine::new_in_memory().unwrap();

        engine
            .insert(
                KnowledgeObject::new("Acme Corp content")
                    .with_metadata("stage", "series_a")
                    .with_importance(0.9),
            )
            .await
            .unwrap();
        engine
            .insert(
                KnowledgeObject::new("Globex Inc. content")
                    .with_metadata("stage", "series_b")
                    .with_importance(0.4),
            )
            .await
            .unwrap();
        engine
            .insert(
                KnowledgeObject::new("Initech content")
                    .with_metadata("stage", "series_a")
                    .with_importance(0.7),
            )
            .await
            .unwrap();

        // Aggregate via DataFusion (not expressible through the index-routed path).
        let rows = engine
            .sql_select("SELECT COUNT(*) AS n FROM knowledge")
            .await
            .unwrap();
        assert_eq!(rows[0]["n"], "3");

        // Projection + filter + ORDER BY, all executed by DataFusion.
        let rows = engine
            .sql_select(
                "SELECT content, importance FROM knowledge \
                 WHERE importance >= 0.5 ORDER BY importance DESC",
            )
            .await
            .unwrap();
        assert_eq!(rows.len(), 2);
        assert!(rows[0]["content"].contains("Acme"));
        assert!(rows[1]["content"].contains("Initech"));
    }

    #[tokio::test]
    async fn test_sql_query_metadata_filter() {
        let engine = KowitoDBEngine::new_in_memory().unwrap();

        let acme = KnowledgeObject::new("Acme Corp content")
            .with_keywords(vec!["acme".into()])
            .with_metadata("company", "Acme Corp")
            .with_metadata("stage", "series_a");
        let globex = KnowledgeObject::new("Globex Inc. content")
            .with_keywords(vec!["globex".into()])
            .with_metadata("company", "Globex Inc.")
            .with_metadata("stage", "series_b");

        engine.insert(acme).await.unwrap();
        engine.insert(globex).await.unwrap();

        // SQL: filter by metadata
        let results = engine
            .sql_query("SELECT * FROM knowledge WHERE metadata.stage = 'series_a'")
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert!(results[0].content.contains("Acme"));

        // SQL: with LIMIT
        let results = engine
            .sql_query("SELECT content FROM knowledge WHERE metadata.company LIKE '%Inc%' LIMIT 5")
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert!(results[0].content.contains("Globex"));
    }
}
