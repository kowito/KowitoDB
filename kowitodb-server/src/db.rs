use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use kowitodb_core::Result as KResult;
use kowitodb_core::{Embedding, KnowledgeObject, ObjectId, Relationship};
use kowitodb_index::{
    FullTextIndex, GraphIndex, HnswParams, IndexResult, IndexSource, MetadataIndex,
    MultiVectorIndex, ShardedHnswIndex, TimeIndex, VectorIndex,
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
use crate::llm::LlmClient;
use crate::memory::{AgentMemory, TurnRole};
use crate::openai::{OpenAiConfig, OpenAiEmbeddingClient};
use crate::proto;
use crate::rerank::CrossEncoder;

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
    env_flag("KOWITODB_VECTOR_QUANTIZE")
}

/// Whether to RaBitQ-style 1-bit binary-quantize stored vectors (~32× less
/// memory), via `KOWITODB_VECTOR_BINARY_QUANTIZE=1`. Off by default; takes
/// precedence over int8 quantization when both are set.
fn vector_binary_quantize_enabled() -> bool {
    env_flag("KOWITODB_VECTOR_BINARY_QUANTIZE")
}

fn env_flag(name: &str) -> bool {
    std::env::var(name)
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE"))
        .unwrap_or(false)
}

/// Matryoshka adaptive-retrieval coarse dimension, via
/// `KOWITODB_VECTOR_COARSE_DIM=<n>`. When set, the index navigates on the first
/// `n` dimensions and refines top-k at full precision. Requires MRL embeddings.
fn vector_coarse_dim() -> Option<usize> {
    std::env::var("KOWITODB_VECTOR_COARSE_DIM")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&d| d > 0)
}

/// Vector-index parameters built from the environment.
fn vector_index_params() -> HnswParams {
    HnswParams {
        quantize: vector_quantize_enabled(),
        binary_quantize: vector_binary_quantize_enabled(),
        // Retain int8 vectors to re-score binary candidates (higher recall).
        binary_rerank: env_flag("KOWITODB_VECTOR_BINARY_RERANK"),
        coarse_dim: vector_coarse_dim(),
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

/// Whether to use the LLM to generate per-object context at ingest (the
/// faithful Contextual Retrieval; opt-in via `KOWITODB_LLM_CONTEXTUAL=1` since
/// it issues one LLM call per insert). Off by default.
fn llm_contextual_enabled() -> bool {
    std::env::var("KOWITODB_LLM_CONTEXTUAL")
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE"))
        .unwrap_or(false)
}

/// Whether to auto-enrich the graph with `co_mentions` edges at ingest
/// (LazyGraphRAG-style; default on, disable with `KOWITODB_AUTO_GRAPH=0`).
fn auto_graph_enabled() -> bool {
    std::env::var("KOWITODB_AUTO_GRAPH")
        .map(|v| !matches!(v.as_str(), "0" | "false" | "FALSE"))
        .unwrap_or(true)
}

/// Max prior objects linked per shared entity at ingest (bounds fan-out).
const AUTO_GRAPH_FANOUT: usize = 5;

/// Cheap deterministic entity extraction: capitalized tokens (proper nouns)
/// from the content plus the object's explicit keywords, normalized for
/// matching. The LazyGraphRAG insight — a light extractor at ingest is enough
/// to enrich a graph — without the cost of full LLM relation extraction.
fn extract_entities(obj: &KnowledgeObject) -> Vec<String> {
    let mut set: HashSet<String> = HashSet::new();
    for word in obj.content.split_whitespace() {
        let clean: String = word.chars().filter(|c| c.is_alphanumeric()).collect();
        if clean.chars().count() > 2 && clean.chars().next().is_some_and(|c| c.is_uppercase()) {
            set.insert(clean.to_lowercase());
        }
    }
    for kw in &obj.keywords {
        let k = kw.trim().to_lowercase();
        if k.len() > 1 {
            set.insert(k);
        }
    }
    set.into_iter().collect()
}

/// Whether `sql` is a single read-only query (`SELECT`/`WITH`) safe to execute
/// against DataFusion. Rejects multiple statements and any write/DDL/filesystem
/// keyword. Conservative by design — it gates client- and LLM-generated SQL.
fn is_read_only_sql(sql: &str) -> bool {
    let trimmed = sql.trim().trim_end_matches(';').trim();
    // Disallow statement chaining (`SELECT ...; DROP ...`).
    if trimmed.contains(';') {
        return false;
    }
    let lower = trimmed.to_lowercase();
    if !(lower.starts_with("select ") || lower.starts_with("with ")) {
        return false;
    }
    // Reject embedded write/DDL/filesystem verbs even inside a leading SELECT
    // (e.g. CTEs or sub-statements). Matched with surrounding spaces to avoid
    // tripping on column names like `created_at`.
    const FORBIDDEN: &[&str] = &[
        " insert ",
        " update ",
        " delete ",
        " drop ",
        " create ",
        " alter ",
        " copy ",
        " attach ",
        " grant ",
        " truncate ",
        " replace ",
        " merge ",
        " call ",
        " execute ",
    ];
    let padded = format!(" {lower} ");
    !FORBIDDEN.iter().any(|kw| padded.contains(kw))
}

/// Strip Markdown code fences and a leading `sql` tag from an LLM SQL reply.
fn strip_sql_fence(s: &str) -> String {
    let mut t = s.trim();
    if let Some(rest) = t.strip_prefix("```") {
        t = rest;
        if let Some(nl) = t.find('\n') {
            // Drop an optional language tag on the opening fence line.
            if t[..nl].trim().eq_ignore_ascii_case("sql") {
                t = &t[nl + 1..];
            }
        }
        if let Some(end) = t.rfind("```") {
            t = &t[..end];
        }
    }
    t.trim().trim_end_matches(';').trim().to_string()
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

/// A GraphRAG community: a cluster of related objects plus an LLM-generated
/// summary of them, used to answer global/holistic ("what are the themes?")
/// questions that no single object answers.
#[derive(Debug, Clone)]
pub struct CommunitySummary {
    pub members: Vec<ObjectId>,
    pub summary: String,
}

/// Minimum community size worth summarizing, and the per-community member cap
/// (bounds summarization token cost).
const MIN_COMMUNITY_SIZE: usize = 2;
const COMMUNITY_SUMMARY_MAX_MEMBERS: usize = 50;

/// Core engine wiring storage, all 6 indexes, query planner, and all optimizers.
pub struct KowitoDBEngine {
    pub storage: Arc<dyn StorageBackend>,
    pub hnsw_index: Arc<ShardedHnswIndex>,
    pub vector_index: Arc<VectorIndex>,
    pub fulltext_index: Arc<FullTextIndex>,
    pub metadata_index: Arc<MetadataIndex>,
    pub time_index: Arc<TimeIndex>,
    pub graph_index: Arc<GraphIndex>,
    /// Late-interaction (ColBERT-style) multi-vector index for MaxSim retrieval.
    /// Populated only when token vectors are supplied (needs a multi-vector model).
    pub multivector_index: Arc<MultiVectorIndex>,
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
    /// Optional second-stage cross-encoder reranker (re-scores the top results).
    reranker_model: Option<Arc<dyn CrossEncoder>>,
    /// Optional generative LLM client powering contextual retrieval, NL→SQL
    /// routing, and Mem0-style consolidation. `None` ⇒ those features fall back
    /// to their deterministic behavior.
    llm_client: Option<Arc<dyn LlmClient>>,
    /// Inverted index of extracted entity → objects mentioning it, used to
    /// auto-build `co_mentions` graph edges at ingest (LazyGraphRAG-style).
    entity_index: Arc<Mutex<HashMap<String, Vec<ObjectId>>>>,
    /// GraphRAG community summaries (built on demand by
    /// `build_community_summaries`), used by `global_query` for holistic answers.
    community_summaries: Arc<Mutex<Vec<CommunitySummary>>>,
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
            multivector_index: Arc::new(MultiVectorIndex::new()),
            planner: Arc::new(QueryPlanner::new()),
            reranker: Arc::new(Reranker::new()),
            context_optimizer: Arc::new(ContextOptimizer::new(4096)),
            cost_tracker: Arc::new(CostTracker::new()),
            agent_memory: Arc::new(agent_memory),
            embedding_client,
            plan_cache: Arc::new(plan_cache),
            content_cache: Arc::new(ContentCache::new(CONTENT_CACHE_CAP)),
            index_path,
            reranker_model: select_reranker(),
            llm_client: crate::llm::from_env(),
            entity_index: Arc::new(Mutex::new(HashMap::new())),
            community_summaries: Arc::new(Mutex::new(Vec::new())),
            default_model: "default".to_string(),
        }
    }

    /// Path of the persisted vector-index snapshot, if this engine has an index
    /// directory.
    fn multivector_snapshot_path(&self) -> Option<std::path::PathBuf> {
        self.index_path.as_ref().map(|p| p.join("multivector.bin"))
    }

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
        // Restore the late-interaction index if a snapshot exists (token vectors
        // can't be rebuilt from storage without a multi-vector model).
        if let Some(path) = self.multivector_snapshot_path() {
            if let Ok(Some(mv)) = MultiVectorIndex::load(&path) {
                info!("Loaded late-interaction index ({} docs)", mv.len());
                self.multivector_index = Arc::new(mv);
            }
        }
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
        // Persist the late-interaction index too (token vectors can't be rebuilt
        // from storage — there is no bundled multi-vector model).
        if let Some(path) = self.multivector_snapshot_path() {
            if !self.multivector_index.is_empty() {
                self.multivector_index
                    .save(&path)
                    .map_err(kowitodb_core::KowitoError::Io)?;
            }
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

        if count > 0 {
            info!(
                "Reindexing {} object(s) from storage… (this may take a moment)",
                count
            );
        }

        // Collect vectors so the sharded index can build them in parallel.
        let mut vectors: Vec<(ObjectId, Embedding)> = Vec::new();
        // Log progress every 10% (or at least once for small datasets).
        let report_every = (count / 10).max(1);

        for (i, stored) in objects.iter().enumerate() {
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

            if (i + 1) % report_every == 0 || i + 1 == count {
                info!(
                    "Reindex progress: {}/{} objects ({:.0}%)",
                    i + 1,
                    count,
                    (i + 1) as f64 / count as f64 * 100.0
                );
            }
        }

        if !vectors.is_empty() {
            info!("Building vector index from {} embedding(s)…", vectors.len());
            self.hnsw_index.build_parallel(vectors);
        }

        if count > 0 {
            info!(
                "Reindex complete: {} object(s) loaded into in-memory indexes",
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
        let indexed_text = self.contextualize(&obj).await;

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

        // LazyGraphRAG-style auto-enrichment: link to prior objects that share
        // an entity, so the graph is useful even without explicit relationships.
        if auto_graph_enabled() {
            self.auto_link_entities(id, &obj);
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
        self.multivector_index.remove(id);
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

        // Rerank with intent-conditioned source weights (the planner's detected
        // intent steers RRF fusion toward the indexes that matter for it).
        let mut ranked = self
            .reranker
            .rerank_for_intent(&all_results, &intent.intent);

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
                ranked = self
                    .reranker
                    .rerank_for_intent(&all_results, &intent.intent);
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

        // Second-stage cross-encoder rerank of the top results (when configured).
        let loaded = self.apply_cross_encoder(question, loaded).await;

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

    /// Re-score loaded results with the cross-encoder (joint query↔document
    /// relevance) and re-sort. No-op when no reranker is configured.
    async fn apply_cross_encoder(
        &self,
        query: &str,
        mut loaded: Vec<LoadedResult>,
    ) -> Vec<LoadedResult> {
        let Some(reranker) = &self.reranker_model else {
            return loaded;
        };
        if loaded.is_empty() {
            return loaded;
        }
        let docs: Vec<String> = loaded.iter().map(|l| l.content.clone()).collect();
        let scores = reranker.rerank(query, &docs).await;
        if scores.len() == loaded.len() {
            for (l, s) in loaded.iter_mut().zip(scores) {
                l.relevance_score = s;
            }
            loaded.sort_by(|a, b| {
                b.relevance_score
                    .partial_cmp(&a.relevance_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        loaded
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
        // Read-only guard: this runs client- and LLM-generated SQL (the NL→SQL
        // feature) against DataFusion, which can otherwise CREATE tables,
        // `COPY ... TO` the filesystem, or read external files. Since the engine
        // ingests untrusted documents (RAG), reject anything that isn't a single
        // read-only SELECT/WITH statement before executing it.
        if !is_read_only_sql(sql) {
            return Err(kowitodb_core::KowitoError::Planner(
                "only a single read-only SELECT/WITH query is permitted".into(),
            ));
        }
        let ctx = kowitodb_sql::SqlContext::new(self.storage.clone())
            .map_err(|e| kowitodb_core::KowitoError::Planner(e.to_string()))?;
        ctx.query_rows(sql)
            .await
            .map_err(|e| kowitodb_core::KowitoError::Planner(e.to_string()))
    }

    // ---- Generative (LLM-backed) features ----

    /// LLM-generated contextual retrieval — the faithful form of Anthropic's
    /// Contextual Retrieval: a one-sentence situating context, generated by the
    /// LLM, prepended to the *indexed* text (the stored/returned content is
    /// untouched). Falls back to the deterministic structured-field context when
    /// no LLM client is configured or `KOWITODB_LLM_CONTEXTUAL` is unset.
    async fn contextualize(&self, obj: &KnowledgeObject) -> String {
        let base = contextualize_for_index(obj);
        if !llm_contextual_enabled() {
            return base;
        }
        let Some(llm) = &self.llm_client else {
            return base;
        };
        let system = "Write a single short sentence situating the following text \
            within its likely broader context, to improve search retrieval. \
            Output only the sentence, with no preamble.";
        match llm.complete(system, &obj.content).await {
            Ok(ctx) if !ctx.trim().is_empty() => format!("{}\n{}", ctx.trim(), base),
            Ok(_) => base,
            Err(e) => {
                debug!("LLM contextualization failed: {e}");
                base
            }
        }
    }

    /// Translate a natural-language question into a single SQL query over the
    /// `knowledge` table using the LLM. `None` when no client is configured.
    async fn nl_to_sql(&self, question: &str) -> Option<String> {
        let llm = self.llm_client.as_ref()?;
        let system = "Translate the user's question into ONE SQL query over a \
            table `knowledge` with columns: id (text), content (text), \
            importance (real), created_at (text), updated_at (text), keywords \
            (text), metadata (text, JSON). Use only SELECT. Output only the SQL \
            — no markdown fences, no explanation.";
        match llm.complete(system, question).await {
            Ok(sql) => Some(strip_sql_fence(&sql)),
            Err(e) => {
                debug!("NL→SQL translation failed: {e}");
                None
            }
        }
    }

    /// Answer an analytical/aggregational question by translating it to SQL via
    /// the LLM and executing it over the store. Returns `None` when no LLM
    /// client is configured (callers fall back to retrieval).
    pub async fn answer_with_sql(
        &self,
        question: &str,
    ) -> KResult<Option<Vec<HashMap<String, String>>>> {
        let Some(sql) = self.nl_to_sql(question).await else {
            return Ok(None);
        };
        debug!("NL→SQL: {sql}");
        Ok(Some(self.sql_select(&sql).await?))
    }

    /// Mem0-style memory distillation: extract the single durable fact worth
    /// remembering from a raw conversation turn. `Some(fact)` to promote that
    /// fact, `None` to skip promotion (the LLM judged it not memory-worthy, or
    /// the call failed). Only consulted when an LLM client is configured.
    async fn distill_memory(&self, content: &str) -> Option<String> {
        let llm = self.llm_client.as_ref()?;
        let system = "Extract the single most important durable fact or \
            preference from the message that is worth remembering long-term. \
            Reply with just that fact as a concise statement, or exactly NOOP if \
            nothing is worth remembering.";
        match llm.complete(system, content).await {
            Ok(s) if s.trim().eq_ignore_ascii_case("noop") => None,
            Ok(s) if !s.trim().is_empty() => Some(s.trim().to_string()),
            _ => None,
        }
    }

    // ---- GraphRAG community summarization ----

    /// Build (or rebuild) **GraphRAG community summaries**: detect communities in
    /// the knowledge graph (the auto-built `co_mentions` graph makes this work
    /// without explicit relationships), then summarize each community's members.
    /// With an `LlmClient` the summary is LLM-generated; otherwise it is a
    /// deterministic digest. Returns the number of communities summarized.
    ///
    /// This is the indexing half of Microsoft GraphRAG. It is **opt-in / on
    /// demand** precisely because full community summarization is token-expensive
    /// — call it after a batch ingest, not on every write.
    pub async fn build_community_summaries(&self) -> KResult<usize> {
        let communities = self.graph_index.detect_communities();
        let mut summaries = Vec::new();
        for members in communities
            .into_iter()
            .filter(|c| c.len() >= MIN_COMMUNITY_SIZE)
        {
            let mut texts = Vec::new();
            for &id in members.iter().take(COMMUNITY_SUMMARY_MAX_MEMBERS) {
                if let Some(obj) = self.get(id).await? {
                    texts.push(obj.content);
                }
            }
            if texts.is_empty() {
                continue;
            }
            let summary = self.summarize_community(&texts).await;
            summaries.push(CommunitySummary { members, summary });
        }
        let n = summaries.len();
        *self.community_summaries.lock() = summaries;
        info!("Built {n} GraphRAG community summaries");
        Ok(n)
    }

    /// Summarize one community's member texts (LLM when configured, else a
    /// deterministic digest).
    async fn summarize_community(&self, texts: &[String]) -> String {
        let joined = texts.join("\n- ");
        match &self.llm_client {
            Some(llm) => {
                let system = "Summarize these related items into a concise \
                    paragraph capturing the key entities, themes, and \
                    relationships among them. Output only the summary.";
                llm.complete(system, &joined).await.unwrap_or_else(|e| {
                    debug!("community summarization failed: {e}");
                    joined.chars().take(280).collect()
                })
            }
            None => format!(
                "Community of {} items: {}",
                texts.len(),
                joined.chars().take(280).collect::<String>()
            ),
        }
    }

    /// Answer a **global/holistic** question via GraphRAG map-reduce over the
    /// community summaries: each summary yields a partial answer (map), then the
    /// partials are combined into a final answer (reduce). Builds the summaries
    /// on first use. Returns `None` when no LLM client is configured or no
    /// community is relevant (callers fall back to ordinary retrieval).
    pub async fn global_query(&self, question: &str) -> KResult<Option<String>> {
        let Some(llm) = self.llm_client.clone() else {
            return Ok(None);
        };
        if self.community_summaries.lock().is_empty() {
            self.build_community_summaries().await?;
        }
        let summaries: Vec<String> = self
            .community_summaries
            .lock()
            .iter()
            .map(|c| c.summary.clone())
            .collect();
        if summaries.is_empty() {
            return Ok(None);
        }

        // Map: a partial answer from each community summary.
        let mut partials = Vec::new();
        for s in &summaries {
            let system = "Using ONLY the community summary, give a partial answer \
                to the question, or reply exactly NONE if the summary is \
                irrelevant.";
            let user = format!("Community summary:\n{s}\n\nQuestion: {question}");
            if let Ok(ans) = llm.complete(system, &user).await {
                let a = ans.trim();
                if !a.is_empty() && !a.eq_ignore_ascii_case("none") {
                    partials.push(a.to_string());
                }
            }
        }
        if partials.is_empty() {
            return Ok(None);
        }

        // Reduce: combine the partials into one comprehensive answer.
        let system = "Combine the partial answers into one comprehensive, \
            non-redundant final answer to the question. Output only the answer.";
        let user = format!(
            "Question: {question}\n\nPartial answers:\n- {}",
            partials.join("\n- ")
        );
        let final_answer = llm
            .complete(system, &user)
            .await
            .unwrap_or_else(|_| partials.join(" "));
        Ok(Some(final_answer))
    }

    /// Snapshot of the current community summaries (for inspection / RPC).
    pub fn community_summaries(&self) -> Vec<CommunitySummary> {
        self.community_summaries.lock().clone()
    }

    // ---- Late interaction (ColBERT-style multi-vector) ----

    /// Store late-interaction token vectors for an object, enabling MaxSim
    /// retrieval via [`Self::late_interaction_search`]. The token vectors come
    /// from a multi-vector model (e.g. ColBERT) — KowitoDB indexes and scores
    /// them; it does not bundle the model.
    pub fn index_token_vectors(&self, id: ObjectId, tokens: Vec<Vec<f32>>) {
        self.multivector_index.insert(id, tokens);
    }

    /// Late-interaction retrieval: top-`k` objects by MaxSim against the
    /// multi-vector `query`. With `candidates`, MaxSim only re-ranks that
    /// shortlist (the production ANN→MaxSim two-stage); otherwise it scores all
    /// objects that have token vectors.
    pub fn late_interaction_search(
        &self,
        query: &[Vec<f32>],
        k: usize,
        candidates: Option<&[ObjectId]>,
    ) -> Vec<(ObjectId, f32)> {
        self.multivector_index.search(query, k, candidates)
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
        // the graph to the existing knowledge the turn mentions. With an LLM
        // client, the raw turn is first distilled to a salient durable fact
        // (Mem0-style); `None` means "nothing worth remembering" → skip.
        if !matches!(turn_role, TurnRole::System) && !content.trim().is_empty() {
            let promote = if self.llm_client.is_some() {
                self.distill_memory(&content).await
            } else {
                Some(content.clone())
            };
            if let Some(fact) = promote {
                let mem_id = stable_memory_id(session_id, &fact);
                if self.get(mem_id).await?.is_none() {
                    let related = self.find_related_objects(&fact, 3);
                    let mut obj = KnowledgeObject::new(fact)
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
        }
        Ok(count)
    }

    /// LazyGraphRAG-style auto-enrichment: extract entities from `obj`, link it
    /// (bidirectional `co_mentions` edges) to prior objects sharing an entity,
    /// and register its own entities for future inserts. Cheap and deterministic
    /// — no LLM. Bounded by [`AUTO_GRAPH_FANOUT`] per shared entity.
    fn auto_link_entities(&self, id: ObjectId, obj: &KnowledgeObject) {
        let entities = extract_entities(obj);
        if entities.is_empty() {
            return;
        }
        let mut idx = self.entity_index.lock();
        let mut targets: HashSet<ObjectId> = HashSet::new();
        for e in &entities {
            if let Some(objs) = idx.get(e) {
                for &o in objs.iter().rev().take(AUTO_GRAPH_FANOUT) {
                    if o != id {
                        targets.insert(o);
                    }
                }
            }
        }
        for e in &entities {
            idx.entry(e.clone()).or_default().push(id);
        }
        drop(idx);

        if targets.is_empty() {
            return;
        }
        let edge = |target_id: ObjectId| Relationship {
            relation_type: "co_mentions".into(),
            target_id,
            weight: Some(0.5),
        };
        let out: Vec<Relationship> = targets.iter().map(|&t| edge(t)).collect();
        self.graph_index.insert_relationships(id, &out);
        for &t in &targets {
            self.graph_index.insert_relationships(t, &[edge(id)]);
        }
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

/// Select the optional cross-encoder reranker from `KOWITODB_RERANKER_PROVIDER`
/// (`local` → on-device Candle, requires the `cross-encoder-rerank` feature).
/// Returns `None` when unset/unavailable, leaving the RRF ranking in place.
fn select_reranker() -> Option<Arc<dyn CrossEncoder>> {
    let provider = std::env::var("KOWITODB_RERANKER_PROVIDER")
        .unwrap_or_default()
        .to_lowercase();
    if provider != "local" {
        return None;
    }
    #[cfg(feature = "cross-encoder-rerank")]
    {
        let model = std::env::var("KOWITODB_RERANKER_MODEL")
            .unwrap_or_else(|_| crate::rerank::DEFAULT_RERANKER_MODEL.to_string());
        match crate::rerank::CandleCrossEncoder::load(&model) {
            Ok(reranker) => {
                info!("Reranker: on-device cross-encoder ({model})");
                Some(Arc::new(reranker))
            }
            Err(e) => {
                tracing::error!("Failed to load cross-encoder ({e}); using RRF ranking only");
                None
            }
        }
    }
    #[cfg(not(feature = "cross-encoder-rerank"))]
    {
        tracing::warn!(
            "KOWITODB_RERANKER_PROVIDER=local but built without the \
             cross-encoder-rerank feature; using RRF ranking only"
        );
        None
    }
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
    async fn test_llm_contextualization_prepends_generated_context() {
        std::env::set_var("KOWITODB_LLM_CONTEXTUAL", "1");
        let mut engine = KowitoDBEngine::new_in_memory().unwrap();
        engine.llm_client = Some(std::sync::Arc::new(crate::llm::testing::MockLlm {
            response: "This is from Acme's Q3 earnings report.".into(),
        }));
        let obj = KnowledgeObject::new("Revenue grew 20%.");
        let text = engine.contextualize(&obj).await;
        assert!(
            text.contains("Acme's Q3 earnings report"),
            "LLM-generated context should be prepended to the indexed text"
        );
        assert!(text.contains("Revenue grew 20%."));
        std::env::remove_var("KOWITODB_LLM_CONTEXTUAL");
    }

    #[tokio::test]
    async fn test_nl_to_sql_executes_against_store() {
        let mut engine = KowitoDBEngine::new_in_memory().unwrap();
        engine
            .insert(KnowledgeObject::new("Acme raised a Series A"))
            .await
            .unwrap();
        engine
            .insert(KnowledgeObject::new("Initech churned last quarter"))
            .await
            .unwrap();
        engine.llm_client = Some(std::sync::Arc::new(crate::llm::testing::MockLlm {
            response: "```sql\nSELECT content FROM knowledge;\n```".into(),
        }));
        let rows = engine
            .answer_with_sql("how many records are there?")
            .await
            .unwrap()
            .expect("LLM client present → Some rows");
        assert_eq!(rows.len(), 2, "NL→SQL query should run over the store");
    }

    #[tokio::test]
    async fn test_memory_distillation_promotes_salient_fact() {
        let mut engine = KowitoDBEngine::new_in_memory().unwrap();
        engine.llm_client = Some(std::sync::Arc::new(crate::llm::testing::MockLlm {
            response: "The user prefers dark mode.".into(),
        }));
        engine
            .remember_turn(
                "s1",
                "user",
                "hey so, um, I really like dark mode I guess".into(),
            )
            .await
            .unwrap();
        // The promoted memory is the distilled fact, not the raw rambling turn.
        let resp = engine.ask("dark mode preference", 5).await.unwrap();
        assert!(
            resp.results
                .iter()
                .any(|r| r.content == "The user prefers dark mode."),
            "distilled fact should be the searchable memory"
        );
    }

    #[tokio::test]
    async fn test_late_interaction_maxsim_retrieval() {
        let engine = KowitoDBEngine::new_in_memory().unwrap();
        let basis = |i: usize| {
            let mut v = vec![0.0f32; 4];
            v[i] = 1.0;
            v
        };
        let a = engine
            .insert(KnowledgeObject::new("alpha doc"))
            .await
            .unwrap();
        let b = engine
            .insert(KnowledgeObject::new("beta doc"))
            .await
            .unwrap();
        // Token vectors (from a hypothetical ColBERT model): A covers e0/e1, B e2/e3.
        engine.index_token_vectors(a, vec![basis(0), basis(1)]);
        engine.index_token_vectors(b, vec![basis(2), basis(3)]);

        let res = engine.late_interaction_search(&[basis(0)], 2, None);
        assert_eq!(res[0].0, a, "MaxSim ranks the token-matching doc first");

        // Delete drops its token vectors from the index too.
        engine.delete(a).await.unwrap();
        let res = engine.late_interaction_search(&[basis(0)], 2, None);
        assert!(res.iter().all(|(id, _)| *id != a));
    }

    #[tokio::test]
    async fn test_graphrag_community_summaries_and_global_query() {
        let mut engine = KowitoDBEngine::new_in_memory().unwrap();
        // Two entity clusters; the auto-graph links each cluster internally.
        for c in [
            "Acme launched Rocket",
            "Acme hired Director",
            "Acme raised Capital",
        ] {
            engine.insert(KnowledgeObject::new(c)).await.unwrap();
        }
        for c in ["Globex shipped Gadget", "Globex acquired Startup"] {
            engine.insert(KnowledgeObject::new(c)).await.unwrap();
        }

        engine.llm_client = Some(std::sync::Arc::new(crate::llm::testing::MockLlm {
            response: "Two companies, Acme and Globex, are active.".into(),
        }));

        // Two communities (Acme ×3, Globex ×2) are detected and summarized.
        let n = engine.build_community_summaries().await.unwrap();
        assert_eq!(n, 2, "two entity clusters → two community summaries");

        // Global map-reduce query produces a holistic answer.
        let answer = engine
            .global_query("Which companies are mentioned?")
            .await
            .unwrap()
            .expect("LLM present → Some answer");
        assert!(answer.contains("Acme") && answer.contains("Globex"));
    }

    #[tokio::test]
    async fn test_graphrag_global_query_without_llm_is_none() {
        let engine = KowitoDBEngine::new_in_memory().unwrap();
        engine
            .insert(KnowledgeObject::new("Acme launched Rocket"))
            .await
            .unwrap();
        engine
            .insert(KnowledgeObject::new("Acme hired Director"))
            .await
            .unwrap();
        // No LLM client → global query falls back (None), no panic.
        assert!(engine.global_query("themes?").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_auto_graph_links_co_mentions() {
        let engine = KowitoDBEngine::new_in_memory().unwrap();
        let a = engine
            .insert(KnowledgeObject::new(
                "Acme Corporation raised a Series A round.",
            ))
            .await
            .unwrap();
        let b = engine
            .insert(KnowledgeObject::new(
                "Later, Acme Corporation hired a new CEO.",
            ))
            .await
            .unwrap();
        // Shared entity "Acme"/"Corporation" → an auto co_mentions edge b↔a.
        let out_b = engine.graph_index.out_edges(b);
        assert!(
            out_b
                .iter()
                .any(|r| r.target_id == a && r.relation_type == "co_mentions"),
            "auto-graph should link co-mentioning objects"
        );
        // And it is bidirectional.
        assert!(engine
            .graph_index
            .out_edges(a)
            .iter()
            .any(|r| r.target_id == b));
    }

    #[tokio::test]
    async fn test_strip_sql_fence() {
        assert_eq!(strip_sql_fence("```sql\nSELECT 1;\n```"), "SELECT 1");
        assert_eq!(
            strip_sql_fence("SELECT * FROM knowledge"),
            "SELECT * FROM knowledge"
        );
    }

    #[test]
    fn test_is_read_only_sql() {
        assert!(is_read_only_sql("SELECT content FROM knowledge"));
        assert!(is_read_only_sql("  with x as (select 1) select * from x  "));
        assert!(is_read_only_sql(
            "SELECT count(*) FROM knowledge WHERE created_at > '2020'"
        ));
        // Writes / DDL / chaining / filesystem are rejected.
        assert!(!is_read_only_sql("DROP TABLE knowledge"));
        assert!(!is_read_only_sql("SELECT 1; DROP TABLE knowledge"));
        assert!(!is_read_only_sql("COPY knowledge TO '/tmp/x.csv'"));
        assert!(!is_read_only_sql("CREATE TABLE t AS SELECT 1"));
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

    struct MockReranker;
    #[async_trait::async_trait]
    impl CrossEncoder for MockReranker {
        async fn rerank(&self, _query: &str, documents: &[String]) -> Vec<f32> {
            // Score documents mentioning "beta" much higher.
            documents
                .iter()
                .map(|d| if d.contains("beta") { 10.0 } else { 1.0 })
                .collect()
        }
    }

    #[tokio::test]
    async fn test_cross_encoder_reranks_results() {
        let mut engine = KowitoDBEngine::new_in_memory().unwrap();
        engine
            .insert(
                KnowledgeObject::new("enterprise widget alpha")
                    .with_keywords(vec!["enterprise".into()]),
            )
            .await
            .unwrap();
        engine
            .insert(
                KnowledgeObject::new("enterprise widget beta")
                    .with_keywords(vec!["enterprise".into()]),
            )
            .await
            .unwrap();

        // Inject a cross-encoder that prefers "beta"; it should reorder results.
        engine.reranker_model = Some(std::sync::Arc::new(MockReranker));
        let resp = engine.ask("enterprise widget", 5).await.unwrap();
        assert!(
            resp.results
                .first()
                .map(|r| r.content.contains("beta"))
                .unwrap_or(false),
            "cross-encoder's preferred document should rank first"
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
