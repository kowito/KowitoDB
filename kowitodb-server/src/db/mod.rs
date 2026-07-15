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

mod ask;
mod graphrag;
mod ingest;
mod memory;
mod sql;

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
mod tests;
