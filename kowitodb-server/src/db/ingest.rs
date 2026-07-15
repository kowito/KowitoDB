//! `KowitoDBEngine` — ingest methods (split from the former db.rs god-file).

use super::*;

impl KowitoDBEngine {
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

    /// Store late-interaction token vectors for an object, enabling MaxSim
    /// retrieval via [`Self::late_interaction_search`]. The token vectors come
    /// from a multi-vector model (e.g. ColBERT) — KowitoDB indexes and scores
    /// them; it does not bundle the model.
    pub fn index_token_vectors(&self, id: ObjectId, tokens: Vec<Vec<f32>>) {
        self.multivector_index.insert(id, tokens);
    }

    /// LazyGraphRAG-style auto-enrichment: extract entities from `obj`, link it
    /// (bidirectional `co_mentions` edges) to prior objects sharing an entity,
    /// and register its own entities for future inserts. Cheap and deterministic
    /// — no LLM. Bounded by [`AUTO_GRAPH_FANOUT`] per shared entity.
    pub(crate) fn auto_link_entities(&self, id: ObjectId, obj: &KnowledgeObject) {
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
    pub(crate) fn find_related_objects(&self, text: &str, k: usize) -> Vec<ObjectId> {
        match self.fulltext_index.search(text, k) {
            Ok(results) => results.into_iter().map(|(id, _)| id).collect(),
            Err(_) => Vec::new(),
        }
    }
}
