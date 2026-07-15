//! `KowitoDBEngine` — ask methods (split from the former db.rs god-file).

use super::*;

impl KowitoDBEngine {
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
    pub(crate) async fn apply_ranking_signals(
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
    pub(crate) fn metadata_allowed_set(
        &self,
        filter: &HashMap<String, String>,
    ) -> HashSet<ObjectId> {
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
    pub(crate) async fn corrective_retrieval(&self, question: &str) -> KResult<Vec<IndexResult>> {
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
    pub(crate) async fn apply_cross_encoder(
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
    pub(crate) async fn load_results(&self, ranked: &[RankedResult]) -> Vec<LoadedResult> {
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
    pub(crate) fn assemble_context_from_loaded(
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
    pub(crate) async fn execute_plan(
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
    pub(crate) async fn execute_graph_traversal(
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

    /// LLM-generated contextual retrieval — the faithful form of Anthropic's
    /// Contextual Retrieval: a one-sentence situating context, generated by the
    /// LLM, prepended to the *indexed* text (the stored/returned content is
    /// untouched). Falls back to the deterministic structured-field context when
    /// no LLM client is configured or `KOWITODB_LLM_CONTEXTUAL` is unset.
    pub(crate) async fn contextualize(&self, obj: &KnowledgeObject) -> String {
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
}
