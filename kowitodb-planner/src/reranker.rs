use std::collections::HashMap;

use kowitodb_core::ObjectId;
use kowitodb_index::{IndexResult, IndexSource, VectorIndex};
use tracing::debug;

use super::intent::Intent;

/// Result with a unified score from all retrieval sources.
#[derive(Debug, Clone)]
pub struct RankedResult {
    pub id: ObjectId,
    /// Final relevance score (0.0 - 1.0).
    pub score: f32,
    /// Which sources contributed to this result.
    pub sources: Vec<IndexSource>,
    /// Individual per-source scores.
    pub source_scores: HashMap<IndexSource, f32>,
}

/// Multi-source result reranker.
///
/// Combines results from multiple indexes using Reciprocal Rank Fusion (RRF)
/// and source-specific boosting. In Phase 2+, this can be replaced with a
/// cross-encoder model for learned reranking.
pub struct Reranker {
    /// Weight multiplier per index source.
    source_weights: HashMap<IndexSource, f32>,
    /// RRF decay constant k (default 60, as used in research).
    rrf_k: f32,
}

impl Reranker {
    pub fn new() -> Self {
        let mut source_weights = HashMap::new();
        // Vector search provides the strongest semantic signal
        source_weights.insert(IndexSource::Vector, 1.5);
        // Full-text provides precision for exact matches
        source_weights.insert(IndexSource::FullText, 1.2);
        // Graph traversal finds related entities
        source_weights.insert(IndexSource::Graph, 1.3);
        // Metadata and time are filters, lower weight
        source_weights.insert(IndexSource::Metadata, 0.8);
        source_weights.insert(IndexSource::Time, 0.7);

        Self {
            source_weights,
            rrf_k: 60.0,
        }
    }

    /// Set the weight for a specific index source.
    pub fn with_weight(mut self, source: IndexSource, weight: f32) -> Self {
        self.source_weights.insert(source, weight);
        self
    }

    /// Rerank raw index results into a single ranked list.
    ///
    /// Uses Reciprocal Rank Fusion: for each result across sources,
    /// score = sum over sources of (1 / (k + rank_in_source)).
    /// Then multiplies by source weight to boost preferred sources.
    pub fn rerank(
        &self,
        raw_results: &[IndexResult],
        _vector_index: Option<&VectorIndex>,
        _query_embedding: Option<&[f32]>,
    ) -> Vec<RankedResult> {
        self.rerank_with_weights(raw_results, &self.source_weights)
    }

    /// Rerank with **intent-conditioned** source weights: the base per-source
    /// weights are scaled by an intent-specific multiplier so the planner's
    /// detected intent steers fusion toward the indexes that matter for it
    /// (e.g. temporal → time, entity → graph, code → exact full-text). Falls
    /// back to the base weights for intents with no preference.
    pub fn rerank_for_intent(
        &self,
        raw_results: &[IndexResult],
        intent: &Intent,
    ) -> Vec<RankedResult> {
        let multipliers = intent_weight_multipliers(intent);
        if multipliers.is_empty() {
            return self.rerank_with_weights(raw_results, &self.source_weights);
        }
        let mut weights = self.source_weights.clone();
        for (source, mult) in multipliers {
            let w = weights.entry(source).or_insert(1.0);
            *w *= mult;
        }
        self.rerank_with_weights(raw_results, &weights)
    }

    fn rerank_with_weights(
        &self,
        raw_results: &[IndexResult],
        weights: &HashMap<IndexSource, f32>,
    ) -> Vec<RankedResult> {
        // Phase 1: RRF scoring
        let mut scored: HashMap<ObjectId, RankedResult> = HashMap::new();

        for result in raw_results {
            let source_weight = weights.get(&result.source).copied().unwrap_or(1.0);

            for (rank, (&id, &score)) in result.ids.iter().zip(result.scores.iter()).enumerate() {
                let rrf_score = 1.0 / (self.rrf_k + rank as f32 + 1.0);
                let weighted_rrf = rrf_score * source_weight;

                let entry = scored.entry(id).or_insert_with(|| RankedResult {
                    id,
                    score: 0.0,
                    sources: Vec::new(),
                    source_scores: HashMap::new(),
                });

                entry.score += weighted_rrf;
                if !entry.sources.contains(&result.source) {
                    entry.sources.push(result.source.clone());
                }
                entry.source_scores.insert(result.source.clone(), score);
            }
        }

        // Phase 2: Cross-source boosting
        // Results found by multiple sources get a multiplicative bonus
        for entry in scored.values_mut() {
            let source_count = entry.sources.len() as f32;
            if source_count > 1.0 {
                // Multi-source agreement: boost proportional to source count
                entry.score *= 1.0 + (source_count - 1.0) * 0.15;
            }
        }

        // Phase 3: Normalize scores to [0, 1] range
        let mut ranked: Vec<RankedResult> = scored.into_values().collect();
        let max_score = ranked
            .iter()
            .map(|r| r.score)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(1.0);

        if max_score > 0.0 {
            for entry in &mut ranked {
                entry.score /= max_score;
            }
        }

        // Sort by descending score
        ranked.sort_unstable_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        debug!(
            "Reranker: {} raw results -> {} ranked (multi-source: {})",
            raw_results.len(),
            ranked.len(),
            ranked.iter().filter(|r| r.sources.len() > 1).count()
        );

        ranked
    }

    /// Simple rerank without vector re-scoring (for use when embedding unavailable).
    pub fn rerank_simple(&self, raw_results: &[IndexResult]) -> Vec<RankedResult> {
        self.rerank(raw_results, None, None)
    }
}

impl Default for Reranker {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-intent source-weight multipliers applied on top of the base RRF weights.
/// Empty ⇒ no preference (use base weights). Tuned to the index each intent
/// most depends on; values are deliberately modest (≤1.6×) so fusion still
/// blends every source rather than collapsing to one.
fn intent_weight_multipliers(intent: &Intent) -> Vec<(IndexSource, f32)> {
    match intent {
        // Time-bounded → lean on the time index, keep semantic signal.
        Intent::Temporal => vec![(IndexSource::Time, 1.8), (IndexSource::Vector, 1.1)],
        // Entity-driven → graph relationships + exact name matches.
        Intent::EntitySearch => vec![
            (IndexSource::Graph, 1.5),
            (IndexSource::FullText, 1.3),
            (IndexSource::Metadata, 1.2),
        ],
        // Comparison spans entities → graph + semantics.
        Intent::Comparison => vec![(IndexSource::Graph, 1.4), (IndexSource::Vector, 1.2)],
        // Code → exact token matches dominate.
        Intent::CodeSearch => vec![(IndexSource::FullText, 1.5)],
        // Factoid lookups reward precise term matches.
        Intent::Factoid => vec![(IndexSource::FullText, 1.3), (IndexSource::Vector, 1.1)],
        // Explanation is semantic.
        Intent::Explanation => vec![(IndexSource::Vector, 1.3)],
        // Enumeration and aggregation want broad structured recall.
        Intent::Listing => vec![(IndexSource::Metadata, 1.3), (IndexSource::FullText, 1.2)],
        Intent::Analytical => vec![(IndexSource::Metadata, 1.4), (IndexSource::FullText, 1.2)],
        // No preference.
        Intent::Summary | Intent::General => vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_result(ids: &[ObjectId], scores: &[f32], source: IndexSource) -> IndexResult {
        IndexResult::new(ids.to_vec(), scores.to_vec(), source)
    }

    #[test]
    fn test_rerank_single_source() {
        let reranker = Reranker::new();
        let id1 = uuid::Uuid::new_v4();
        let id2 = uuid::Uuid::new_v4();

        let results = vec![make_result(&[id1, id2], &[0.9, 0.5], IndexSource::Vector)];
        let ranked = reranker.rerank_simple(&results);

        assert_eq!(ranked.len(), 2);
        assert!(ranked[0].score > ranked[1].score);
        assert_eq!(ranked[0].id, id1);
    }

    #[test]
    fn test_rerank_multi_source_boosts() {
        let reranker = Reranker::new();
        let id = uuid::Uuid::new_v4();
        let other = uuid::Uuid::new_v4();

        // ID appears in both vector and keyword search — should get boosted
        let results = vec![
            make_result(&[id, other], &[0.8, 0.6], IndexSource::Vector),
            make_result(&[id], &[0.7], IndexSource::FullText),
        ];
        let ranked = reranker.rerank_simple(&results);

        assert_eq!(ranked[0].id, id);
        assert!(ranked[0].sources.len() >= 2);
    }

    #[test]
    fn test_rerank_for_intent_boosts_graph_for_entity_search() {
        let reranker = Reranker::new();
        let graph_hit = uuid::Uuid::new_v4();
        let vector_hit = uuid::Uuid::new_v4();

        // Two distinct results, one from graph and one from vector, each rank 0
        // in its own source. Under EntitySearch the graph hit should win.
        let results = vec![
            make_result(&[vector_hit], &[0.9], IndexSource::Vector),
            make_result(&[graph_hit], &[0.9], IndexSource::Graph),
        ];

        let general = reranker.rerank_for_intent(&results, &Intent::General);
        // Base weights: Vector (1.5) > Graph (1.3) → vector hit ranks first.
        assert_eq!(general[0].id, vector_hit);

        let entity = reranker.rerank_for_intent(&results, &Intent::EntitySearch);
        // Graph ×1.5 (=1.95) now outranks Vector (1.5) → graph hit ranks first.
        assert_eq!(entity[0].id, graph_hit);
    }

    #[test]
    fn test_rerank_for_intent_general_matches_base() {
        let reranker = Reranker::new();
        let id1 = uuid::Uuid::new_v4();
        let id2 = uuid::Uuid::new_v4();
        let results = vec![make_result(&[id1, id2], &[0.9, 0.5], IndexSource::Vector)];

        let base = reranker.rerank_simple(&results);
        let general = reranker.rerank_for_intent(&results, &Intent::General);
        assert_eq!(base.len(), general.len());
        assert_eq!(base[0].id, general[0].id);
    }

    #[test]
    fn test_rerank_empty() {
        let reranker = Reranker::new();
        let ranked = reranker.rerank_simple(&[]);
        assert!(ranked.is_empty());
    }
}
