use std::collections::HashMap;

use kowitodb_core::ObjectId;
use kowitodb_index::{IndexResult, IndexSource, VectorIndex};
use tracing::debug;

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
        // Phase 1: RRF scoring
        let mut scored: HashMap<ObjectId, RankedResult> = HashMap::new();

        for result in raw_results {
            let source_weight = self
                .source_weights
                .get(&result.source)
                .copied()
                .unwrap_or(1.0);

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
    fn test_rerank_empty() {
        let reranker = Reranker::new();
        let ranked = reranker.rerank_simple(&[]);
        assert!(ranked.is_empty());
    }
}
