use std::collections::HashMap;
use std::sync::Arc;

use kowitodb_core::{Embedding, KowitoError, ObjectId, Result};
use parking_lot::RwLock;
use tracing::debug;

/// A simple brute-force vector index for Phase 1 MVP.
///
/// In production, HNSW or DiskANN would replace the inner search algorithm.
/// This implementation keeps the interface stable and can be swapped later.
///
/// Brute-force cosine similarity is used for simplicity; HNSW is the
/// planned upgrade path per the architecture.
pub struct VectorIndex {
    /// All stored vectors, keyed by object ID and model name.
    vectors: Arc<RwLock<HashMap<(ObjectId, String), Embedding>>>,
    /// Cached dimensionality for each model.
    dimensions: Arc<RwLock<HashMap<String, usize>>>,
}

impl VectorIndex {
    pub fn new() -> Self {
        Self {
            vectors: Arc::new(RwLock::new(HashMap::new())),
            dimensions: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Insert or update a vector for an object + model combination.
    pub fn insert(&self, id: ObjectId, model: &str, vec: Embedding) -> Result<()> {
        let dim = vec.len();
        {
            let mut dims = self.dimensions.write();
            if let Some(existing) = dims.get(model) {
                if *existing != dim {
                    return Err(KowitoError::InvalidInput(format!(
                        "Embedding dimensionality mismatch for model '{}': expected {}, got {}",
                        model, existing, dim
                    )));
                }
            } else {
                dims.insert(model.to_string(), dim);
            }
        }

        let mut vectors = self.vectors.write();
        vectors.insert((id, model.to_string()), vec);
        debug!("Inserted vector for object {} (model={}, dim={})", id, model, dim);
        Ok(())
    }

    /// Remove all vectors for a given object.
    pub fn remove(&self, id: ObjectId) {
        let mut vectors = self.vectors.write();
        vectors.retain(|(obj_id, _), _| *obj_id != id);
    }

    /// Search for the top-k nearest neighbors to `query`.
    ///
    /// Returns (object_id, similarity_score) pairs sorted by descending similarity.
    pub fn search(&self, query: &Embedding, model: &str, top_k: usize) -> Result<Vec<(ObjectId, f32)>> {
        let vectors = self.vectors.read();

        // Filter to the requested model
        let candidates: Vec<_> = vectors
            .iter()
            .filter(|((_, m), _)| m == model)
            .collect();

        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        // Compute cosine similarities
        let mut scored: Vec<(ObjectId, f32)> = candidates
            .iter()
            .map(|((id, _), vec)| (*id, cosine_similarity(query, vec)))
            .collect();

        // Partial sort for top-k
        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k.min(scored.len()));

        Ok(scored)
    }

    /// Number of vectors stored.
    pub fn len(&self) -> usize {
        self.vectors.read().len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.vectors.read().is_empty()
    }
}

impl Default for VectorIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// Cosine similarity between two equal-length float slices.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_vector_index_insert_and_search() {
        let idx = VectorIndex::new();
        let id = uuid::Uuid::new_v4();
        idx.insert(id, "test-model", vec![1.0, 0.0, 0.0]).unwrap();
        idx.insert(uuid::Uuid::new_v4(), "test-model", vec![0.0, 1.0, 0.0]).unwrap();
        idx.insert(uuid::Uuid::new_v4(), "test-model", vec![0.0, 0.0, 1.0]).unwrap();

        let results = idx.search(&vec![0.9, 0.1, 0.0], "test-model", 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, id);
        assert!(results[0].1 > 0.9);
    }
}
