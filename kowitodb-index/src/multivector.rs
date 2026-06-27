//! Late-interaction (ColBERTv2-style) multi-vector index.
//!
//! Each object is represented by a **set** of token/chunk vectors rather than a
//! single pooled vector. Relevance is **MaxSim**: for every query token, take
//! its maximum similarity to any document token, and sum those maxima. This
//! preserves fine-grained term-level matching that a single pooled vector
//! averages away — the highest quality ceiling among dense retrievers
//! (ColBERTv2, arXiv 2112.01488).
//!
//! The token vectors come from a late-interaction model (e.g. ColBERT); this
//! crate provides the index + MaxSim scoring (deterministic, model-agnostic,
//! and unit-tested). Vectors are L2-normalized on insert so a dot product is
//! cosine similarity.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use kowitodb_core::ObjectId;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

/// A document's token vectors (each L2-normalized).
type TokenVectors = Vec<Vec<f32>>;

/// L2-normalize a vector in place; a zero vector is left unchanged.
fn normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Cosine similarity between two already-normalized vectors (a dot product).
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// MaxSim of a normalized query against a document's normalized tokens:
/// `Σ_q max_d (q · d)`. Returns 0 for an empty document.
fn maxsim(query: &[Vec<f32>], doc: &[Vec<f32>]) -> f32 {
    query
        .iter()
        .map(|q| {
            doc.iter()
                .map(|d| dot(q, d))
                .fold(f32::NEG_INFINITY, f32::max)
                .max(0.0) // a token with no positive match contributes nothing
        })
        .sum()
}

/// In-memory late-interaction index keyed by object id.
pub struct MultiVectorIndex {
    docs: Arc<RwLock<HashMap<ObjectId, TokenVectors>>>,
}

#[derive(Serialize, Deserialize)]
struct Snapshot {
    docs: HashMap<ObjectId, TokenVectors>,
}

impl MultiVectorIndex {
    pub fn new() -> Self {
        Self {
            docs: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Insert (or replace) an object's token vectors. Each vector is normalized.
    pub fn insert(&self, id: ObjectId, mut tokens: TokenVectors) {
        for t in tokens.iter_mut() {
            normalize(t);
        }
        self.docs.write().insert(id, tokens);
    }

    /// Remove an object's token vectors.
    pub fn remove(&self, id: ObjectId) {
        self.docs.write().remove(&id);
    }

    pub fn len(&self) -> usize {
        self.docs.read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.docs.read().is_empty()
    }

    /// Top-`k` objects by MaxSim against the multi-vector `query`. The query
    /// tokens are normalized first. When `candidates` is `Some`, only those ids
    /// are scored (the production two-stage pattern: ANN to shortlist, MaxSim to
    /// rerank); otherwise every document is scored (brute force).
    pub fn search(
        &self,
        query: &[Vec<f32>],
        k: usize,
        candidates: Option<&[ObjectId]>,
    ) -> Vec<(ObjectId, f32)> {
        if query.is_empty() {
            return Vec::new();
        }
        let mut q = query.to_vec();
        for t in q.iter_mut() {
            normalize(t);
        }

        let docs = self.docs.read();
        let mut scored: Vec<(ObjectId, f32)> = match candidates {
            Some(ids) => ids
                .iter()
                .filter_map(|id| docs.get(id).map(|d| (*id, maxsim(&q, d))))
                .collect(),
            None => docs.iter().map(|(id, d)| (*id, maxsim(&q, d))).collect(),
        };
        scored.sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        scored.truncate(k);
        scored
    }

    /// Serialize to a byte buffer.
    pub fn to_bytes(&self) -> std::io::Result<Vec<u8>> {
        let snapshot = Snapshot {
            docs: self.docs.read().clone(),
        };
        bincode::serialize(&snapshot).map_err(std::io::Error::other)
    }

    /// Reconstruct from a buffer produced by [`Self::to_bytes`].
    pub fn from_bytes(bytes: &[u8]) -> std::io::Result<Self> {
        let snapshot: Snapshot = bincode::deserialize(bytes).map_err(std::io::Error::other)?;
        Ok(Self {
            docs: Arc::new(RwLock::new(snapshot.docs)),
        })
    }

    /// Persist to `path` (atomic temp + rename).
    pub fn save(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let path = path.as_ref();
        let tmp = path.with_extension("bin.tmp");
        std::fs::write(&tmp, self.to_bytes()?)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Load from `path`, or `Ok(None)` if it does not exist.
    pub fn load(path: impl AsRef<Path>) -> std::io::Result<Option<Self>> {
        let path = path.as_ref();
        if !path.exists() {
            return Ok(None);
        }
        Ok(Some(Self::from_bytes(&std::fs::read(path)?)?))
    }
}

impl Default for MultiVectorIndex {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn e(i: usize, dim: usize) -> Vec<f32> {
        let mut v = vec![0.0; dim];
        v[i] = 1.0;
        v
    }

    #[test]
    fn test_maxsim_token_level_matching() {
        let idx = MultiVectorIndex::new();
        let a = uuid::Uuid::from_u128(1);
        let b = uuid::Uuid::from_u128(2);
        // Doc A covers basis tokens e0,e1; doc B covers e2,e3.
        idx.insert(a, vec![e(0, 4), e(1, 4)]);
        idx.insert(b, vec![e(2, 4), e(3, 4)]);

        // A query token aligned with e0 matches A perfectly, B not at all.
        let res = idx.search(&[e(0, 4)], 2, None);
        assert_eq!(res[0].0, a);
        assert!((res[0].1 - 1.0).abs() < 1e-6);
        assert!(res[1].1.abs() < 1e-6);

        // A two-token query split across the docs scores both, but the doc that
        // covers more query tokens wins.
        let res = idx.search(&[e(2, 4), e(3, 4)], 2, None);
        assert_eq!(res[0].0, b, "B covers both query tokens → highest MaxSim");
        assert!((res[0].1 - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_candidate_restriction() {
        let idx = MultiVectorIndex::new();
        let a = uuid::Uuid::from_u128(1);
        let b = uuid::Uuid::from_u128(2);
        idx.insert(a, vec![e(0, 4)]);
        idx.insert(b, vec![e(0, 4)]);
        // Restrict scoring to B only (two-stage rerank shortlist).
        let res = idx.search(&[e(0, 4)], 5, Some(&[b]));
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].0, b);
    }

    #[test]
    fn test_save_load_roundtrip() {
        let idx = MultiVectorIndex::new();
        let a = uuid::Uuid::from_u128(1);
        idx.insert(a, vec![e(0, 4), e(1, 4)]);
        let path = std::env::temp_dir().join(format!("kowitodb-mv-{}.bin", uuid::Uuid::new_v4()));
        idx.save(&path).unwrap();
        let loaded = MultiVectorIndex::load(&path).unwrap().expect("loads");
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded.search(&[e(0, 4)], 1, None)[0].0, a);
        let _ = std::fs::remove_file(&path);
    }
}
