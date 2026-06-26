mod fulltext;
mod graph;
mod hnsw;
mod metadata;
mod time_index;
mod vector;

pub use fulltext::FullTextIndex;
pub use graph::GraphIndex;
pub use hnsw::{HnswIndex, HnswParams};
pub use metadata::MetadataIndex;
pub use time_index::TimeIndex;
pub use vector::VectorIndex;

/// Collects results from all indexes into a unified set.
#[derive(Debug, Clone)]
pub struct IndexResult {
    /// Object IDs matching the query.
    pub ids: Vec<kowitodb_core::ObjectId>,
    /// Scores (higher = better match), aligned with `ids`.
    pub scores: Vec<f32>,
    /// Which index produced the result.
    pub source: IndexSource,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum IndexSource {
    Vector,
    FullText,
    Metadata,
    Time,
    Graph,
}

impl IndexResult {
    pub fn new(ids: Vec<kowitodb_core::ObjectId>, scores: Vec<f32>, source: IndexSource) -> Self {
        Self {
            ids,
            scores,
            source,
        }
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}
