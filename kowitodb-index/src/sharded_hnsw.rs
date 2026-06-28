//! Sharded HNSW index — horizontal partitioning for parallel build and scale.
//!
//! Vectors are partitioned across `N` independent [`HnswIndex`] shards by object
//! id. This buys two things the single global-locked index cannot:
//!
//! - **Parallel build:** each shard is constructed on its own thread, so build
//!   throughput scales with cores ([`Self::build_parallel`]).
//! - **Horizontal scale:** shards are independent and could live on separate
//!   nodes; queries scatter to all shards and merge. (Cross-machine placement
//!   and replication are future work — this provides the in-process foundation.)
//!
//! Queries run against every shard in parallel and the local top-k results are
//! merged into a global top-k. Object ids are unique to one shard, so the merge
//! needs no de-duplication, and per-shard recall is at least as good as a single
//! index over the same points (each shard is a smaller, more accurate graph).

use std::path::Path;

use kowitodb_core::{Embedding, ObjectId};
use rayon::prelude::*;

use crate::hnsw::{HnswIndex, HnswParams};

/// An HNSW index partitioned into independent shards.
pub struct ShardedHnswIndex {
    shards: Vec<HnswIndex>,
}

impl ShardedHnswIndex {
    /// Create an index with `num_shards` shards (minimum 1).
    pub fn new(num_shards: usize, params: HnswParams) -> Self {
        let n = num_shards.max(1);
        let shards = (0..n).map(|_| HnswIndex::new(params.clone())).collect();
        Self { shards }
    }

    /// Number of shards.
    pub fn num_shards(&self) -> usize {
        self.shards.len()
    }

    /// The established vector dimension across shards (set by the first insert).
    pub fn dimension(&self) -> Option<usize> {
        self.shards.iter().find_map(|s| s.dimension())
    }

    #[inline]
    fn shard_of(&self, id: ObjectId) -> usize {
        (id.as_u128() % self.shards.len() as u128) as usize
    }

    /// Insert a single vector (routed to its shard).
    pub fn insert(&self, id: ObjectId, vector: Embedding) {
        self.shards[self.shard_of(id)].insert(id, vector);
    }

    /// Remove a vector by id.
    pub fn remove(&self, id: ObjectId) {
        self.shards[self.shard_of(id)].remove(id);
    }

    /// Build many vectors in parallel — one thread per shard.
    pub fn build_parallel(&self, items: Vec<(ObjectId, Embedding)>) {
        let n = self.shards.len();
        let mut groups: Vec<Vec<(ObjectId, Embedding)>> = (0..n).map(|_| Vec::new()).collect();
        for (id, vector) in items {
            let shard = (id.as_u128() % n as u128) as usize;
            groups[shard].push((id, vector));
        }
        self.shards
            .par_iter()
            .zip(groups.into_par_iter())
            .for_each(|(shard, group)| {
                for (id, vector) in group {
                    shard.insert(id, vector);
                }
            });
    }

    /// Query all shards in parallel and merge their local top-k into a global
    /// top-k. Higher score = closer.
    pub fn search(&self, query: &Embedding, k: usize) -> Vec<(ObjectId, f32)> {
        let mut merged: Vec<(ObjectId, f32)> = self
            .shards
            .par_iter()
            .flat_map_iter(|shard| shard.search(query, k))
            .collect();
        merged.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        merged.truncate(k);
        merged
    }

    /// Total number of vectors across all shards.
    pub fn len(&self) -> usize {
        self.shards.iter().map(|s| s.len()).sum()
    }

    /// Whether every shard is empty.
    pub fn is_empty(&self) -> bool {
        self.shards.iter().all(|s| s.is_empty())
    }

    /// Persist all shards to a single file (atomic temp + rename).
    pub fn save(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let shard_bytes: Vec<Vec<u8>> = self
            .shards
            .iter()
            .map(|s| s.to_bytes())
            .collect::<std::io::Result<_>>()?;
        let bytes = bincode::serialize(&shard_bytes).map_err(std::io::Error::other)?;
        let path = path.as_ref();
        let tmp = path.with_extension("bin.tmp");
        std::fs::write(&tmp, bytes)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Load a sharded index from `path`, or `Ok(None)` if it does not exist.
    pub fn load(path: impl AsRef<Path>) -> std::io::Result<Option<Self>> {
        let path = path.as_ref();
        if !path.exists() {
            return Ok(None);
        }
        let bytes = std::fs::read(path)?;
        let shard_bytes: Vec<Vec<u8>> =
            bincode::deserialize(&bytes).map_err(std::io::Error::other)?;
        let shards = shard_bytes
            .iter()
            .map(|b| HnswIndex::from_bytes(b))
            .collect::<std::io::Result<_>>()?;
        Ok(Some(Self { shards }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vec_for(i: usize, dim: usize) -> Vec<f32> {
        (0..dim).map(|j| ((i * 7 + j * 3) as f32).sin()).collect()
    }

    #[test]
    fn test_sharded_build_search_and_persist() {
        let params = HnswParams {
            m: 8,
            ef_construction: 50,
            ef_search: 50,
            ..Default::default()
        };
        let index = ShardedHnswIndex::new(4, params);

        let items: Vec<_> = (0..200)
            .map(|i| (uuid::Uuid::from_u128(i as u128 + 1), vec_for(i, 16)))
            .collect();
        let probe = items[42].clone();
        index.build_parallel(items);

        assert_eq!(index.len(), 200);
        assert_eq!(index.num_shards(), 4);

        // Querying with a stored vector returns it as the nearest neighbor.
        let results = index.search(&probe.1, 5);
        assert_eq!(results.len(), 5);
        assert_eq!(results[0].0, probe.0, "exact match should be top-1");
        // Scores are descending (closer first).
        for w in results.windows(2) {
            assert!(w[0].1 >= w[1].1);
        }

        // Persist and reload; results are identical.
        let path =
            std::env::temp_dir().join(format!("kowitodb-sharded-{}.bin", uuid::Uuid::new_v4()));
        index.save(&path).unwrap();
        let loaded = ShardedHnswIndex::load(&path).unwrap().unwrap();
        assert_eq!(loaded.len(), 200);
        assert_eq!(loaded.search(&probe.1, 5)[0].0, probe.0);
        let _ = std::fs::remove_file(&path);
    }
}
