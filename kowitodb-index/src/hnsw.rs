//! HNSW (Hierarchical Navigable Small World) vector index.
//!
//! A graph-based approximate nearest neighbor algorithm that provides
//! logarithmic search complexity. Replaces the brute-force cosine search.
//!
//! Parameters:
//! - M: number of bidirectional connections per node per layer (default 16)
//! - ef_construction: beam width during insertion (default 200)
//! - ef_search: beam width during search (default 50)

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

use kowitodb_core::{Embedding, ObjectId};
use parking_lot::RwLock;
use rand::Rng;
use serde::{Deserialize, Serialize};
use tracing::debug;

/// Set of object ids using a fast (ahash) hasher. UUID hashing with the default
/// SipHasher dominates the hot path, so the index uses ahash everywhere ids are
/// keyed.
type IdSet = HashSet<ObjectId, ahash::RandomState>;
/// Node map keyed by object id, ahash-hashed for the same reason.
type NodeMap = HashMap<ObjectId, HnswNode, ahash::RandomState>;

/// A float wrapper that provides total ordering for BinaryHeap use.
#[derive(Debug, Clone, Copy)]
struct OrdFloat(f32);

impl PartialEq for OrdFloat {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl Eq for OrdFloat {}
impl PartialOrd for OrdFloat {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for OrdFloat {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// A node in the HNSW graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct HnswNode {
    id: ObjectId,
    vector: Embedding,
    max_layer: usize,
    /// Per-layer neighbor sets: layer → set of neighbor IDs.
    neighbors: HashMap<usize, IdSet>,
}

/// HNSW index parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswParams {
    /// Number of neighbors per node per layer (default 16).
    pub m: usize,
    /// Beam width during construction (default 200).
    pub ef_construction: usize,
    /// Beam width during search (default 50).
    pub ef_search: usize,
    /// Maximum number of nodes at layer 0 before starting layer 1, etc.
    pub m_max: usize,
    /// Multiplier for M at layer 0 (typically 2*M).
    pub m0: usize,
}

impl Default for HnswParams {
    fn default() -> Self {
        let m = 16;
        Self {
            m,
            ef_construction: 200,
            // ef_search=200 targets ~94% recall@10 on 384-dim data (see the
            // `bench_hnsw` example); ef_search=50 only reached ~60%. The modest
            // extra query latency is worth it for a knowledge DB.
            ef_search: 200,
            m_max: m,
            m0: 2 * m,
        }
    }
}

/// HNSW vector index.
///
/// Thread-safe via RwLock. Supports concurrent reads and serialized writes.
pub struct HnswIndex {
    /// All nodes, keyed by object ID.
    nodes: Arc<RwLock<NodeMap>>,
    /// Entry point (top-layer node).
    entry_point: Arc<RwLock<Option<ObjectId>>>,
    /// Current maximum layer across all nodes.
    max_layer: Arc<RwLock<usize>>,
    /// Configuration.
    params: HnswParams,
}

/// Borrowed view of the index for zero-copy serialization on `save`.
#[derive(Serialize)]
struct HnswSnapshotRef<'a> {
    params: &'a HnswParams,
    nodes: &'a NodeMap,
    entry_point: Option<ObjectId>,
    max_layer: usize,
}

/// Owned snapshot for deserialization on `load`.
#[derive(Deserialize)]
struct HnswSnapshot {
    params: HnswParams,
    nodes: NodeMap,
    entry_point: Option<ObjectId>,
    max_layer: usize,
}

impl HnswIndex {
    pub fn new(params: HnswParams) -> Self {
        Self {
            nodes: Arc::new(RwLock::new(NodeMap::default())),
            entry_point: Arc::new(RwLock::new(None)),
            max_layer: Arc::new(RwLock::new(0)),
            params,
        }
    }

    /// Insert a vector for an object.
    ///
    /// If the object already exists, it is re-inserted (updated).
    pub fn insert(&self, id: ObjectId, vector: Embedding) {
        let mut nodes = self.nodes.write();
        let mut entry_point = self.entry_point.write();
        let mut max_layer = self.max_layer.write();

        // Compute random layer using exponential distribution
        let node_layer = self.random_layer();

        let mut node = HnswNode {
            id,
            vector: vector.clone(),
            max_layer: node_layer,
            neighbors: HashMap::new(),
        };

        // If this is the first node
        let ep = match *entry_point {
            Some(ep) => ep,
            None => {
                node.neighbors.insert(0, IdSet::default());
                *max_layer = node_layer;
                nodes.insert(id, node);
                *entry_point = Some(id);
                debug!("HNSW: inserted first node {} at layer {}", id, node_layer);
                return;
            }
        };

        // Find the current entry point at the top layer.
        let mut curr_ep = ep;
        // Only the existence of the entry-point node matters here, so avoid
        // cloning the whole node (and its vector) on every descent step.
        let mut curr_ep_exists = nodes.contains_key(&ep);
        let global_max = *max_layer;

        // Greedy descent from top layer to node_layer + 1
        for lc in ((node_layer + 1)..=global_max).rev() {
            if curr_ep_exists {
                let (nearest, _) = self.search_layer_greedy(&vector, &[curr_ep], lc, 1, &nodes);
                if let Some(nearest_id) = nearest.into_iter().next() {
                    curr_ep = nearest_id;
                    curr_ep_exists = nodes.contains_key(&nearest_id);
                }
            }
        }

        // Insert at each layer from min(node_layer, global_max) down to 0
        let start_layer = node_layer.min(global_max);
        let mut ep_set = vec![curr_ep];

        for lc in (0..=start_layer).rev() {
            // Search for neighbors at this layer
            let (candidates, _) =
                self.search_layer_beam(&vector, &ep_set, lc, self.params.ef_construction, &nodes);

            // Select M (or M0 at layer 0) best neighbors
            let m = if lc == 0 {
                self.params.m0
            } else {
                self.params.m
            };
            let selected: Vec<ObjectId> =
                self.select_neighbors_heuristic(&vector, &candidates, m, lc, &nodes);

            // Add bidirectional edges
            node.neighbors.entry(lc).or_default();
            for &neighbor_id in &selected {
                node.neighbors.get_mut(&lc).unwrap().insert(neighbor_id);
                if let Some(neighbor) = nodes.get_mut(&neighbor_id) {
                    neighbor.neighbors.entry(lc).or_default();
                    neighbor.neighbors.get_mut(&lc).unwrap().insert(id);
                }
            }

            // Next layer's entry points
            ep_set = selected.clone();
        }

        // Update entry point if this node is at a higher layer
        if node_layer > global_max {
            *entry_point = Some(id);
            *max_layer = node_layer;
        }

        nodes.insert(id, node);
        debug!(
            "HNSW: inserted node {} at layer {} (global_max={})",
            id, node_layer, *max_layer
        );
    }

    /// Remove a node from the index.
    pub fn remove(&self, id: ObjectId) {
        let mut nodes = self.nodes.write();
        let mut entry_point = self.entry_point.write();

        if let Some(node) = nodes.remove(&id) {
            // Remove this node from all its neighbors
            for (layer, neighbors) in &node.neighbors {
                for neighbor_id in neighbors {
                    if let Some(neighbor) = nodes.get_mut(neighbor_id) {
                        if let Some(nbr_set) = neighbor.neighbors.get_mut(layer) {
                            nbr_set.remove(&id);
                        }
                    }
                }
            }
        }

        // Update entry point if it was this node
        if entry_point.is_some_and(|ep| ep == id) {
            *entry_point = nodes.keys().next().copied();
        }
    }

    /// Search for the k-nearest neighbors.
    pub fn search(&self, query: &Embedding, k: usize) -> Vec<(ObjectId, f32)> {
        let nodes = self.nodes.read();
        let entry_point = self.entry_point.read();
        let max_layer = self.max_layer.read();

        let ep = match *entry_point {
            Some(ep) => ep,
            None => return Vec::new(),
        };

        // Greedy descent from top layer to layer 1
        let mut curr_ep = ep;
        let global_max = *max_layer;

        for lc in (1..=global_max).rev() {
            let (nearest, _) = self.search_layer_greedy(query, &[curr_ep], lc, 1, &nodes);
            if let Some(nearest_id) = nearest.into_iter().next() {
                curr_ep = nearest_id;
            }
        }

        // Beam search at layer 0
        let ef = self.params.ef_search.max(k);
        let (candidates, distances) = self.search_layer_beam(query, &[curr_ep], 0, ef, &nodes);

        // Take top-k
        let mut results: Vec<(ObjectId, f32)> = candidates.into_iter().zip(distances).collect();

        results.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        results.truncate(k);

        // Convert squared distance to a Euclidean-distance similarity
        // (1 / (1 + distance)). sqrt is applied only to the k returned results.
        results
            .into_iter()
            .map(|(id, dist)| (id, 1.0 / (1.0 + dist.sqrt())))
            .collect()
    }

    /// Number of nodes in the index.
    pub fn len(&self) -> usize {
        self.nodes.read().len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.read().is_empty()
    }

    /// Persist the index to `path` (atomic write via a temp file + rename).
    pub fn save(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let path = path.as_ref();
        let nodes = self.nodes.read();
        let snapshot = HnswSnapshotRef {
            params: &self.params,
            nodes: &nodes,
            entry_point: *self.entry_point.read(),
            max_layer: *self.max_layer.read(),
        };
        let bytes = bincode::serialize(&snapshot).map_err(std::io::Error::other)?;
        let tmp = path.with_extension("bin.tmp");
        std::fs::write(&tmp, bytes)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Load an index from `path`, or `Ok(None)` if the file does not exist.
    pub fn load(path: impl AsRef<Path>) -> std::io::Result<Option<Self>> {
        let path = path.as_ref();
        if !path.exists() {
            return Ok(None);
        }
        let bytes = std::fs::read(path)?;
        let snapshot: HnswSnapshot = bincode::deserialize(&bytes).map_err(std::io::Error::other)?;
        Ok(Some(Self {
            nodes: Arc::new(RwLock::new(snapshot.nodes)),
            entry_point: Arc::new(RwLock::new(snapshot.entry_point)),
            max_layer: Arc::new(RwLock::new(snapshot.max_layer)),
            params: snapshot.params,
        }))
    }

    // ---- Internal methods ----

    /// Generate a random layer using exponential decay.
    fn random_layer(&self) -> usize {
        let mut rng = rand::thread_rng();
        let ml: f64 = 1.0 / (self.params.m as f64).ln();
        let r: f64 = rng.gen();
        ((-r.ln() * ml).floor() as usize).min(10) // Cap at layer 10
    }

    /// Greedy 1-nearest-neighbor search on a single layer.
    fn search_layer_greedy(
        &self,
        query: &[f32],
        entry_points: &[ObjectId],
        layer: usize,
        _ef: usize,
        nodes: &NodeMap,
    ) -> (Vec<ObjectId>, Vec<f32>) {
        let mut best_id = entry_points[0];
        let mut best_dist = squared_dist(query, &nodes[&best_id].vector);

        loop {
            let mut improved = false;
            // Iterate the current node's neighbors in place (no per-step alloc).
            if let Some(neighbor_set) = nodes.get(&best_id).and_then(|n| n.neighbors.get(&layer)) {
                for &neighbor_id in neighbor_set {
                    if let Some(neighbor) = nodes.get(&neighbor_id) {
                        let dist = squared_dist(query, &neighbor.vector);
                        if dist < best_dist {
                            best_dist = dist;
                            best_id = neighbor_id;
                            improved = true;
                        }
                    }
                }
            }

            if !improved {
                break;
            }
        }

        (vec![best_id], vec![best_dist])
    }

    /// Beam search on a single layer.
    fn search_layer_beam(
        &self,
        query: &[f32],
        entry_points: &[ObjectId],
        layer: usize,
        ef: usize,
        nodes: &NodeMap,
    ) -> (Vec<ObjectId>, Vec<f32>) {
        #[derive(Debug)]
        struct Candidate {
            id: ObjectId,
            /// Distance (smaller = better)
            dist: OrdFloat,
        }
        impl PartialEq for Candidate {
            fn eq(&self, other: &Self) -> bool {
                self.dist == other.dist
            }
        }
        impl Eq for Candidate {}
        impl PartialOrd for Candidate {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for Candidate {
            fn cmp(&self, other: &Self) -> Ordering {
                // Max-heap: larger OrdFloat = larger distance = worse
                // But we want min-heap (smallest distance first), so reverse
                other.dist.cmp(&self.dist)
            }
        }

        let mut visited: IdSet = IdSet::default();
        // candidates: min-heap (closest first)
        let mut candidates = BinaryHeap::new();
        // results: max-heap (worst first, for eviction). Since Candidate is
        // reversed, worst (largest distance) pops first from the BinaryHeap.
        // We need a "worst-heap" — store with distance directly (not reversed).
        #[derive(Debug)]
        struct WorstCandidate {
            id: ObjectId,
            dist: OrdFloat,
        }
        impl PartialEq for WorstCandidate {
            fn eq(&self, other: &Self) -> bool {
                self.dist == other.dist
            }
        }
        impl Eq for WorstCandidate {}
        impl PartialOrd for WorstCandidate {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for WorstCandidate {
            fn cmp(&self, other: &Self) -> Ordering {
                self.dist.cmp(&other.dist)
            }
        }
        let mut results: BinaryHeap<WorstCandidate> = BinaryHeap::new();

        for &ep in entry_points {
            let dist = squared_dist(query, &nodes[&ep].vector);
            candidates.push(Candidate {
                id: ep,
                dist: OrdFloat(dist),
            });
            results.push(WorstCandidate {
                id: ep,
                dist: OrdFloat(dist),
            });
            visited.insert(ep);
        }

        while let Some(current) = candidates.pop() {
            let current_dist = current.dist.0;

            // Stop if current is farther than the worst result we're keeping
            if results.len() >= ef {
                if let Some(worst) = results.peek() {
                    if current_dist >= worst.dist.0 {
                        break;
                    }
                }
            }

            // Expand neighbors in place; `visited.insert` returns false when the
            // id was already present, combining the contains+insert check.
            if let Some(neighbor_set) = nodes.get(&current.id).and_then(|n| n.neighbors.get(&layer))
            {
                for &neighbor_id in neighbor_set {
                    if !visited.insert(neighbor_id) {
                        continue;
                    }
                    if let Some(neighbor) = nodes.get(&neighbor_id) {
                        let dist = squared_dist(query, &neighbor.vector);
                        let od = OrdFloat(dist);

                        let should_add = results.len() < ef
                            || dist < results.peek().map(|c| c.dist.0).unwrap_or(f32::MAX);

                        if should_add {
                            candidates.push(Candidate {
                                id: neighbor_id,
                                dist: od,
                            });
                            results.push(WorstCandidate {
                                id: neighbor_id,
                                dist: od,
                            });
                            if results.len() > ef {
                                results.pop();
                            }
                        }
                    }
                }
            }
        }

        // Collect results (worst-first in heap, so reverse for closest-first)
        let mut ids = Vec::with_capacity(results.len());
        let mut dists = Vec::with_capacity(results.len());
        let mut temp = Vec::new();
        while let Some(c) = results.pop() {
            temp.push((c.id, c.dist.0));
        }
        for (id, dist) in temp.into_iter().rev() {
            ids.push(id);
            dists.push(dist);
        }

        (ids, dists)
    }

    /// Heuristic neighbor selection (prunes candidates for better graph quality).
    fn select_neighbors_heuristic(
        &self,
        query: &[f32],
        candidates: &[ObjectId],
        m: usize,
        _layer: usize,
        nodes: &NodeMap,
    ) -> Vec<ObjectId> {
        if candidates.len() <= m {
            return candidates.to_vec();
        }

        // Sort candidates by distance
        let mut scored: Vec<(ObjectId, f32)> = candidates
            .iter()
            .map(|&id| {
                let dist = squared_dist(query, &nodes[&id].vector);
                (id, dist)
            })
            .collect();
        scored.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        // Simple pruning: keep closest m candidates
        let mut selected = Vec::with_capacity(m);
        for (id, _) in scored.into_iter().take(m) {
            selected.push(id);
        }

        selected
    }
}

/// Squared Euclidean distance between two vectors.
///
/// HNSW only ever *compares* distances, and squared distance preserves ordering,
/// so the `sqrt` is dropped from the inner loop — it is applied only to the
/// final k results when converting to a similarity score. The simple loop
/// auto-vectorizes (NEON/SSE) under the release profile.
#[inline]
fn squared_dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hnsw_insert_and_search() {
        let idx = HnswIndex::new(HnswParams {
            m: 8,
            ef_construction: 50,
            ef_search: 20,
            ..Default::default()
        });

        // Insert 50 random vectors
        let mut ids = Vec::new();
        for i in 0..50 {
            let id = uuid::Uuid::new_v4();
            let vec: Vec<f32> = (0..16).map(|j| ((i * 7 + j * 3) as f32).sin()).collect();
            idx.insert(id, vec);
            ids.push(id);
        }

        // Search should return results
        let query: Vec<f32> = (0..16)
            .map(|j| (25.0 * 7.0 + j as f32 * 3.0).sin())
            .collect();
        let results = idx.search(&query, 5);
        assert_eq!(results.len(), 5);
        // Scores should be in descending order (similarity)
        for w in results.windows(2) {
            assert!(
                w[0].1 >= w[1].1,
                "Results should be sorted by descending score"
            );
        }
    }

    #[test]
    fn test_hnsw_save_load_roundtrip() {
        let idx = HnswIndex::new(HnswParams {
            m: 8,
            ef_construction: 50,
            ef_search: 20,
            ..Default::default()
        });
        let mut ids = Vec::new();
        for i in 0..50 {
            let id = uuid::Uuid::new_v4();
            let vec: Vec<f32> = (0..16).map(|j| ((i * 7 + j * 3) as f32).sin()).collect();
            idx.insert(id, vec);
            ids.push(id);
        }

        let path = std::env::temp_dir().join(format!("kowitodb-hnsw-{}.bin", uuid::Uuid::new_v4()));
        idx.save(&path).unwrap();

        // Loading a missing file yields None.
        let missing = std::env::temp_dir().join("kowitodb-hnsw-does-not-exist.bin");
        assert!(HnswIndex::load(&missing).unwrap().is_none());

        let loaded = HnswIndex::load(&path)
            .unwrap()
            .expect("snapshot should load");
        assert_eq!(loaded.len(), idx.len());

        // The loaded index returns the same neighbors as the original.
        let query: Vec<f32> = (0..16)
            .map(|j| (25.0 * 7.0 + j as f32 * 3.0).sin())
            .collect();
        let before = idx.search(&query, 5);
        let after = loaded.search(&query, 5);
        let before_ids: Vec<_> = before.iter().map(|(id, _)| *id).collect();
        let after_ids: Vec<_> = after.iter().map(|(id, _)| *id).collect();
        assert_eq!(
            before_ids, after_ids,
            "search results must survive save/load"
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_hnsw_empty_search() {
        let idx = HnswIndex::new(HnswParams::default());
        let results = idx.search(&vec![1.0, 2.0, 3.0], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_hnsw_remove() {
        let idx = HnswIndex::new(HnswParams {
            m: 4,
            ef_construction: 20,
            ef_search: 10,
            ..Default::default()
        });

        let id = uuid::Uuid::new_v4();
        idx.insert(id, vec![1.0, 0.0, 0.0]);
        idx.insert(uuid::Uuid::new_v4(), vec![0.0, 1.0, 0.0]);

        assert_eq!(idx.len(), 2);
        idx.remove(id);
        assert_eq!(idx.len(), 1);

        let results = idx.search(&vec![0.9, 0.1, 0.0], 3);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_squared_dist() {
        let a = vec![0.0, 3.0, 4.0];
        let b = vec![0.0, 0.0, 0.0];
        // 3^2 + 4^2 = 25 (squared distance — no sqrt).
        assert!((squared_dist(&a, &b) - 25.0).abs() < 1e-6);
    }
}
