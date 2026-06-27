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

/// int8 quantization scale. Assumes ~unit-norm vectors (components in [-1, 1]),
/// as produced by the embedding models KowitoDB uses.
const QUANT_SCALE: f32 = 127.0;
/// Reciprocal of [`QUANT_SCALE`] — dequantize by multiply (faster than divide).
const INV_QUANT_SCALE: f32 = 1.0 / QUANT_SCALE;

/// Fixed seed for the structured random rotation used by binary quantization.
/// Deterministic so a saved index reloads with the same basis.
const ROTATION_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

/// In-place fast Walsh–Hadamard transform. `a.len()` must be a power of two.
fn fwht(a: &mut [f32]) {
    let n = a.len();
    let mut h = 1;
    while h < n {
        let mut i = 0;
        while i < n {
            for j in i..i + h {
                let x = a[j];
                let y = a[j + h];
                a[j] = x + y;
                a[j + h] = x - y;
            }
            i += 2 * h;
        }
        h *= 2;
    }
}

/// A structured random rotation (random ±1 sign flip followed by a normalized
/// fast Walsh–Hadamard transform). Orthonormal, so it preserves L2 distances
/// while decorrelating coordinates — the precondition that makes 1-bit
/// (sign) quantization a well-behaved estimator (RaBitQ, SIGMOD 2024). Cheap
/// O(d log d) to apply and trivially serializable (just the sign vector).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Rotation {
    /// ±1 per padded dimension.
    signs: Vec<f32>,
    /// Working dimension (the original dim rounded up to a power of two).
    dim_padded: usize,
}

impl Rotation {
    fn new(orig_dim: usize, seed: u64) -> Self {
        let dim_padded = orig_dim.max(1).next_power_of_two();
        // Deterministic ±1 signs from a SplitMix64-style stream.
        let mut state = seed | 1;
        let signs = (0..dim_padded)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                if (state >> 63) & 1 == 1 {
                    1.0
                } else {
                    -1.0
                }
            })
            .collect();
        Self { signs, dim_padded }
    }

    /// Rotate `v` into the working basis (length `dim_padded`).
    fn apply(&self, v: &[f32]) -> Vec<f32> {
        let mut buf = vec![0.0f32; self.dim_padded];
        for i in 0..v.len().min(self.dim_padded) {
            buf[i] = v[i] * self.signs[i];
        }
        fwht(&mut buf);
        let scale = 1.0 / (self.dim_padded as f32).sqrt();
        for x in &mut buf {
            *x *= scale;
        }
        buf
    }
}

/// Stored vector — full f32, int8-quantized (4× smaller), or RaBitQ-style
/// 1-bit binary (~32× smaller).
#[derive(Debug, Clone, Serialize, Deserialize)]
enum NodeVector {
    Full(Vec<f32>),
    /// Scalar-quantized to int8 at `QUANT_SCALE`; dequantized on the fly.
    Quantized(Vec<i8>),
    /// RaBitQ-style 1-bit code over the *rotated* vector: one sign bit per
    /// working dimension plus two scalars for an unbiased distance estimator.
    /// Queries are searched in the same rotated basis (see [`Rotation`]).
    Binary {
        /// Sign bits, packed 64 per word (bit set ⇔ rotated component ≥ 0).
        code: Vec<u64>,
        /// ‖o‖² of the original vector.
        norm_sq: f32,
        /// `‖o‖²·√D / Σ|õ_i|` — rescales the sign-dot into an inner-product
        /// estimate (handles the quantization-induced shrinkage).
        factor: f32,
        /// Working (padded) dimension.
        dim_padded: usize,
    },
}

impl NodeVector {
    /// Build from a full-precision vector, quantizing if requested.
    fn new(vector: Vec<f32>, quantize: bool) -> Self {
        if quantize {
            NodeVector::Quantized(
                vector
                    .iter()
                    .map(|x| (x * QUANT_SCALE).round().clamp(-127.0, 127.0) as i8)
                    .collect(),
            )
        } else {
            NodeVector::Full(vector)
        }
    }

    /// Build a 1-bit binary code from an already-rotated vector `rotated`
    /// (length `dim_padded`).
    fn new_binary(rotated: &[f32], dim_padded: usize) -> Self {
        let code = pack_sign_code(rotated, dim_padded);
        let mut abs_sum = 0.0f32;
        let mut norm_sq = 0.0f32;
        for &x in rotated.iter().take(dim_padded) {
            norm_sq += x * x;
            abs_sum += x.abs();
        }
        let factor = if abs_sum > 0.0 {
            norm_sq * (dim_padded as f32).sqrt() / abs_sum
        } else {
            0.0
        };
        NodeVector::Binary {
            code,
            norm_sq,
            factor,
            dim_padded,
        }
    }

    /// Hamming distance (as `f32`) between this node's sign code and a query's
    /// sign `code` — the popcount fast path for binary navigation. Only valid
    /// for `Binary` nodes (the only kind present under binary quantization).
    #[inline]
    fn hamming(&self, query_code: &[u64]) -> f32 {
        match self {
            NodeVector::Binary { code, .. } => {
                let mut d = 0u32;
                for (a, b) in code.iter().zip(query_code) {
                    d += (a ^ b).count_ones();
                }
                d as f32
            }
            _ => f32::MAX,
        }
    }

    /// Squared Euclidean distance to a query vector. For `Full`/`Quantized` the
    /// query is in the original space; for `Binary` it is the **rotated** query
    /// (length `dim_padded`), and the result is the RaBitQ distance *estimate*.
    #[inline]
    fn dist_sq(&self, query: &[f32]) -> f32 {
        match self {
            NodeVector::Full(v) => squared_dist(query, v),
            NodeVector::Quantized(q) => int8_dist_sq(query, q),
            NodeVector::Binary {
                code,
                norm_sq,
                factor,
                dim_padded,
            } => {
                let mut signed_sum = 0.0f32;
                let mut q_norm_sq = 0.0f32;
                for (i, &qi) in query.iter().enumerate().take(*dim_padded) {
                    q_norm_sq += qi * qi;
                    let bit = (code[i / 64] >> (i % 64)) & 1;
                    if bit == 1 {
                        signed_sum += qi;
                    } else {
                        signed_sum -= qi;
                    }
                }
                let inv_sqrt = 1.0 / (*dim_padded as f32).sqrt();
                let ip_est = *factor * signed_sum * inv_sqrt;
                (*norm_sq + q_norm_sq - 2.0 * ip_est).max(0.0)
            }
        }
    }

    /// Distance using only the first `coarse` dimensions (Matryoshka coarse
    /// pass) when `Some`, else the full distance. Prefix scoring is valid for
    /// `Full`/`Quantized` (where prefixes of MRL embeddings are themselves
    /// embeddings); `Binary` rotates the space so prefixes are meaningless and
    /// it falls back to the full estimator.
    #[inline]
    fn dist_sq_coarse(&self, query: &[f32], coarse: Option<usize>) -> f32 {
        let Some(d) = coarse else {
            return self.dist_sq(query);
        };
        match self {
            NodeVector::Full(v) => {
                let n = d.min(v.len()).min(query.len());
                squared_dist(&query[..n], &v[..n])
            }
            NodeVector::Quantized(q) => {
                let n = d.min(q.len()).min(query.len());
                query[..n]
                    .iter()
                    .zip(&q[..n])
                    .map(|(x, &qi)| {
                        let e = x - qi as f32 / QUANT_SCALE;
                        e * e
                    })
                    .sum()
            }
            NodeVector::Binary { .. } => self.dist_sq(query),
        }
    }
}

/// Pack the sign bits of `rotated` (≥0 ⇒ bit set) into 64-bit words.
fn pack_sign_code(rotated: &[f32], dim_padded: usize) -> Vec<u64> {
    let mut code = vec![0u64; dim_padded.div_ceil(64)];
    for (i, &x) in rotated.iter().enumerate().take(dim_padded) {
        if x >= 0.0 {
            code[i / 64] |= 1u64 << (i % 64);
        }
    }
    code
}

/// How the graph traversal scores a candidate node against the query. Built
/// once per search/insert and passed down to the layer searches, so the hot
/// loop dispatches on a cheap enum rather than re-deciding per node.
enum Scorer<'a> {
    /// Full (or `coarse`-prefix) f32 distance against the query — the query is
    /// in the original space, or the rotated space under binary quantization.
    Float {
        query: &'a [f32],
        coarse: Option<usize>,
    },
    /// Popcount Hamming distance against a precomputed query sign code — the
    /// binary fast path (no float ops during navigation).
    Hamming { code: &'a [u64] },
}

impl Scorer<'_> {
    #[inline]
    fn score(&self, v: &NodeVector) -> f32 {
        match self {
            Scorer::Float { query, coarse } => v.dist_sq_coarse(query, *coarse),
            Scorer::Hamming { code } => v.hamming(code),
        }
    }
}

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

/// Min-heap entry (closest popped first) for the beam frontier.
#[derive(Debug)]
struct Candidate {
    id: ObjectId,
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
        // Reverse so the BinaryHeap (a max-heap) yields the smallest distance.
        other.dist.cmp(&self.dist)
    }
}

/// Max-heap entry (worst popped first) for the bounded result set.
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

/// Per-thread reusable scratch for beam search, so the hot query path allocates
/// nothing after warmup (allocator contention otherwise caps concurrent QPS).
/// Safe because a search runs to completion synchronously on one thread.
#[derive(Default)]
struct BeamScratch {
    visited: IdSet,
    candidates: BinaryHeap<Candidate>,
    results: BinaryHeap<WorstCandidate>,
}

thread_local! {
    static BEAM_SCRATCH: std::cell::RefCell<BeamScratch> =
        std::cell::RefCell::new(BeamScratch::default());
}

/// A node in the HNSW graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct HnswNode {
    id: ObjectId,
    vector: NodeVector,
    max_layer: usize,
    /// Per-layer neighbor sets: layer → set of neighbor IDs.
    neighbors: HashMap<usize, IdSet>,
    /// Optional higher-fidelity vector (int8) retained *only* to re-score the
    /// final top-k under binary quantization — the oversample→rescore pattern
    /// that recovers recall the 1-bit codes lose. `None` unless
    /// `binary_rerank` is set. Stored in the *original* (unrotated) space.
    #[serde(default)]
    rerank: Option<NodeVector>,
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
    /// Store vectors int8-quantized (4× less memory, slight recall cost).
    /// Off by default; best for normalized embeddings.
    #[serde(default)]
    pub quantize: bool,
    /// Store vectors RaBitQ-style 1-bit binary (~32× less memory) over a
    /// structured random rotation. Off by default; takes precedence over
    /// `quantize` when both are set. Recall is lower than full/int8 but the
    /// memory win is the lever for very large in-RAM collections.
    #[serde(default)]
    pub binary_quantize: bool,
    /// Matryoshka adaptive retrieval: when `Some(d)`, navigate the graph using
    /// only the first `d` vector dimensions (a cheap coarse pass) and refine the
    /// final top-k with full-dimension distances. Requires MRL-trained
    /// embeddings (valid prefixes). Ignored under binary quantization. `None`
    /// (default) searches at full precision throughout.
    #[serde(default)]
    pub coarse_dim: Option<usize>,
    /// Under binary quantization, retain an int8 copy of each vector and
    /// re-score the oversampled top-k with it (the production oversample→rescore
    /// pattern). Recovers most of the recall the 1-bit codes lose while keeping
    /// fast popcount navigation; memory is ~int8 (¼×) rather than 1/32×. No
    /// effect unless `binary_quantize` is also set. Off by default.
    #[serde(default)]
    pub binary_rerank: bool,
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
            quantize: false,
            binary_quantize: false,
            coarse_dim: None,
            binary_rerank: false,
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
    /// Structured random rotation for binary quantization (lazily created on
    /// the first insert once the dimensionality is known). `None` unless
    /// `params.binary_quantize` is set.
    rotation: Arc<RwLock<Option<Rotation>>>,
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
    #[serde(default)]
    rotation: Option<Rotation>,
}

/// Owned snapshot for deserialization on `load`.
#[derive(Deserialize)]
struct HnswSnapshot {
    params: HnswParams,
    nodes: NodeMap,
    entry_point: Option<ObjectId>,
    max_layer: usize,
    #[serde(default)]
    rotation: Option<Rotation>,
}

impl HnswIndex {
    pub fn new(params: HnswParams) -> Self {
        Self {
            nodes: Arc::new(RwLock::new(NodeMap::default())),
            entry_point: Arc::new(RwLock::new(None)),
            max_layer: Arc::new(RwLock::new(0)),
            rotation: Arc::new(RwLock::new(None)),
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

        // Binary quantization navigates the graph in a rotated basis via the
        // Hamming (popcount) fast path; otherwise it scores with full f32
        // distance. `search_vec` is the rotated/raw query used for the more
        // accurate neighbor *selection* step.
        let (node_vector, rotated, query_code) = if self.params.binary_quantize {
            let mut rot = self.rotation.write();
            if rot.is_none() {
                *rot = Some(Rotation::new(vector.len(), ROTATION_SEED));
            }
            let r = rot.as_ref().unwrap();
            let rotated = r.apply(&vector);
            let code = pack_sign_code(&rotated, r.dim_padded);
            (
                NodeVector::new_binary(&rotated, r.dim_padded),
                Some(rotated),
                Some(code),
            )
        } else {
            (
                NodeVector::new(vector.clone(), self.params.quantize),
                None,
                None,
            )
        };
        let search_vec: &[f32] = rotated.as_deref().unwrap_or(&vector);
        let scorer = match &query_code {
            Some(code) => Scorer::Hamming { code },
            None => Scorer::Float {
                query: search_vec,
                coarse: None,
            },
        };

        // Under binary quantization, optionally retain an int8 copy (original
        // space) to re-score the final top-k for higher recall.
        let rerank = if self.params.binary_quantize && self.params.binary_rerank {
            Some(NodeVector::new(vector.clone(), true))
        } else {
            None
        };

        let mut node = HnswNode {
            id,
            vector: node_vector,
            max_layer: node_layer,
            neighbors: HashMap::new(),
            rerank,
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
                let nearest_id = self.search_layer_greedy(&scorer, curr_ep, lc, &nodes);
                curr_ep = nearest_id;
                curr_ep_exists = nodes.contains_key(&nearest_id);
            }
        }

        // Insert at each layer from min(node_layer, global_max) down to 0
        let start_layer = node_layer.min(global_max);
        let mut ep_set = vec![curr_ep];

        for lc in (0..=start_layer).rev() {
            // Search for neighbors at this layer
            let (candidates, _) =
                self.search_layer_beam(&scorer, &ep_set, lc, self.params.ef_construction, &nodes);

            // Select M (or M0 at layer 0) best neighbors
            let m = if lc == 0 {
                self.params.m0
            } else {
                self.params.m
            };
            let selected: Vec<ObjectId> =
                self.select_neighbors_heuristic(search_vec, &candidates, m, lc, &nodes);

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

        // The original (unrotated) query, kept for re-scoring against retained
        // int8 vectors under binary quantization.
        let orig_query: &[f32] = query;

        // Rotate (and sign-code) the query once for binary mode; the graph then
        // navigates via the Hamming popcount fast path and the final candidates
        // are refined with the accurate asymmetric estimator. The rotation lock
        // is only touched when binary quantization is on, so the common path
        // avoids it entirely.
        let (rotated, query_code) = if self.params.binary_quantize {
            match self.rotation.read().as_ref() {
                Some(r) => {
                    let rv = r.apply(query);
                    let code = pack_sign_code(&rv, r.dim_padded);
                    (Some(rv), Some(code))
                }
                None => (None, None),
            }
        } else {
            (None, None)
        };
        // Matryoshka coarse pass — navigate with a dimension prefix, then refine
        // the final candidates at full dimension. Disabled under binary mode,
        // where prefixes of the rotated vector are meaningless.
        let coarse = if query_code.is_some() {
            None
        } else {
            self.params.coarse_dim.filter(|&d| d > 0 && d < query.len())
        };
        let query: &[f32] = rotated.as_deref().unwrap_or(query);
        let scorer = match &query_code {
            Some(code) => Scorer::Hamming { code },
            None => Scorer::Float { query, coarse },
        };
        // A coarse or binary (Hamming) navigation is approximate, so the top-k
        // is re-scored at full fidelity before returning.
        let refine = query_code.is_some() || coarse.is_some();

        // Greedy descent from top layer to layer 1
        let mut curr_ep = ep;
        let global_max = *max_layer;

        for lc in (1..=global_max).rev() {
            curr_ep = self.search_layer_greedy(&scorer, curr_ep, lc, &nodes);
        }

        // Beam search at layer 0. When refining, over-fetch candidates so the
        // re-score below has a good pool to re-rank.
        let ef = if refine {
            self.params.ef_search.max(k * 4)
        } else {
            self.params.ef_search.max(k)
        };
        let (candidates, distances) = self.search_layer_beam(&scorer, &[curr_ep], 0, ef, &nodes);

        // Take top-k. When navigation was approximate, re-score candidates with
        // the accurate distance (full-dim for coarse, asymmetric RaBitQ
        // estimator for binary); otherwise use the beam's distances directly.
        let mut results: Vec<(ObjectId, f32)> = if refine {
            candidates
                .iter()
                .map(|&id| {
                    let d = nodes
                        .get(&id)
                        .map(|n| match &n.rerank {
                            // Retained int8 vector — exact-ish rescore in the
                            // original space (recovers binary's lost recall).
                            Some(rv) => rv.dist_sq(orig_query),
                            // Else the asymmetric estimator (binary) or full-dim
                            // distance (coarse), both against `query`.
                            None => n.vector.dist_sq(query),
                        })
                        .unwrap_or(f32::MAX);
                    (id, d)
                })
                .collect()
        } else {
            candidates.into_iter().zip(distances).collect()
        };

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

    /// Serialize the index to a byte buffer.
    pub fn to_bytes(&self) -> std::io::Result<Vec<u8>> {
        let nodes = self.nodes.read();
        let snapshot = HnswSnapshotRef {
            params: &self.params,
            nodes: &nodes,
            entry_point: *self.entry_point.read(),
            max_layer: *self.max_layer.read(),
            rotation: self.rotation.read().clone(),
        };
        bincode::serialize(&snapshot).map_err(std::io::Error::other)
    }

    /// Reconstruct an index from a buffer produced by [`Self::to_bytes`].
    pub fn from_bytes(bytes: &[u8]) -> std::io::Result<Self> {
        let snapshot: HnswSnapshot = bincode::deserialize(bytes).map_err(std::io::Error::other)?;
        Ok(Self {
            nodes: Arc::new(RwLock::new(snapshot.nodes)),
            entry_point: Arc::new(RwLock::new(snapshot.entry_point)),
            max_layer: Arc::new(RwLock::new(snapshot.max_layer)),
            rotation: Arc::new(RwLock::new(snapshot.rotation)),
            params: snapshot.params,
        })
    }

    /// Persist the index to `path` (atomic write via a temp file + rename).
    pub fn save(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let path = path.as_ref();
        let bytes = self.to_bytes()?;
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
        Ok(Some(Self::from_bytes(&bytes)?))
    }

    // ---- Internal methods ----

    /// Generate a random layer using exponential decay.
    fn random_layer(&self) -> usize {
        let mut rng = rand::thread_rng();
        let ml: f64 = 1.0 / (self.params.m as f64).ln();
        let r: f64 = rng.gen();
        ((-r.ln() * ml).floor() as usize).min(10) // Cap at layer 10
    }

    /// Greedy 1-nearest-neighbor search on a single layer. Returns the single
    /// nearest node id (allocation-free — called once per layer during descent).
    fn search_layer_greedy(
        &self,
        scorer: &Scorer,
        entry: ObjectId,
        layer: usize,
        nodes: &NodeMap,
    ) -> ObjectId {
        let mut best_id = entry;
        let mut best_dist = scorer.score(&nodes[&best_id].vector);

        loop {
            let mut improved = false;
            // Iterate the current node's neighbors in place (no per-step alloc).
            if let Some(neighbor_set) = nodes.get(&best_id).and_then(|n| n.neighbors.get(&layer)) {
                for &neighbor_id in neighbor_set {
                    if let Some(neighbor) = nodes.get(&neighbor_id) {
                        let dist = scorer.score(&neighbor.vector);
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

        best_id
    }

    /// Beam search on a single layer. Uses per-thread reusable scratch
    /// ([`BEAM_SCRATCH`]) so the hot path allocates only the returned vectors.
    fn search_layer_beam(
        &self,
        scorer: &Scorer,
        entry_points: &[ObjectId],
        layer: usize,
        ef: usize,
        nodes: &NodeMap,
    ) -> (Vec<ObjectId>, Vec<f32>) {
        BEAM_SCRATCH.with(|scratch| {
            let s = &mut *scratch.borrow_mut();
            // `clear` keeps the backing capacity from prior queries → no realloc.
            s.visited.clear();
            s.candidates.clear();
            s.results.clear();

            for &ep in entry_points {
                let dist = scorer.score(&nodes[&ep].vector);
                s.candidates.push(Candidate {
                    id: ep,
                    dist: OrdFloat(dist),
                });
                s.results.push(WorstCandidate {
                    id: ep,
                    dist: OrdFloat(dist),
                });
                s.visited.insert(ep);
            }

            while let Some(current) = s.candidates.pop() {
                let current_dist = current.dist.0;

                // Stop if current is farther than the worst result we're keeping.
                if s.results.len() >= ef {
                    if let Some(worst) = s.results.peek() {
                        if current_dist >= worst.dist.0 {
                            break;
                        }
                    }
                }

                // Expand neighbors in place; `visited.insert` returns false when
                // the id was already present (combines the contains+insert check).
                if let Some(neighbor_set) =
                    nodes.get(&current.id).and_then(|n| n.neighbors.get(&layer))
                {
                    for &neighbor_id in neighbor_set {
                        if !s.visited.insert(neighbor_id) {
                            continue;
                        }
                        if let Some(neighbor) = nodes.get(&neighbor_id) {
                            let dist = scorer.score(&neighbor.vector);
                            let od = OrdFloat(dist);

                            let should_add = s.results.len() < ef
                                || dist < s.results.peek().map(|c| c.dist.0).unwrap_or(f32::MAX);

                            if should_add {
                                s.candidates.push(Candidate {
                                    id: neighbor_id,
                                    dist: od,
                                });
                                s.results.push(WorstCandidate {
                                    id: neighbor_id,
                                    dist: od,
                                });
                                if s.results.len() > ef {
                                    s.results.pop();
                                }
                            }
                        }
                    }
                }
            }

            // Pop the worst-first heap (so it empties for reuse), then reverse
            // to closest-first — preserving the original ordering contract.
            let n = s.results.len();
            let mut ids = Vec::with_capacity(n);
            let mut dists = Vec::with_capacity(n);
            while let Some(c) = s.results.pop() {
                ids.push(c.id);
                dists.push(c.dist.0);
            }
            ids.reverse();
            dists.reverse();
            (ids, dists)
        })
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
                let dist = nodes[&id].vector.dist_sq(query);
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
/// so the `sqrt` is dropped — it is applied only to the final k results when
/// converting to a similarity score.
///
/// Summed over **8 independent accumulators** rather than one: a single `.sum()`
/// is latency-bound (each add waits on the previous on the FP pipeline), whereas
/// 8 lanes break the dependency chain so the CPU pipelines them and the loop
/// auto-vectorizes cleanly to NEON/SSE. `chunks_exact(8)` keeps the hot loop
/// bounds-check-free.
#[inline]
fn squared_dist(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; 8];
    let mut ai = a.chunks_exact(8);
    let mut bi = b.chunks_exact(8);
    for (ca, cb) in ai.by_ref().zip(bi.by_ref()) {
        for j in 0..8 {
            let d = ca[j] - cb[j];
            acc[j] += d * d;
        }
    }
    let mut sum = acc.iter().sum::<f32>();
    for (x, y) in ai.remainder().iter().zip(bi.remainder()) {
        let d = x - y;
        sum += d * d;
    }
    sum
}

/// Squared distance between a full-precision query and an int8-quantized vector,
/// dequantizing on the fly. Same 8-accumulator structure as [`squared_dist`].
#[inline]
fn int8_dist_sq(query: &[f32], q: &[i8]) -> f32 {
    let mut acc = [0.0f32; 8];
    let mut qi = query.chunks_exact(8);
    let mut ci = q.chunks_exact(8);
    for (cq, cc) in qi.by_ref().zip(ci.by_ref()) {
        for j in 0..8 {
            let d = cq[j] - cc[j] as f32 * INV_QUANT_SCALE;
            acc[j] += d * d;
        }
    }
    let mut sum = acc.iter().sum::<f32>();
    for (x, &c) in qi.remainder().iter().zip(ci.remainder()) {
        let d = x - c as f32 * INV_QUANT_SCALE;
        sum += d * d;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantized_index_search() {
        // int8-quantized index: exact matches stay top-1 despite quantization.
        let idx = HnswIndex::new(HnswParams {
            m: 8,
            ef_construction: 50,
            ef_search: 50,
            quantize: true,
            ..Default::default()
        });
        let mut items = Vec::new();
        for i in 0..100 {
            let id = uuid::Uuid::new_v4();
            // Components in [-1, 1], matching the quantization assumption.
            let v: Vec<f32> = (0..16).map(|j| ((i * 7 + j * 3) as f32).sin()).collect();
            idx.insert(id, v.clone());
            items.push((id, v));
        }

        let (qid, qv) = &items[42];
        let results = idx.search(qv, 5);
        assert_eq!(results.len(), 5);
        assert_eq!(
            results[0].0, *qid,
            "exact match should remain top-1 under int8 quantization"
        );
    }

    #[test]
    fn test_binary_quantized_search_recall() {
        // RaBitQ-style 1-bit index: exact matches stay near the top and
        // recall@10 vs brute force is reasonable despite ~32× compression.
        let idx = HnswIndex::new(HnswParams {
            m: 16,
            ef_construction: 100,
            ef_search: 100,
            binary_quantize: true,
            ..Default::default()
        });
        let mut items = Vec::new();
        for i in 0..200u32 {
            let id = uuid::Uuid::from_u128(i as u128 + 1);
            let v: Vec<f32> = (0..64)
                .map(|j| (((i * 13 + j * 7) as f32) * 0.1).sin())
                .collect();
            idx.insert(id, v.clone());
            items.push((id, v));
        }

        let (qid, qv) = &items[42];
        let res = idx.search(qv, 5);
        assert!(
            res.iter().take(3).any(|(id, _)| id == qid),
            "exact match should be near top-1 under binary quantization"
        );

        // Recall@10 vs brute-force ground truth over a few probes.
        let probes = [7usize, 99, 150];
        let (mut hit, mut total) = (0usize, 0usize);
        for &p in &probes {
            let q = &items[p].1;
            let mut bf: Vec<_> = items
                .iter()
                .map(|(id, v)| (*id, squared_dist(q, v)))
                .collect();
            bf.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
            let truth: HashSet<_> = bf.iter().take(10).map(|(id, _)| *id).collect();
            for (id, _) in idx.search(q, 10) {
                if truth.contains(&id) {
                    hit += 1;
                }
                total += 1;
            }
        }
        let recall = hit as f32 / total as f32;
        assert!(recall >= 0.5, "binary recall@10 too low: {recall}");
    }

    #[test]
    fn test_matryoshka_coarse_search_refines() {
        // Coarse (prefix-dim) navigation + full-dim refine: exact matches stay
        // top-1 and recall@10 vs brute force stays high because the final
        // ranking is computed at full dimension.
        let idx = HnswIndex::new(HnswParams {
            m: 16,
            ef_construction: 100,
            ef_search: 100,
            coarse_dim: Some(16), // navigate on the first 16 of 64 dims
            ..Default::default()
        });
        let mut items = Vec::new();
        for i in 0..200u32 {
            let id = uuid::Uuid::from_u128(i as u128 + 1);
            let v: Vec<f32> = (0..64)
                .map(|j| (((i * 13 + j * 7) as f32) * 0.1).sin())
                .collect();
            idx.insert(id, v.clone());
            items.push((id, v));
        }

        let (qid, qv) = &items[42];
        assert_eq!(
            idx.search(qv, 5)[0].0,
            *qid,
            "exact match should be top-1 after full-dim refine"
        );

        let probes = [7usize, 99, 150];
        let (mut hit, mut total) = (0usize, 0usize);
        for &p in &probes {
            let q = &items[p].1;
            let mut bf: Vec<_> = items
                .iter()
                .map(|(id, v)| (*id, squared_dist(q, v)))
                .collect();
            bf.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
            let truth: HashSet<_> = bf.iter().take(10).map(|(id, _)| *id).collect();
            for (id, _) in idx.search(q, 10) {
                if truth.contains(&id) {
                    hit += 1;
                }
                total += 1;
            }
        }
        let recall = hit as f32 / total as f32;
        assert!(recall >= 0.7, "matryoshka recall@10 too low: {recall}");
    }

    #[test]
    fn test_binary_rerank_improves_recall() {
        // Oversample→rescore with retained int8 vectors should recover recall
        // the 1-bit codes lose: rerank recall ≥ plain-binary recall, and high.
        let build = |rerank: bool| {
            let idx = HnswIndex::new(HnswParams {
                m: 16,
                ef_construction: 100,
                ef_search: 100,
                binary_quantize: true,
                binary_rerank: rerank,
                ..Default::default()
            });
            let mut items = Vec::new();
            for i in 0..200u32 {
                let id = uuid::Uuid::from_u128(i as u128 + 1);
                let v: Vec<f32> = (0..64)
                    .map(|j| (((i * 13 + j * 7) as f32) * 0.1).sin())
                    .collect();
                idx.insert(id, v.clone());
                items.push((id, v));
            }
            (idx, items)
        };

        let recall_of = |idx: &HnswIndex, items: &[(uuid::Uuid, Vec<f32>)]| -> f32 {
            let probes = [7usize, 42, 99, 150, 175];
            let (mut hit, mut total) = (0usize, 0usize);
            for &p in &probes {
                let q = &items[p].1;
                let mut bf: Vec<_> = items
                    .iter()
                    .map(|(id, v)| (*id, squared_dist(q, v)))
                    .collect();
                bf.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
                let truth: HashSet<_> = bf.iter().take(10).map(|(id, _)| *id).collect();
                for (id, _) in idx.search(q, 10) {
                    if truth.contains(&id) {
                        hit += 1;
                    }
                    total += 1;
                }
            }
            hit as f32 / total as f32
        };

        let (plain, items) = build(false);
        let (reranked, items2) = build(true);
        let plain_recall = recall_of(&plain, &items);
        let rerank_recall = recall_of(&reranked, &items2);

        assert!(
            rerank_recall >= plain_recall,
            "rerank ({rerank_recall}) should not hurt recall vs plain binary ({plain_recall})"
        );
        assert!(
            rerank_recall >= 0.85,
            "binary+int8-rerank recall too low: {rerank_recall}"
        );
    }

    #[test]
    fn test_binary_quantized_save_load() {
        let idx = HnswIndex::new(HnswParams {
            m: 8,
            ef_construction: 50,
            ef_search: 50,
            binary_quantize: true,
            ..Default::default()
        });
        let mut items = Vec::new();
        for i in 0..60u32 {
            let id = uuid::Uuid::from_u128(i as u128 + 1);
            let v: Vec<f32> = (0..32)
                .map(|j| (((i * 11 + j * 5) as f32) * 0.1).sin())
                .collect();
            idx.insert(id, v.clone());
            items.push((id, v));
        }

        let path = std::env::temp_dir().join(format!("kowitodb-bq-{}.bin", uuid::Uuid::new_v4()));
        idx.save(&path).unwrap();
        let loaded = HnswIndex::load(&path).unwrap().expect("snapshot loads");

        // The rotation must survive the round-trip, so results match exactly.
        let q = &items[20].1;
        let before: Vec<_> = idx.search(q, 5).into_iter().map(|(id, _)| id).collect();
        let after: Vec<_> = loaded.search(q, 5).into_iter().map(|(id, _)| id).collect();
        assert_eq!(before, after, "binary search must survive save/load");
        let _ = std::fs::remove_file(&path);
    }

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
