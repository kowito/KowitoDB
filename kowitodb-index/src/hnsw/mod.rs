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
use std::collections::{BinaryHeap, HashMap};
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
///
/// Dense node index into [`Graph::nodes`]. Graph traversal works in these
/// indices — plain array access, no `ObjectId` hashing on the hot path.
type NodeIdx = u32;

/// Contiguous node storage plus an `ObjectId → index` map. Traversal reads
/// `nodes` by index; the map is only consulted at insert/remove/search entry
/// and when mapping results back to ids.
#[derive(Default)]
pub(crate) struct Graph {
    nodes: Vec<HnswNode>,
    id_to_idx: HashMap<ObjectId, NodeIdx, ahash::RandomState>,
}

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

    /// Pointer to the start of the vector's backing data, for prefetch hints.
    #[inline]
    fn data_ptr(&self) -> *const u8 {
        match self {
            NodeVector::Full(v) => v.as_ptr() as *const u8,
            NodeVector::Quantized(v) => v.as_ptr() as *const u8,
            NodeVector::Binary { code, .. } => code.as_ptr() as *const u8,
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

    /// Distance to another stored vector of the same variant, in the same units
    /// as the float scorer (`dist_sq`): squared Euclidean for `Full`, dequantized
    /// squared Euclidean for `Quantized`. Used by the HNSW diversity heuristic.
    #[inline]
    fn dist_to(&self, other: &NodeVector) -> f32 {
        match (self, other) {
            (NodeVector::Full(a), NodeVector::Full(b)) => squared_dist(a, b),
            (NodeVector::Quantized(a), NodeVector::Quantized(b)) => a
                .iter()
                .zip(b)
                .map(|(&x, &y)| {
                    let d = (x as f32 - y as f32) * INV_QUANT_SCALE;
                    d * d
                })
                .sum(),
            // Mismatched variants never occur within one index.
            _ => f32::MAX,
        }
    }

    /// Hamming distance to another binary code — the node-to-node distance in
    /// the same units as the Hamming scorer (binary navigation).
    #[inline]
    fn hamming_to(&self, other: &NodeVector) -> f32 {
        match (self, other) {
            (NodeVector::Binary { code: a, .. }, NodeVector::Binary { code: b, .. }) => {
                a.iter()
                    .zip(b)
                    .map(|(x, y)| (x ^ y).count_ones())
                    .sum::<u32>() as f32
            }
            _ => f32::MAX,
        }
    }
}

/// Best-effort hint to prefetch the cache line at `ptr` into L1 for reading.
/// A no-op on architectures without a stable prefetch path. `prfm`/`_mm_prefetch`
/// are pure hints, so any address is safe.
#[inline(always)]
fn prefetch_read(ptr: *const u8) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: `_mm_prefetch` is a hint; any pointer value is valid.
    unsafe {
        core::arch::x86_64::_mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T0 }>(ptr as *const i8);
    }
    #[cfg(target_arch = "aarch64")]
    // SAFETY: `prfm` is a hint instruction with no memory effects.
    unsafe {
        core::arch::asm!(
            "prfm pldl1keep, [{p}]",
            p = in(reg) ptr,
            options(nostack, preserves_flags),
        );
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = ptr;
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
pub(crate) enum Scorer<'a> {
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

    /// Distance between two stored nodes, in the same units as `score`, for the
    /// HNSW diversity heuristic (so `node_dist(e, r)` and `score(e)` compare).
    #[inline]
    fn node_dist(&self, a: &NodeVector, b: &NodeVector) -> f32 {
        match self {
            Scorer::Float { .. } => a.dist_to(b),
            Scorer::Hamming { .. } => a.hamming_to(b),
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
    id: NodeIdx,
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
    id: NodeIdx,
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
///
/// `visited` is a **generation-stamped** array indexed by node id: a node is
/// "visited this query" iff `visited[idx] == gen`. Bumping `gen` per query makes
/// reset O(1) (no clearing, no hashing) — the standard fast-HNSW visited set.
#[derive(Default)]
struct BeamScratch {
    visited: Vec<u32>,
    gen: u32,
    candidates: BinaryHeap<Candidate>,
    results: BinaryHeap<WorstCandidate>,
}

impl BeamScratch {
    /// Start a new query over `node_count` nodes, returning the active
    /// generation. Resizes the visited array and bumps the generation (clearing
    /// on wrap so stale stamps never alias).
    fn begin(&mut self, node_count: usize) -> u32 {
        if self.visited.len() < node_count {
            self.visited.resize(node_count, 0);
        }
        self.gen = self.gen.wrapping_add(1);
        if self.gen == 0 {
            for v in self.visited.iter_mut() {
                *v = 0;
            }
            self.gen = 1;
        }
        self.candidates.clear();
        self.results.clear();
        self.gen
    }

    /// Mark `idx` visited for the current generation; returns `true` if it was
    /// not already visited (combines the contains+insert check).
    #[inline]
    fn visit(&mut self, idx: NodeIdx) -> bool {
        let slot = &mut self.visited[idx as usize];
        if *slot == self.gen {
            false
        } else {
            *slot = self.gen;
            true
        }
    }
}

thread_local! {
    static BEAM_SCRATCH: std::cell::RefCell<BeamScratch> =
        std::cell::RefCell::new(BeamScratch::default());
}

/// A node in the HNSW graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct HnswNode {
    id: ObjectId,
    vector: NodeVector,
    max_layer: usize,
    /// Per-layer neighbor lists: `neighbors[layer]` holds neighbor node indices.
    /// `Vec` (not a set) for cache-friendly iteration; dedup is enforced on
    /// insert. Indexed by layer (0..=max_layer).
    neighbors: Vec<Vec<NodeIdx>>,
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
    /// Build with the **full standard-HNSW recipe** (Malkov & Yashunin): the
    /// neighbor-selection **diversity heuristic** (Alg. 4) *and* **degree
    /// pruning** of over-full neighbor lists. On *clustered* (real-embedding)
    /// data this is a Pareto win at low `ef_search` — e.g. recall 0.92 → 0.95 at
    /// the same QPS — because pruning bounds degree (keeping queries fast) while
    /// diversity keeps the graph navigable. On uniform/high-dim data the default
    /// (no pruning, unbounded degree) gives higher recall, so this is **off by
    /// default**; enable for real embeddings, especially when targeting low `ef`.
    #[serde(default)]
    pub diversify_neighbors: bool,
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
            diversify_neighbors: false,
        }
    }
}

/// HNSW vector index.
///
/// Thread-safe via RwLock. Supports concurrent reads and serialized writes.
pub struct HnswIndex {
    /// All nodes, stored contiguously with an id→index map.
    graph: Arc<RwLock<Graph>>,
    /// Entry point (top-layer node), as a node index.
    entry_point: Arc<RwLock<Option<NodeIdx>>>,
    /// Current maximum layer across all nodes.
    max_layer: Arc<RwLock<usize>>,
    /// Structured random rotation for binary quantization (lazily created on
    /// the first insert once the dimensionality is known). `None` unless
    /// `params.binary_quantize` is set.
    rotation: Arc<RwLock<Option<Rotation>>>,
    /// Established vector dimension (set on the first insert). Mismatched inserts
    /// are skipped and mismatched queries return empty — otherwise the distance
    /// loops silently truncate to the shorter vector and return wrong results.
    dim: Arc<RwLock<Option<usize>>>,
    /// Configuration.
    params: HnswParams,
}

/// Borrowed view of the index for zero-copy serialization on `save`.
#[derive(Serialize)]
struct HnswSnapshotRef<'a> {
    params: &'a HnswParams,
    nodes: &'a [HnswNode],
    entry_point: Option<NodeIdx>,
    max_layer: usize,
    #[serde(default)]
    rotation: Option<Rotation>,
    #[serde(default)]
    dim: Option<usize>,
}

/// Owned snapshot for deserialization on `load`.
#[derive(Deserialize)]
struct HnswSnapshot {
    params: HnswParams,
    nodes: Vec<HnswNode>,
    entry_point: Option<NodeIdx>,
    max_layer: usize,
    #[serde(default)]
    rotation: Option<Rotation>,
    #[serde(default)]
    dim: Option<usize>,
}

mod build;
mod persist;
mod search;
#[cfg(test)]
mod tests;

impl HnswIndex {
    pub fn new(params: HnswParams) -> Self {
        Self {
            graph: Arc::new(RwLock::new(Graph::default())),
            entry_point: Arc::new(RwLock::new(None)),
            max_layer: Arc::new(RwLock::new(0)),
            rotation: Arc::new(RwLock::new(None)),
            dim: Arc::new(RwLock::new(None)),
            params,
        }
    }

    /// The established vector dimension (set by the first insert), or `None` if
    /// the index is empty.
    pub fn dimension(&self) -> Option<usize> {
        *self.dim.read()
    }

    /// Number of nodes in the index.
    pub fn len(&self) -> usize {
        self.graph.read().nodes.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.graph.read().nodes.is_empty()
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
