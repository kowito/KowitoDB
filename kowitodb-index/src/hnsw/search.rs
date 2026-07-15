//! `HnswIndex` — k-NN query (greedy + beam layer search) (split from the former hnsw.rs god-file).

use super::*;

impl HnswIndex {
    /// Search for the k-nearest neighbors.
    pub fn search(&self, query: &Embedding, k: usize) -> Vec<(ObjectId, f32)> {
        // A wrong-dimension query would silently produce garbage distances.
        if let Some(d) = *self.dim.read() {
            if query.len() != d {
                debug!(
                    "HNSW: {}-dim query against a {}-dim index — returning no results",
                    query.len(),
                    d
                );
                return Vec::new();
            }
        }

        let graph = self.graph.read();
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
            curr_ep = self.search_layer_greedy(&scorer, curr_ep, lc, &graph.nodes);
        }

        // Beam search at layer 0. When refining, over-fetch candidates so the
        // re-score below has a good pool to re-rank.
        let ef = if refine {
            self.params.ef_search.max(k * 4)
        } else {
            self.params.ef_search.max(k)
        };
        let (candidates, distances) =
            self.search_layer_beam(&scorer, &[curr_ep], 0, ef, &graph.nodes);

        // Take top-k (still working in node indices). When navigation was
        // approximate, re-score candidates with the accurate distance (full-dim
        // for coarse, asymmetric RaBitQ estimator for binary); otherwise use the
        // beam's distances directly.
        let mut results: Vec<(NodeIdx, f32)> = if refine {
            candidates
                .iter()
                .map(|&idx| {
                    let n = &graph.nodes[idx as usize];
                    let d = match &n.rerank {
                        // Retained int8 vector — exact-ish rescore in the
                        // original space (recovers binary's lost recall).
                        Some(rv) => rv.dist_sq(orig_query),
                        // Else the asymmetric estimator (binary) or full-dim
                        // distance (coarse), both against `query`.
                        None => n.vector.dist_sq(query),
                    };
                    (idx, d)
                })
                .collect()
        } else {
            candidates.into_iter().zip(distances).collect()
        };

        results.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        results.truncate(k);

        // Map indices back to object ids and convert squared distance to a
        // similarity (1 / (1 + √distance)); sqrt only on the k returned results.
        results
            .into_iter()
            .map(|(idx, dist)| (graph.nodes[idx as usize].id, 1.0 / (1.0 + dist.sqrt())))
            .collect()
    }

    /// Greedy 1-nearest-neighbor search on a single layer. Returns the single
    /// nearest node index (allocation-free — called once per layer in descent).
    pub(crate) fn search_layer_greedy(
        &self,
        scorer: &Scorer,
        entry: NodeIdx,
        layer: usize,
        nodes: &[HnswNode],
    ) -> NodeIdx {
        let mut best = entry;
        let mut best_dist = scorer.score(&nodes[best as usize].vector);

        loop {
            let mut improved = false;
            if let Some(neighbors) = nodes[best as usize].neighbors.get(layer) {
                for &nb in neighbors {
                    let dist = scorer.score(&nodes[nb as usize].vector);
                    if dist < best_dist {
                        best_dist = dist;
                        best = nb;
                        improved = true;
                    }
                }
            }
            if !improved {
                break;
            }
        }
        best
    }

    /// Beam search on a single layer. Uses per-thread reusable scratch
    /// ([`BEAM_SCRATCH`]) — the visited set is a generation-stamped array indexed
    /// by node id, so the hot path does no hashing and allocates only the
    /// returned vectors.
    pub(crate) fn search_layer_beam(
        &self,
        scorer: &Scorer,
        entry_points: &[NodeIdx],
        layer: usize,
        ef: usize,
        nodes: &[HnswNode],
    ) -> (Vec<NodeIdx>, Vec<f32>) {
        BEAM_SCRATCH.with(|scratch| {
            let s = &mut *scratch.borrow_mut();
            s.begin(nodes.len());

            for &ep in entry_points {
                let dist = scorer.score(&nodes[ep as usize].vector);
                s.candidates.push(Candidate {
                    id: ep,
                    dist: OrdFloat(dist),
                });
                s.results.push(WorstCandidate {
                    id: ep,
                    dist: OrdFloat(dist),
                });
                s.visit(ep);
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

                // Expand neighbors; `visit` returns false when already seen.
                if let Some(neighbors) = nodes[current.id as usize].neighbors.get(layer) {
                    for i in 0..neighbors.len() {
                        // Prefetch the *next* neighbor's vector while we score
                        // this one — hides the cache-miss latency of the random
                        // node access that dominates the hot loop.
                        if let Some(&next) = neighbors.get(i + 1) {
                            prefetch_read(nodes[next as usize].vector.data_ptr());
                        }
                        let nb = neighbors[i];
                        if !s.visit(nb) {
                            continue;
                        }
                        let dist = scorer.score(&nodes[nb as usize].vector);
                        let od = OrdFloat(dist);
                        let should_add = s.results.len() < ef
                            || dist < s.results.peek().map(|c| c.dist.0).unwrap_or(f32::MAX);
                        if should_add {
                            s.candidates.push(Candidate { id: nb, dist: od });
                            s.results.push(WorstCandidate { id: nb, dist: od });
                            if s.results.len() > ef {
                                s.results.pop();
                            }
                        }
                    }
                }
            }

            // Pop the worst-first heap (emptying it for reuse), then reverse to
            // closest-first — preserving the original ordering contract.
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
}
