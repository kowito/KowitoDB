//! `HnswIndex` — graph construction (insert / remove / neighbor selection) (split from the former hnsw.rs god-file).

use super::*;

impl HnswIndex {
    /// Insert a vector for an object.
    ///
    /// If the object already exists, it is re-inserted (updated).
    pub fn insert(&self, id: ObjectId, vector: Embedding) {
        // Enforce a single vector dimension. The first insert sets it; a later
        // mismatch is skipped (with a warning) rather than silently corrupting
        // the index — distance over differing lengths returns wrong results.
        {
            let mut dim = self.dim.write();
            match *dim {
                None => *dim = Some(vector.len()),
                Some(d) if d != vector.len() => {
                    debug!(
                        "HNSW: skipping insert of {}-dim vector into a {}-dim index ({})",
                        vector.len(),
                        d,
                        id
                    );
                    return;
                }
                Some(_) => {}
            }
        }

        let mut graph = self.graph.write();
        let mut entry_point = self.entry_point.write();
        let mut max_layer = self.max_layer.write();

        // Re-insert (update) = remove the old copy first, so neighbor search
        // can't find the node itself and indices stay dense.
        if graph.id_to_idx.contains_key(&id) {
            Self::remove_locked(&mut graph, &mut entry_point, &mut max_layer, id);
        }

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

        // The new node lands at the end of the dense node vector. It is pushed
        // *now*, with empty neighbor lists: with no inbound edges yet it is
        // unreachable during this insert's own searches, but its index is valid
        // so neighbor pruning (which may reference it) never goes out of bounds.
        let idx = graph.nodes.len() as NodeIdx;
        graph.nodes.push(HnswNode {
            id,
            vector: node_vector,
            max_layer: node_layer,
            neighbors: vec![Vec::new(); node_layer + 1],
            rerank,
        });
        graph.id_to_idx.insert(id, idx);

        // If this is the first node, it becomes the entry point.
        let ep = match *entry_point {
            Some(ep) => ep,
            None => {
                *max_layer = node_layer;
                *entry_point = Some(idx);
                debug!("HNSW: inserted first node {} at layer {}", id, node_layer);
                return;
            }
        };

        let mut curr_ep = ep;
        let global_max = *max_layer;

        // Greedy descent from top layer to node_layer + 1
        for lc in ((node_layer + 1)..=global_max).rev() {
            curr_ep = self.search_layer_greedy(&scorer, curr_ep, lc, &graph.nodes);
        }

        // Insert at each layer from min(node_layer, global_max) down to 0
        let start_layer = node_layer.min(global_max);
        let mut ep_set = vec![curr_ep];

        for lc in (0..=start_layer).rev() {
            let (candidates, _) = self.search_layer_beam(
                &scorer,
                &ep_set,
                lc,
                self.params.ef_construction,
                &graph.nodes,
            );

            let max_deg = if lc == 0 {
                self.params.m0
            } else {
                self.params.m
            };
            let selected =
                self.select_neighbors_heuristic(&scorer, &candidates, max_deg, &graph.nodes);

            // Add bidirectional edges (the immutable search borrow has ended).
            for &nb in &selected {
                if nb == idx {
                    continue; // never link to self
                }
                graph.nodes[idx as usize].neighbors[lc].push(nb);
                let nbu = nb as usize;
                if graph.nodes[nbu].neighbors.len() <= lc {
                    graph.nodes[nbu].neighbors.resize(lc + 1, Vec::new());
                }
                if graph.nodes[nbu].neighbors[lc].contains(&idx) {
                    continue;
                }
                graph.nodes[nbu].neighbors[lc].push(idx);
                // Standard-HNSW neighbor pruning (bound `nb`'s degree to
                // `max_deg`) — only in the `diversify_neighbors` "standard HNSW"
                // mode. The default deliberately leaves degrees unbounded: on
                // uniform/high-dim data the denser graph gives higher recall at a
                // given ef (measured), and KowitoDB favors recall there.
                if self.params.diversify_neighbors && graph.nodes[nbu].neighbors[lc].len() > max_deg
                {
                    let cands = graph.nodes[nbu].neighbors[lc].clone();
                    let kept = self.prune_neighbors(nb, &cands, max_deg, &graph.nodes, &scorer);
                    graph.nodes[nbu].neighbors[lc] = kept;
                }
            }

            ep_set = selected;
        }

        // Update entry point if this node is at a higher layer.
        if node_layer > global_max {
            *entry_point = Some(idx);
            *max_layer = node_layer;
        }
        debug!(
            "HNSW: inserted node {} at layer {} (global_max={})",
            id, node_layer, *max_layer
        );
    }

    /// Remove a node from the index.
    pub fn remove(&self, id: ObjectId) {
        let mut graph = self.graph.write();
        let mut entry_point = self.entry_point.write();
        let mut max_layer = self.max_layer.write();
        Self::remove_locked(&mut graph, &mut entry_point, &mut max_layer, id);
    }

    /// Remove `id` from an already-locked graph. Uses `swap_remove` to keep the
    /// node vector dense, then fixes every edge: references to the removed slot
    /// are dropped and references to the moved (formerly-last) node are remapped.
    /// O(N) in the node count, but removals are rare relative to queries.
    pub(crate) fn remove_locked(
        graph: &mut Graph,
        entry_point: &mut Option<NodeIdx>,
        max_layer: &mut usize,
        id: ObjectId,
    ) {
        let Some(r) = graph.id_to_idx.remove(&id) else {
            return;
        };
        let last = (graph.nodes.len() - 1) as NodeIdx;
        graph.nodes.swap_remove(r as usize);
        // If a node was moved into slot `r`, repoint its id → index mapping.
        if r != last {
            let moved_id = graph.nodes[r as usize].id;
            graph.id_to_idx.insert(moved_id, r);
        }
        // Rewrite all adjacency: drop edges to `r` (gone), remap `last` → `r`.
        for node in graph.nodes.iter_mut() {
            for layer in node.neighbors.iter_mut() {
                let mut w = 0;
                for read in 0..layer.len() {
                    let v = layer[read];
                    if v == r {
                        continue;
                    }
                    layer[w] = if v == last { r } else { v };
                    w += 1;
                }
                layer.truncate(w);
            }
        }
        // Restore the HNSW invariant: the entry point must be a top-layer node,
        // and `max_layer` must match. Removing the old entry point (e.g. on every
        // re-insert of the current top node) otherwise silently collapses recall —
        // descent would start from an arbitrary low-layer node. Recompute both
        // from the remaining nodes (O(N), but removals are rare vs queries).
        if graph.nodes.is_empty() {
            *entry_point = None;
            *max_layer = 0;
        } else {
            let (top_idx, top_layer) = graph
                .nodes
                .iter()
                .enumerate()
                .map(|(i, n)| (i as NodeIdx, n.max_layer))
                .max_by_key(|&(_, l)| l)
                .unwrap();
            *entry_point = Some(top_idx);
            *max_layer = top_layer;
        }
    }

    /// Generate a random layer using exponential decay.
    pub(crate) fn random_layer(&self) -> usize {
        // `m == 1` would make `ln(m) == 0` → division by zero (every node at the
        // cap layer, a degenerate graph); clamp the level multiplier's base to ≥2.
        let mut rng = rand::thread_rng();
        let ml: f64 = 1.0 / (self.params.m.max(2) as f64).ln();
        let r: f64 = rng.gen();
        ((-r.ln() * ml).floor() as usize).min(10) // Cap at layer 10
    }

    /// HNSW neighbor-selection **diversity heuristic** (Malkov & Yashunin,
    /// Algorithm 4, with kept-pruned connections). Rather than just keeping the
    /// `m` closest candidates — which clusters all edges in one direction and
    /// hurts navigability — a candidate `e` is accepted only if it is closer to
    /// the query than to every already-selected neighbor. This spreads edges
    /// across directions, materially improving graph quality and recall@ef.
    /// If fewer than `m` survive, the best pruned candidates backfill the rest.
    pub(crate) fn select_neighbors_heuristic(
        &self,
        scorer: &Scorer,
        candidates: &[NodeIdx],
        m: usize,
        nodes: &[HnswNode],
    ) -> Vec<NodeIdx> {
        if candidates.len() <= m {
            return candidates.to_vec();
        }
        // Candidates sorted by distance to the query (closest first).
        let mut sorted: Vec<(NodeIdx, f32)> = candidates
            .iter()
            .map(|&idx| (idx, scorer.score(&nodes[idx as usize].vector)))
            .collect();
        sorted.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        // Default: keep the `m` closest (fast). The diversity heuristic below is
        // opt-in via `diversify_neighbors`.
        if !self.params.diversify_neighbors {
            return sorted.into_iter().take(m).map(|(idx, _)| idx).collect();
        }

        let mut selected: Vec<NodeIdx> = Vec::with_capacity(m);
        let mut pruned: Vec<NodeIdx> = Vec::new();
        for (e, dist_eq) in sorted {
            if selected.len() >= m {
                break;
            }
            // Accept `e` only if it is closer to the query than to any neighbor
            // already chosen (keeps the selected set diverse in direction).
            let diverse = selected.iter().all(|&r| {
                scorer.node_dist(&nodes[e as usize].vector, &nodes[r as usize].vector) >= dist_eq
            });
            if diverse {
                selected.push(e);
            } else {
                pruned.push(e);
            }
        }
        // Backfill from the closest pruned candidates if we came up short.
        for e in pruned {
            if selected.len() >= m {
                break;
            }
            selected.push(e);
        }
        selected
    }

    /// Prune `center`'s over-full neighbor list back to the best `m`, centered on
    /// `center` itself (node-to-node distances). Mirrors `select_neighbors_*`:
    /// plain closest-`m` by default, the diversity heuristic when enabled.
    pub(crate) fn prune_neighbors(
        &self,
        center: NodeIdx,
        candidates: &[NodeIdx],
        m: usize,
        nodes: &[HnswNode],
        scorer: &Scorer,
    ) -> Vec<NodeIdx> {
        let center_vec = &nodes[center as usize].vector;
        let mut sorted: Vec<(NodeIdx, f32)> = candidates
            .iter()
            .map(|&c| (c, scorer.node_dist(center_vec, &nodes[c as usize].vector)))
            .collect();
        sorted.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        if !self.params.diversify_neighbors {
            return sorted.into_iter().take(m).map(|(c, _)| c).collect();
        }
        let mut selected: Vec<NodeIdx> = Vec::with_capacity(m);
        let mut pruned: Vec<NodeIdx> = Vec::new();
        for (c, dist_c) in sorted {
            if selected.len() >= m {
                break;
            }
            let diverse = selected.iter().all(|&r| {
                scorer.node_dist(&nodes[c as usize].vector, &nodes[r as usize].vector) >= dist_c
            });
            if diverse {
                selected.push(c);
            } else {
                pruned.push(c);
            }
        }
        for c in pruned {
            if selected.len() >= m {
                break;
            }
            selected.push(c);
        }
        selected
    }
}
