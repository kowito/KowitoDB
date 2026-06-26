use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use kowitodb_core::{ObjectId, Relationship};
use parking_lot::RwLock;
use tracing::debug;

/// In-memory knowledge graph index for relationship traversal.
///
/// Stores directed edges between knowledge objects. Supports:
/// - Finding all objects related to a given entity
/// - Graph traversal (BFS/DFS) from a starting node
/// - Reverse-lookups ("which objects reference X?")
pub struct GraphIndex {
    /// Forward adjacency list: source_id -> [(relation_type, target_id)]
    forward: Arc<RwLock<HashMap<ObjectId, Vec<Relationship>>>>,
    /// Reverse adjacency list: target_id -> [(relation_type, source_id)]
    reverse: Arc<RwLock<HashMap<ObjectId, Vec<(String, ObjectId)>>>>,
}

impl GraphIndex {
    pub fn new() -> Self {
        Self {
            forward: Arc::new(RwLock::new(HashMap::new())),
            reverse: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Insert all relationships for an object. Replaces any existing edges
    /// from this object.
    pub fn insert_relationships(&self, source_id: ObjectId, relationships: &[Relationship]) {
        let mut forward = self.forward.write();
        let mut reverse = self.reverse.write();

        // Remove old forward edges for this source
        if let Some(old_rels) = forward.remove(&source_id) {
            for rel in &old_rels {
                Self::remove_reverse_edge(&mut reverse, rel.target_id, source_id);
            }
        }

        // Insert new forward edges
        forward.insert(source_id, relationships.to_vec());

        // Insert reverse edges
        for rel in relationships {
            reverse
                .entry(rel.target_id)
                .or_default()
                .push((rel.relation_type.clone(), source_id));
        }

        debug!(
            "Graph: inserted {} relationships for object {}",
            relationships.len(),
            source_id
        );
    }

    fn remove_reverse_edge(
        reverse: &mut HashMap<ObjectId, Vec<(String, ObjectId)>>,
        target_id: ObjectId,
        source_id: ObjectId,
    ) {
        if let Some(edges) = reverse.get_mut(&target_id) {
            edges.retain(|(_, src)| *src != source_id);
            if edges.is_empty() {
                reverse.remove(&target_id);
            }
        }
    }

    /// Remove all edges involving this object (both directions).
    pub fn remove_object(&self, id: ObjectId) {
        let mut forward = self.forward.write();
        let mut reverse = self.reverse.write();

        // Remove forward edges
        if let Some(rels) = forward.remove(&id) {
            for rel in &rels {
                Self::remove_reverse_edge(&mut reverse, rel.target_id, id);
            }
        }

        // Remove reverse edges
        if let Some(incoming) = reverse.remove(&id) {
            for (_, src_id) in incoming {
                if let Some(edges) = forward.get_mut(&src_id) {
                    edges.retain(|r| r.target_id != id);
                    if edges.is_empty() {
                        forward.remove(&src_id);
                    }
                }
            }
        }

        debug!("Graph: removed all edges for object {}", id);
    }

    /// Get all objects directly related FROM this object.
    pub fn out_edges(&self, id: ObjectId) -> Vec<Relationship> {
        self.forward.read().get(&id).cloned().unwrap_or_default()
    }

    /// Get all objects that reference this object (incoming edges).
    pub fn in_edges(&self, id: ObjectId) -> Vec<(String, ObjectId)> {
        self.reverse.read().get(&id).cloned().unwrap_or_default()
    }

    /// BFS traversal starting from one or more seed nodes, following forward edges.
    ///
    /// Returns all reachable object IDs within `max_depth` hops, including
    /// the seeds themselves.
    pub fn bfs_traverse(
        &self,
        seeds: &[ObjectId],
        max_depth: usize,
        relation_filter: Option<&str>,
    ) -> Vec<(ObjectId, usize)> {
        // (object_id, depth)
        let mut results: Vec<(ObjectId, usize)> = Vec::new();
        let mut visited: HashMap<ObjectId, usize> = HashMap::new();
        let mut queue: VecDeque<(ObjectId, usize)> = VecDeque::new();

        for seed in seeds {
            visited.insert(*seed, 0);
            queue.push_back((*seed, 0));
            results.push((*seed, 0));
        }

        let forward = self.forward.read();

        while let Some((current, depth)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }

            if let Some(edges) = forward.get(&current) {
                for rel in edges {
                    // Apply relation filter if specified
                    if let Some(filter) = relation_filter {
                        if rel.relation_type != filter {
                            continue;
                        }
                    }

                    let next_depth = depth + 1;
                    // Only visit if not seen or found at a higher depth
                    if let Some(&existing) = visited.get(&rel.target_id) {
                        if next_depth < existing {
                            visited.insert(rel.target_id, next_depth);
                            // Update the result depth
                            if let Some((_, d)) =
                                results.iter_mut().find(|(id, _)| *id == rel.target_id)
                            {
                                *d = next_depth;
                            }
                            queue.push_back((rel.target_id, next_depth));
                        }
                    } else {
                        visited.insert(rel.target_id, next_depth);
                        results.push((rel.target_id, next_depth));
                        queue.push_back((rel.target_id, next_depth));
                    }
                }
            }
        }

        debug!(
            "Graph BFS: {} seeds, max_depth={}, found {} nodes",
            seeds.len(),
            max_depth,
            results.len()
        );

        results
    }

    /// DFS traversal with scoring: nodes closer to seeds get higher scores.
    ///
    /// Returns (object_id, score) pairs where score = 1.0 / (1 + depth).
    pub fn scored_traverse(
        &self,
        seeds: &[ObjectId],
        max_depth: usize,
        relation_filter: Option<&str>,
    ) -> Vec<(ObjectId, f32)> {
        let bfs_results = self.bfs_traverse(seeds, max_depth, relation_filter);
        bfs_results
            .into_iter()
            .map(|(id, depth)| {
                let score = 1.0 / (1.0 + depth as f32);
                (id, score)
            })
            .collect()
    }

    /// Bidirectional BFS: follows both forward and reverse edges.
    ///
    /// This is the right traversal for entity queries like
    /// "who invested in X?" or "what does X reference?"
    pub fn bidirectional_traverse(
        &self,
        seeds: &[ObjectId],
        max_depth: usize,
        relation_filter: Option<&str>,
    ) -> Vec<(ObjectId, usize)> {
        let forward_results = self.bfs_traverse(seeds, max_depth, relation_filter);

        // Also traverse reverse edges (incoming)
        let mut results = forward_results.clone();
        let visited: HashMap<ObjectId, usize> =
            forward_results.iter().map(|(id, d)| (*id, *d)).collect();
        let mut queue: VecDeque<(ObjectId, usize)> = forward_results
            .into_iter()
            .filter(|(_, d)| *d < max_depth)
            .collect();

        let reverse = self.reverse.read();

        while let Some((current, depth)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }

            if let Some(incoming) = reverse.get(&current) {
                for (rel_type, source_id) in incoming {
                    if let Some(filter) = relation_filter {
                        if rel_type != filter {
                            continue;
                        }
                    }

                    let next_depth = depth + 1;
                    if let Some(&existing) = visited.get(source_id) {
                        if next_depth < existing {
                            if let Some((_, d)) =
                                results.iter_mut().find(|(id, _)| *id == *source_id)
                            {
                                *d = next_depth;
                            }
                            queue.push_back((*source_id, next_depth));
                        }
                    } else {
                        results.push((*source_id, next_depth));
                        queue.push_back((*source_id, next_depth));
                    }
                }
            }
        }

        debug!(
            "Graph bidirectional: {} seeds, found {} nodes",
            seeds.len(),
            results.len()
        );

        results
    }

    /// Scored bidirectional traversal.
    pub fn scored_bidirectional_traverse(
        &self,
        seeds: &[ObjectId],
        max_depth: usize,
        relation_filter: Option<&str>,
    ) -> Vec<(ObjectId, f32)> {
        let bfs_results = self.bidirectional_traverse(seeds, max_depth, relation_filter);
        bfs_results
            .into_iter()
            .map(|(id, depth)| {
                let score = 1.0 / (1.0 + depth as f32);
                (id, score)
            })
            .collect()
    }

    /// Find the shortest path between two objects (returns hop count, or None).
    pub fn shortest_path(&self, from: ObjectId, to: ObjectId) -> Option<usize> {
        let mut visited: HashMap<ObjectId, usize> = HashMap::new();
        let mut queue: VecDeque<(ObjectId, usize)> = VecDeque::new();

        visited.insert(from, 0);
        queue.push_back((from, 0));

        let forward = self.forward.read();

        while let Some((current, depth)) = queue.pop_front() {
            if current == to {
                return Some(depth);
            }

            if let Some(edges) = forward.get(&current) {
                for rel in edges {
                    if !visited.contains_key(&rel.target_id) {
                        visited.insert(rel.target_id, depth + 1);
                        queue.push_back((rel.target_id, depth + 1));
                    }
                }
            }
        }

        None
    }

    /// Count of nodes in the graph.
    pub fn node_count(&self) -> usize {
        let forward = self.forward.read();
        let mut all_ids: std::collections::HashSet<ObjectId> = forward.keys().copied().collect();
        let reverse = self.reverse.read();
        all_ids.extend(reverse.keys().copied());
        all_ids.len()
    }

    /// Count of edges in the graph.
    pub fn edge_count(&self) -> usize {
        self.forward.read().values().map(|rels| rels.len()).sum()
    }
}

impl Default for GraphIndex {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_insert_and_traverse() {
        let graph = GraphIndex::new();
        let a = uuid::Uuid::new_v4();
        let b = uuid::Uuid::new_v4();
        let c = uuid::Uuid::new_v4();
        let d = uuid::Uuid::new_v4();

        // A -> B -> C
        // A -> D
        graph.insert_relationships(
            a,
            &[
                Relationship {
                    relation_type: "references".into(),
                    target_id: b,
                    weight: None,
                },
                Relationship {
                    relation_type: "references".into(),
                    target_id: d,
                    weight: None,
                },
            ],
        );
        graph.insert_relationships(
            b,
            &[Relationship {
                relation_type: "contains".into(),
                target_id: c,
                weight: None,
            }],
        );

        // Out edges from A
        let out = graph.out_edges(a);
        assert_eq!(out.len(), 2);

        // Reverse: C is referenced by B
        let incoming = graph.in_edges(c);
        assert_eq!(incoming.len(), 1);
        assert_eq!(incoming[0].1, b);

        // BFS from A, max depth 2
        let bfs = graph.bfs_traverse(&[a], 2, None);
        let ids: Vec<_> = bfs.iter().map(|(id, _)| *id).collect();
        assert!(ids.contains(&a));
        assert!(ids.contains(&b));
        assert!(ids.contains(&c));
        assert!(ids.contains(&d));
    }

    #[test]
    fn test_scored_traverse() {
        let graph = GraphIndex::new();
        let a = uuid::Uuid::new_v4();
        let b = uuid::Uuid::new_v4();
        let c = uuid::Uuid::new_v4();

        graph.insert_relationships(
            a,
            &[Relationship {
                relation_type: "links_to".into(),
                target_id: b,
                weight: None,
            }],
        );
        graph.insert_relationships(
            b,
            &[Relationship {
                relation_type: "links_to".into(),
                target_id: c,
                weight: None,
            }],
        );

        let scores = graph.scored_traverse(&[a], 3, None);
        // A: 1.0, B: 0.5, C: 0.333
        let a_score = scores.iter().find(|(id, _)| *id == a).unwrap().1;
        let c_score = scores.iter().find(|(id, _)| *id == c).unwrap().1;
        assert!((a_score - 1.0).abs() < 1e-6);
        assert!((c_score - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_shortest_path() {
        let graph = GraphIndex::new();
        let a = uuid::Uuid::new_v4();
        let b = uuid::Uuid::new_v4();
        let c = uuid::Uuid::new_v4();

        graph.insert_relationships(
            a,
            &[Relationship {
                relation_type: "x".into(),
                target_id: b,
                weight: None,
            }],
        );
        graph.insert_relationships(
            b,
            &[Relationship {
                relation_type: "x".into(),
                target_id: c,
                weight: None,
            }],
        );

        assert_eq!(graph.shortest_path(a, c), Some(2));
        assert_eq!(graph.shortest_path(a, a), Some(0));
        // No path between disconnected nodes
        let z = uuid::Uuid::new_v4();
        assert_eq!(graph.shortest_path(a, z), None);
    }
}
