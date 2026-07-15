//! `HnswIndex` — snapshot serialization (to_bytes / from_bytes / save / load) (split from the former hnsw.rs god-file).

use super::*;

impl HnswIndex {
    /// Serialize the index to a byte buffer.
    pub fn to_bytes(&self) -> std::io::Result<Vec<u8>> {
        let graph = self.graph.read();
        let snapshot = HnswSnapshotRef {
            params: &self.params,
            nodes: &graph.nodes,
            entry_point: *self.entry_point.read(),
            max_layer: *self.max_layer.read(),
            rotation: self.rotation.read().clone(),
            dim: *self.dim.read(),
        };
        bincode::serialize(&snapshot).map_err(std::io::Error::other)
    }

    /// Reconstruct an index from a buffer produced by [`Self::to_bytes`].
    pub fn from_bytes(bytes: &[u8]) -> std::io::Result<Self> {
        let snapshot: HnswSnapshot = bincode::deserialize(bytes).map_err(std::io::Error::other)?;
        // Rebuild the id → index map from the dense node vector.
        let mut id_to_idx =
            HashMap::with_capacity_and_hasher(snapshot.nodes.len(), ahash::RandomState::default());
        for (i, node) in snapshot.nodes.iter().enumerate() {
            id_to_idx.insert(node.id, i as NodeIdx);
        }
        Ok(Self {
            graph: Arc::new(RwLock::new(Graph {
                nodes: snapshot.nodes,
                id_to_idx,
            })),
            entry_point: Arc::new(RwLock::new(snapshot.entry_point)),
            max_layer: Arc::new(RwLock::new(snapshot.max_layer)),
            rotation: Arc::new(RwLock::new(snapshot.rotation)),
            dim: Arc::new(RwLock::new(snapshot.dim)),
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
}
