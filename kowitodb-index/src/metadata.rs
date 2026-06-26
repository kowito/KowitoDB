use std::collections::HashMap;
use std::sync::Arc;

use kowitodb_core::ObjectId;
use parking_lot::RwLock;
use tracing::debug;

/// Maps each distinct metadata value to the objects carrying it.
type ValueIndex = HashMap<String, Vec<ObjectId>>;

/// In-memory metadata index.
///
/// Maps metadata key-value pairs to object IDs for fast filtering
/// by arbitrary attributes. In production, this could be backed by
/// a columnar store or SQLite.
pub struct MetadataIndex {
    /// key -> (value -> set of object IDs)
    index: Arc<RwLock<HashMap<String, ValueIndex>>>,
}

impl MetadataIndex {
    pub fn new() -> Self {
        Self {
            index: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Index a metadata key-value pair for an object.
    pub fn insert(&self, id: ObjectId, key: &str, value: &str) {
        let mut index = self.index.write();
        let values = index.entry(key.to_string()).or_default();
        let ids = values.entry(value.to_string()).or_default();
        if !ids.contains(&id) {
            ids.push(id);
        }
        debug!("Metadata indexed: {}={} -> {}", key, value, id);
    }

    /// Remove an object from all metadata entries.
    pub fn remove_object(&self, id: ObjectId) {
        let mut index = self.index.write();
        for values in index.values_mut() {
            for ids in values.values_mut() {
                ids.retain(|x| *x != id);
            }
        }
        // Clean up empty value maps
        for values in index.values_mut() {
            values.retain(|_, ids| !ids.is_empty());
        }
        index.retain(|_, values| !values.is_empty());
    }

    /// Query by exact metadata key-value match.
    pub fn query_exact(&self, key: &str, value: &str) -> Vec<ObjectId> {
        let index = self.index.read();
        index
            .get(key)
            .and_then(|values| values.get(value))
            .cloned()
            .unwrap_or_default()
    }

    /// Query by metadata key (returns all object IDs with that key).
    pub fn query_by_key(&self, key: &str) -> Vec<ObjectId> {
        let index = self.index.read();
        let mut ids = Vec::new();
        if let Some(values) = index.get(key) {
            for obj_ids in values.values() {
                ids.extend(obj_ids);
            }
        }
        ids.sort();
        ids.dedup();
        ids
    }

    /// Query by partial value match (substring).
    pub fn query_contains(&self, key: &str, substring: &str) -> Vec<ObjectId> {
        let index = self.index.read();
        let mut ids = Vec::new();
        if let Some(values) = index.get(key) {
            for (val, obj_ids) in values.iter() {
                if val.contains(substring) {
                    ids.extend(obj_ids);
                }
            }
        }
        ids.sort();
        ids.dedup();
        ids
    }
}

impl Default for MetadataIndex {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata_index() {
        let idx = MetadataIndex::new();
        let id1 = uuid::Uuid::new_v4();
        let id2 = uuid::Uuid::new_v4();

        idx.insert(id1, "source", "web");
        idx.insert(id2, "source", "api");
        idx.insert(id1, "author", "Alice");

        assert_eq!(idx.query_exact("source", "web"), vec![id1]);
        assert_eq!(idx.query_exact("source", "api"), vec![id2]);
        assert_eq!(idx.query_contains("source", "a").len(), 1);
        assert_eq!(idx.query_by_key("author"), vec![id1]);

        idx.remove_object(id1);
        assert!(idx.query_exact("source", "web").is_empty());
        assert_eq!(idx.query_exact("source", "api"), vec![id2]);
    }
}
