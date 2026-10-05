use std::collections::HashMap;
use std::sync::Arc;

use kowitodb_core::ObjectId;
use parking_lot::RwLock;
use tracing::debug;

/// Maps each distinct metadata value to the objects carrying it.
type ValueIndex = HashMap<String, Vec<ObjectId>>;

#[derive(Default)]
struct Inner {
    /// key -> (value -> object IDs, in insertion order)
    index: HashMap<String, ValueIndex>,
    /// object ID -> the (key, value) pairs it is indexed under, so removal
    /// and duplicate checks don't have to scan the whole index.
    by_object: HashMap<ObjectId, Vec<(String, String)>>,
}

/// In-memory metadata index.
///
/// Maps metadata key-value pairs to object IDs for fast filtering
/// by arbitrary attributes. In production, this could be backed by
/// a columnar store or SQLite.
pub struct MetadataIndex {
    inner: Arc<RwLock<Inner>>,
}

impl MetadataIndex {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(Inner::default())),
        }
    }

    /// Index a metadata key-value pair for an object.
    pub fn insert(&self, id: ObjectId, key: &str, value: &str) {
        let mut guard = self.inner.write();
        let inner = &mut *guard;
        let pairs = inner.by_object.entry(id).or_default();
        if pairs.iter().any(|(k, v)| k == key && v == value) {
            return;
        }
        pairs.push((key.to_string(), value.to_string()));
        inner
            .index
            .entry(key.to_string())
            .or_default()
            .entry(value.to_string())
            .or_default()
            .push(id);
        debug!("Metadata indexed: {}={} -> {}", key, value, id);
    }

    /// Remove an object from all metadata entries.
    pub fn remove_object(&self, id: ObjectId) {
        let mut guard = self.inner.write();
        let inner = &mut *guard;
        let Some(pairs) = inner.by_object.remove(&id) else {
            return;
        };
        for (key, value) in pairs {
            let Some(values) = inner.index.get_mut(&key) else {
                continue;
            };
            if let Some(ids) = values.get_mut(&value) {
                ids.retain(|x| *x != id);
                if ids.is_empty() {
                    values.remove(&value);
                }
            }
            if values.is_empty() {
                inner.index.remove(&key);
            }
        }
    }

    /// Query by exact metadata key-value match.
    pub fn query_exact(&self, key: &str, value: &str) -> Vec<ObjectId> {
        let inner = self.inner.read();
        inner
            .index
            .get(key)
            .and_then(|values| values.get(value))
            .cloned()
            .unwrap_or_default()
    }

    /// Query by metadata key (returns all object IDs with that key).
    pub fn query_by_key(&self, key: &str) -> Vec<ObjectId> {
        let inner = self.inner.read();
        let mut ids = Vec::new();
        if let Some(values) = inner.index.get(key) {
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
        let inner = self.inner.read();
        let mut ids = Vec::new();
        if let Some(values) = inner.index.get(key) {
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
        assert!(idx.query_by_key("author").is_empty());

        // Duplicate inserts are ignored; re-indexing after removal works.
        idx.insert(id2, "source", "api");
        assert_eq!(idx.query_exact("source", "api"), vec![id2]);
        idx.insert(id1, "source", "api");
        assert_eq!(idx.query_exact("source", "api"), vec![id2, id1]);
    }
}
