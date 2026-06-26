use std::collections::BTreeMap;
use std::sync::Arc;

use kowitodb_core::ObjectId;
use parking_lot::RwLock;
use tracing::debug;

/// Time-based index mapping timestamps to object IDs.
///
/// Uses a BTreeMap for range queries. Supports queries like
/// "after date X", "before date Y", "between X and Y".
pub struct TimeIndex {
    /// Timestamp (milliseconds since epoch) -> list of object IDs.
    index: Arc<RwLock<BTreeMap<i64, Vec<ObjectId>>>>,
    /// Reverse map: object ID -> timestamp for updates.
    reverse: Arc<RwLock<std::collections::HashMap<ObjectId, i64>>>,
}

impl TimeIndex {
    pub fn new() -> Self {
        Self {
            index: Arc::new(RwLock::new(BTreeMap::new())),
            reverse: Arc::new(RwLock::new(std::collections::HashMap::new())),
        }
    }

    /// Insert or update an object at a given timestamp.
    pub fn insert(&self, id: ObjectId, ts_ms: i64) {
        // Remove old entry if it exists
        {
            let reverse = self.reverse.read();
            if let Some(_old_ts) = reverse.get(&id) {
                // We need to modify; drop read lock and take write
                drop(reverse);
                self.remove(id);
            }
        }

        let mut index = self.index.write();
        index.entry(ts_ms).or_default().push(id);

        let mut reverse = self.reverse.write();
        reverse.insert(id, ts_ms);

        debug!("Time indexed: {} at ts={}", id, ts_ms);
    }

    /// Remove an object from the time index.
    pub fn remove(&self, id: ObjectId) {
        let mut reverse = self.reverse.write();
        if let Some(ts) = reverse.remove(&id) {
            let mut index = self.index.write();
            if let Some(ids) = index.get_mut(&ts) {
                ids.retain(|x| *x != id);
            }
            // Clean up empty buckets
            index.retain(|_, ids| !ids.is_empty());
        }
    }

    /// Query objects created after a timestamp (inclusive).
    pub fn after(&self, ts_ms: i64) -> Vec<ObjectId> {
        let index = self.index.read();
        let mut ids = Vec::new();
        for (_ts, obj_ids) in index.range(ts_ms..) {
            ids.extend(obj_ids);
        }
        ids
    }

    /// Query objects created before a timestamp (inclusive).
    pub fn before(&self, ts_ms: i64) -> Vec<ObjectId> {
        let index = self.index.read();
        let mut ids = Vec::new();
        for (_ts, obj_ids) in index.range(..=ts_ms) {
            ids.extend(obj_ids);
        }
        ids
    }

    /// Query objects created between two timestamps (inclusive).
    pub fn between(&self, start_ms: i64, end_ms: i64) -> Vec<ObjectId> {
        let index = self.index.read();
        let mut ids = Vec::new();
        for (_ts, obj_ids) in index.range(start_ms..=end_ms) {
            ids.extend(obj_ids);
        }
        ids
    }

    /// Clear the index.
    pub fn clear(&self) {
        self.index.write().clear();
        self.reverse.write().clear();
    }
}

impl Default for TimeIndex {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_index() {
        let idx = TimeIndex::new();
        let id1 = uuid::Uuid::new_v4();
        let id2 = uuid::Uuid::new_v4();
        let id3 = uuid::Uuid::new_v4();

        idx.insert(id1, 1000);
        idx.insert(id2, 2000);
        idx.insert(id3, 3000);

        assert_eq!(idx.after(2000), vec![id2, id3]);
        assert_eq!(idx.before(2000), vec![id1, id2]);
        assert_eq!(idx.between(1500, 2500), vec![id2]);
    }
}
