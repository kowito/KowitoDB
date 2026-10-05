use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use kowitodb_core::ObjectId;
use parking_lot::RwLock;
use tracing::debug;

/// Forward and reverse maps, kept under a single lock so updates are atomic
/// and there is no lock ordering to get wrong.
#[derive(Default)]
struct Inner {
    /// Timestamp (milliseconds since epoch) -> list of object IDs.
    index: BTreeMap<i64, Vec<ObjectId>>,
    /// Reverse map: object ID -> timestamp for updates.
    reverse: HashMap<ObjectId, i64>,
}

impl Inner {
    fn remove(&mut self, id: ObjectId) {
        if let Some(ts) = self.reverse.remove(&id) {
            if let Some(ids) = self.index.get_mut(&ts) {
                ids.retain(|x| *x != id);
                if ids.is_empty() {
                    self.index.remove(&ts);
                }
            }
        }
    }
}

/// Time-based index mapping timestamps to object IDs.
///
/// Uses a BTreeMap for range queries. Supports queries like
/// "after date X", "before date Y", "between X and Y".
pub struct TimeIndex {
    inner: Arc<RwLock<Inner>>,
}

impl TimeIndex {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(Inner::default())),
        }
    }

    /// Insert or update an object at a given timestamp.
    pub fn insert(&self, id: ObjectId, ts_ms: i64) {
        let mut inner = self.inner.write();
        inner.remove(id);
        inner.index.entry(ts_ms).or_default().push(id);
        inner.reverse.insert(id, ts_ms);

        debug!("Time indexed: {} at ts={}", id, ts_ms);
    }

    /// Remove an object from the time index.
    pub fn remove(&self, id: ObjectId) {
        self.inner.write().remove(id);
    }

    /// Query objects created after a timestamp (inclusive).
    pub fn after(&self, ts_ms: i64) -> Vec<ObjectId> {
        let inner = self.inner.read();
        let mut ids = Vec::new();
        for (_ts, obj_ids) in inner.index.range(ts_ms..) {
            ids.extend(obj_ids);
        }
        ids
    }

    /// Query objects created before a timestamp (inclusive).
    pub fn before(&self, ts_ms: i64) -> Vec<ObjectId> {
        let inner = self.inner.read();
        let mut ids = Vec::new();
        for (_ts, obj_ids) in inner.index.range(..=ts_ms) {
            ids.extend(obj_ids);
        }
        ids
    }

    /// Query objects created between two timestamps (inclusive). An inverted
    /// range (`start_ms > end_ms`) matches nothing.
    pub fn between(&self, start_ms: i64, end_ms: i64) -> Vec<ObjectId> {
        if start_ms > end_ms {
            return Vec::new();
        }
        let inner = self.inner.read();
        let mut ids = Vec::new();
        for (_ts, obj_ids) in inner.index.range(start_ms..=end_ms) {
            ids.extend(obj_ids);
        }
        ids
    }

    /// Number of indexed objects.
    pub fn len(&self) -> usize {
        self.inner.read().reverse.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clear the index.
    pub fn clear(&self) {
        let mut inner = self.inner.write();
        inner.index.clear();
        inner.reverse.clear();
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
        assert!(idx.between(2500, 1500).is_empty());

        // Re-inserting moves the object rather than duplicating it.
        idx.insert(id1, 4000);
        assert_eq!(idx.after(3000), vec![id3, id1]);
        assert_eq!(idx.len(), 3);
        idx.remove(id1);
        assert_eq!(idx.after(0), vec![id2, id3]);
    }

    #[test]
    fn concurrent_insert_and_remove_do_not_deadlock() {
        let idx = Arc::new(TimeIndex::new());
        let ids: Vec<_> = (0..64).map(|_| uuid::Uuid::new_v4()).collect();
        let handles: Vec<_> = (0..8)
            .map(|t| {
                let idx = idx.clone();
                let ids = ids.clone();
                std::thread::spawn(move || {
                    for round in 0..500 {
                        let id = ids[(t * 7 + round) % ids.len()];
                        if round % 3 == 0 {
                            idx.remove(id);
                        } else {
                            idx.insert(id, (round % 50) as i64);
                        }
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert!(idx.len() <= ids.len());
        assert_eq!(idx.after(i64::MIN).len(), idx.len());
    }
}
