use std::path::Path;
use std::sync::Arc;

use dashmap::DashMap;
use kowitodb_core::{KowitoError, ObjectId, Result};
use parking_lot::Mutex;
use sled::Db;
use tracing::{debug, info};

use super::schema::{filter_matches, StorageBackend, StorageFilter, StoredObject};

/// Upper bound on cached objects; beyond it, reads go straight to sled.
const CACHE_CAP: usize = 10_000;

/// Number of lock stripes serializing per-id cache/db updates.
const LOCK_STRIPES: usize = 64;

/// Sled-backed storage engine.
///
/// Uses a bounded in-memory `DashMap` read cache over a persistent `sled`
/// database. Every write is flushed to disk before it is acknowledged.
pub struct StorageEngine {
    db: Db,
    cache: Arc<DashMap<ObjectId, StoredObject>>,
    /// Striped per-id locks held across a db operation and its cache update,
    /// so a concurrent read can't re-populate the cache with a stale value.
    locks: Arc<[Mutex<()>]>,
}

impl StorageEngine {
    /// Open (or create) the storage engine at the given path.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path_ref = path.as_ref();
        let db = sled::open(path_ref).map_err(|e| KowitoError::Storage(e.to_string()))?;
        info!("Storage engine opened at {:?}", path_ref);
        Ok(Self::from_db(db))
    }

    /// Create a new in-memory engine (for testing or ephemeral use).
    pub fn new_in_memory() -> Result<Self> {
        let db = sled::Config::new()
            .temporary(true)
            .open()
            .map_err(|e| KowitoError::Storage(e.to_string()))?;
        Ok(Self::from_db(db))
    }

    fn from_db(db: Db) -> Self {
        Self {
            db,
            cache: Arc::new(DashMap::new()),
            locks: (0..LOCK_STRIPES).map(|_| Mutex::new(())).collect(),
        }
    }

    fn lock_for(&self, id: ObjectId) -> &Mutex<()> {
        &self.locks[(id.as_u128() % LOCK_STRIPES as u128) as usize]
    }

    /// Serialize an object ID to bytes for sled key.
    fn key_bytes(id: ObjectId) -> Vec<u8> {
        id.as_bytes().to_vec()
    }

    /// Make every acknowledged write durable.
    async fn flush(&self) -> Result<()> {
        self.db
            .flush_async()
            .await
            .map_err(|e| KowitoError::Storage(e.to_string()))?;
        Ok(())
    }
}

#[async_trait::async_trait]
impl StorageBackend for StorageEngine {
    async fn put(&self, obj: StoredObject) -> Result<()> {
        let id = obj.id;
        let key = Self::key_bytes(id);
        let value =
            serde_json::to_vec(&obj).map_err(|e| KowitoError::Serialization(e.to_string()))?;

        {
            let _guard = self.lock_for(id).lock();
            self.db
                .insert(key, value)
                .map_err(|e| KowitoError::Storage(e.to_string()))?;
            self.cache.remove(&id);
        }
        self.flush().await?;

        debug!("Stored object {}", id);
        Ok(())
    }

    async fn get(&self, id: ObjectId) -> Result<Option<StoredObject>> {
        if let Some(obj) = self.cache.get(&id) {
            return Ok(Some(obj.clone()));
        }

        let _guard = self.lock_for(id).lock();
        let key = Self::key_bytes(id);
        let raw = self
            .db
            .get(key)
            .map_err(|e| KowitoError::Storage(e.to_string()))?;

        match raw {
            Some(ivec) => {
                let obj: StoredObject = serde_json::from_slice(&ivec)
                    .map_err(|e| KowitoError::Serialization(e.to_string()))?;
                if self.cache.len() < CACHE_CAP {
                    self.cache.insert(id, obj.clone());
                }
                Ok(Some(obj))
            }
            None => Ok(None),
        }
    }

    async fn delete(&self, id: ObjectId) -> Result<bool> {
        let key = Self::key_bytes(id);
        let existed = {
            let _guard = self.lock_for(id).lock();
            let existed = self
                .db
                .remove(key)
                .map_err(|e| KowitoError::Storage(e.to_string()))?
                .is_some();
            self.cache.remove(&id);
            existed
        };
        if existed {
            self.flush().await?;
        }
        debug!("Deleted object {}: {}", id, existed);
        Ok(existed)
    }

    async fn search(&self, filter: StorageFilter) -> Result<Vec<StoredObject>> {
        // Fast path: filtering by id is a single key lookup, not a full scan.
        if let Some(target_id) = filter.id {
            return Ok(match self.get(target_id).await? {
                Some(obj) if filter_matches(&obj, &filter) => vec![obj],
                _ => Vec::new(),
            });
        }

        // Fallback scan over all objects, applying the remaining predicates.
        let mut results: Vec<StoredObject> = Vec::new();
        for item in self.db.iter() {
            let (_key, value) = item.map_err(|e| KowitoError::Storage(e.to_string()))?;
            let obj: StoredObject = serde_json::from_slice(&value)
                .map_err(|e| KowitoError::Serialization(e.to_string()))?;

            if !filter_matches(&obj, &filter) {
                continue;
            }
            results.push(obj);

            if let Some(limit) = filter.limit {
                if results.len() >= limit {
                    break;
                }
            }
        }

        Ok(results)
    }

    async fn count(&self) -> Result<usize> {
        Ok(self.db.len())
    }

    async fn list_ids(&self) -> Result<Vec<ObjectId>> {
        let mut ids = Vec::new();
        for item in self.db.iter() {
            let (key, _) = item.map_err(|e| KowitoError::Storage(e.to_string()))?;
            if key.len() == 16 {
                let mut bytes = [0u8; 16];
                bytes.copy_from_slice(&key);
                ids.push(uuid::Uuid::from_bytes(bytes));
            }
        }
        Ok(ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stored(id: ObjectId, content: &str, metadata_json: &str) -> StoredObject {
        StoredObject {
            id,
            content: content.into(),
            metadata_json: metadata_json.into(),
            keywords_json: "[]".into(),
            relationships_json: "[]".into(),
            embeddings_json: "{}".into(),
            version_history_json: "[]".into(),
            importance: 0.5,
            created_at: "2024-01-01T00:00:00Z".into(),
            updated_at: "2024-01-01T00:00:00Z".into(),
        }
    }

    #[tokio::test]
    async fn reads_see_the_latest_write() {
        let engine = StorageEngine::new_in_memory().unwrap();
        let id = uuid::Uuid::new_v4();
        engine.put(stored(id, "v1", "{}")).await.unwrap();
        assert_eq!(engine.get(id).await.unwrap().unwrap().content, "v1");
        engine.put(stored(id, "v2", "{}")).await.unwrap();
        assert_eq!(engine.get(id).await.unwrap().unwrap().content, "v2");
        assert!(engine.delete(id).await.unwrap());
        assert!(engine.get(id).await.unwrap().is_none());
        assert!(!engine.delete(id).await.unwrap());
    }

    #[tokio::test]
    async fn search_honors_metadata_filter() {
        let engine = StorageEngine::new_in_memory().unwrap();
        let (a, b) = (uuid::Uuid::new_v4(), uuid::Uuid::new_v4());
        engine
            .put(stored(a, "a", r#"{"tenant":"acme","n":3}"#))
            .await
            .unwrap();
        engine
            .put(stored(b, "b", r#"{"tenant":"globex"}"#))
            .await
            .unwrap();

        let by_value = |key: &str, value: Option<&str>| StorageFilter {
            metadata_key: Some(key.into()),
            metadata_value: value.map(Into::into),
            ..Default::default()
        };
        let hits = engine
            .search(by_value("tenant", Some("acme")))
            .await
            .unwrap();
        assert_eq!(hits.iter().map(|o| o.id).collect::<Vec<_>>(), vec![a]);
        let hits = engine.search(by_value("n", Some("3"))).await.unwrap();
        assert_eq!(hits.len(), 1);
        let hits = engine.search(by_value("n", None)).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert!(engine
            .search(by_value("tenant", Some("none")))
            .await
            .unwrap()
            .is_empty());
    }
}
