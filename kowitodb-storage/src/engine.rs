use std::path::Path;
use std::sync::Arc;

use dashmap::DashMap;
use kowitodb_core::{KowitoError, ObjectId, Result};
use sled::Db;
use tracing::{debug, info};

use super::schema::{filter_matches, StorageBackend, StorageFilter, StoredObject};

/// Sled-backed storage engine.
///
/// Uses an in-memory `DashMap` cache over a persistent `sled` database.
/// Sled provides an ACID-compliant embedded DB with B+ tree indexing,
/// suitable for single-node deployment in Phase 1.
pub struct StorageEngine {
    db: Db,
    cache: Arc<DashMap<ObjectId, StoredObject>>,
}

impl StorageEngine {
    /// Open (or create) the storage engine at the given path.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path_ref = path.as_ref();
        let db = sled::open(path_ref).map_err(|e| KowitoError::Storage(e.to_string()))?;
        let cache = Arc::new(DashMap::new());

        info!("Storage engine opened at {:?}", path_ref);
        Ok(Self { db, cache })
    }

    /// Create a new in-memory engine (for testing or ephemeral use).
    pub fn new_in_memory() -> Result<Self> {
        let db = sled::Config::new()
            .temporary(true)
            .open()
            .map_err(|e| KowitoError::Storage(e.to_string()))?;
        let cache = Arc::new(DashMap::new());
        Ok(Self { db, cache })
    }

    /// Serialize an object ID to bytes for sled key.
    fn key_bytes(id: ObjectId) -> Vec<u8> {
        id.as_bytes().to_vec()
    }
}

#[async_trait::async_trait]
impl StorageBackend for StorageEngine {
    async fn put(&self, obj: StoredObject) -> Result<()> {
        let id = obj.id;
        let key = Self::key_bytes(id);
        let value =
            serde_json::to_vec(&obj).map_err(|e| KowitoError::Serialization(e.to_string()))?;

        self.db
            .insert(key, value)
            .map_err(|e| KowitoError::Storage(e.to_string()))?;

        self.cache.insert(id, obj);
        debug!("Stored object {}", id);
        Ok(())
    }

    async fn get(&self, id: ObjectId) -> Result<Option<StoredObject>> {
        if let Some(obj) = self.cache.get(&id) {
            return Ok(Some(obj.clone()));
        }

        let key = Self::key_bytes(id);
        let raw = self
            .db
            .get(key)
            .map_err(|e| KowitoError::Storage(e.to_string()))?;

        match raw {
            Some(ivec) => {
                let obj: StoredObject = serde_json::from_slice(&ivec)
                    .map_err(|e| KowitoError::Serialization(e.to_string()))?;
                self.cache.insert(id, obj.clone());
                Ok(Some(obj))
            }
            None => Ok(None),
        }
    }

    async fn delete(&self, id: ObjectId) -> Result<bool> {
        self.cache.remove(&id);
        let key = Self::key_bytes(id);
        let existed = self
            .db
            .remove(key)
            .map_err(|e| KowitoError::Storage(e.to_string()))?
            .is_some();
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
