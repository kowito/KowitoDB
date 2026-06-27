//! Lance-backed storage engine (feature `lance`).
//!
//! [Lance](https://lancedb.github.io/lance/) is a modern columnar format built
//! for ML/AI data: fast random access, zero-copy versioning, and native vector
//! support. This backend persists [`StoredObject`]s as an Arrow table in a Lance
//! dataset, implementing the same [`StorageBackend`] contract as the default
//! sled engine so it is a drop-in alternative.
//!
//! `put` uses delete-then-append upsert semantics keyed on `id`; reads use Lance
//! scans. The dataset is created lazily on first write.

use std::sync::Arc;

use arrow::array::{Array, Float32Array, RecordBatch, RecordBatchIterator, StringArray};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use kowitodb_core::{KowitoError, ObjectId, Result};
use lance::dataset::{Dataset, WriteMode, WriteParams};
use tokio::sync::RwLock;
use tracing::{debug, info};

use crate::schema::{filter_matches, StorageBackend, StorageFilter, StoredObject};

/// Lance-backed implementation of [`StorageBackend`].
pub struct LanceStorage {
    uri: String,
    schema: SchemaRef,
    /// `None` until the dataset has been created (on first write) or opened.
    dataset: Arc<RwLock<Option<Dataset>>>,
}

impl LanceStorage {
    /// Open an existing Lance dataset at `uri`, or prepare to create one on the
    /// first write if none exists yet.
    pub async fn open(uri: impl Into<String>) -> Result<Self> {
        let uri = uri.into();
        let schema = Self::arrow_schema();
        let existing = Dataset::open(&uri).await.ok();
        if existing.is_some() {
            info!("Opened existing Lance dataset at {}", uri);
        } else {
            debug!("No Lance dataset at {} yet; will create on first write", uri);
        }
        Ok(Self {
            uri,
            schema,
            dataset: Arc::new(RwLock::new(existing)),
        })
    }

    /// The Arrow schema mirroring every field of a [`StoredObject`].
    fn arrow_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("content", DataType::Utf8, false),
            Field::new("metadata_json", DataType::Utf8, false),
            Field::new("keywords_json", DataType::Utf8, false),
            Field::new("relationships_json", DataType::Utf8, false),
            Field::new("embeddings_json", DataType::Utf8, false),
            Field::new("version_history_json", DataType::Utf8, false),
            Field::new("importance", DataType::Float32, false),
            Field::new("created_at", DataType::Utf8, false),
            Field::new("updated_at", DataType::Utf8, false),
        ]))
    }

    /// Build a single-row record batch from a stored object.
    fn batch_from(&self, obj: &StoredObject) -> Result<RecordBatch> {
        RecordBatch::try_new(
            self.schema.clone(),
            vec![
                Arc::new(StringArray::from(vec![obj.id.to_string()])),
                Arc::new(StringArray::from(vec![obj.content.clone()])),
                Arc::new(StringArray::from(vec![obj.metadata_json.clone()])),
                Arc::new(StringArray::from(vec![obj.keywords_json.clone()])),
                Arc::new(StringArray::from(vec![obj.relationships_json.clone()])),
                Arc::new(StringArray::from(vec![obj.embeddings_json.clone()])),
                Arc::new(StringArray::from(vec![obj.version_history_json.clone()])),
                Arc::new(Float32Array::from(vec![obj.importance])),
                Arc::new(StringArray::from(vec![obj.created_at.clone()])),
                Arc::new(StringArray::from(vec![obj.updated_at.clone()])),
            ],
        )
        .map_err(|e| KowitoError::Storage(format!("arrow batch: {e}")))
    }

    /// Reconstruct stored objects from a scanned record batch.
    fn objects_from(batch: &RecordBatch) -> Result<Vec<StoredObject>> {
        let col = |name: &str| -> Result<&StringArray> {
            batch
                .column_by_name(name)
                .and_then(|c| c.as_any().downcast_ref::<StringArray>())
                .ok_or_else(|| KowitoError::Storage(format!("missing/typed column {name}")))
        };
        let id = col("id")?;
        let content = col("content")?;
        let metadata_json = col("metadata_json")?;
        let keywords_json = col("keywords_json")?;
        let relationships_json = col("relationships_json")?;
        let embeddings_json = col("embeddings_json")?;
        let version_history_json = col("version_history_json")?;
        let importance = batch
            .column_by_name("importance")
            .and_then(|c| c.as_any().downcast_ref::<Float32Array>())
            .ok_or_else(|| KowitoError::Storage("missing importance column".into()))?;
        let created_at = col("created_at")?;
        let updated_at = col("updated_at")?;

        let mut out = Vec::with_capacity(batch.num_rows());
        for i in 0..batch.num_rows() {
            let id_str = id.value(i);
            let parsed = ObjectId::parse_str(id_str)
                .map_err(|e| KowitoError::Storage(format!("bad uuid {id_str}: {e}")))?;
            out.push(StoredObject {
                id: parsed,
                content: content.value(i).to_string(),
                metadata_json: metadata_json.value(i).to_string(),
                keywords_json: keywords_json.value(i).to_string(),
                relationships_json: relationships_json.value(i).to_string(),
                embeddings_json: embeddings_json.value(i).to_string(),
                version_history_json: version_history_json.value(i).to_string(),
                importance: importance.value(i),
                created_at: created_at.value(i).to_string(),
                updated_at: updated_at.value(i).to_string(),
            });
        }
        Ok(out)
    }

}

#[async_trait::async_trait]
impl StorageBackend for LanceStorage {
    async fn put(&self, obj: StoredObject) -> Result<()> {
        let id = obj.id;
        let batch = self.batch_from(&obj)?;
        let mut guard = self.dataset.write().await;

        match guard.as_mut() {
            // Dataset exists: upsert via delete-then-append.
            Some(ds) => {
                ds.delete(&format!("id = '{id}'")).await.map_err(map_lance)?;
                let reader = RecordBatchIterator::new(vec![Ok(batch)], self.schema.clone());
                ds.append(reader, None).await.map_err(map_lance)?;
            }
            // First write: create the dataset.
            None => {
                let reader = RecordBatchIterator::new(vec![Ok(batch)], self.schema.clone());
                let params = WriteParams {
                    mode: WriteMode::Create,
                    ..Default::default()
                };
                let ds = Dataset::write(reader, self.uri.as_str(), Some(params))
                    .await
                    .map_err(map_lance)?;
                *guard = Some(ds);
            }
        }
        debug!("Lance stored object {}", id);
        Ok(())
    }

    async fn get(&self, id: ObjectId) -> Result<Option<StoredObject>> {
        let guard = self.dataset.read().await;
        let Some(ds) = guard.as_ref() else {
            return Ok(None);
        };
        let mut scanner = ds.scan();
        scanner
            .filter(&format!("id = '{id}'"))
            .map_err(map_lance)?;
        let batch = scanner.try_into_batch().await.map_err(map_lance)?;
        if batch.num_rows() == 0 {
            return Ok(None);
        }
        Ok(Self::objects_from(&batch)?.into_iter().next())
    }

    async fn delete(&self, id: ObjectId) -> Result<bool> {
        let mut guard = self.dataset.write().await;
        let Some(ds) = guard.as_mut() else {
            return Ok(false);
        };
        let predicate = format!("id = '{id}'");
        let existed = ds
            .count_rows(Some(predicate.clone()))
            .await
            .map_err(map_lance)?
            > 0;
        if existed {
            ds.delete(&predicate).await.map_err(map_lance)?;
            debug!("Lance deleted object {}", id);
        }
        Ok(existed)
    }

    async fn search(&self, filter: StorageFilter) -> Result<Vec<StoredObject>> {
        let guard = self.dataset.read().await;
        let Some(ds) = guard.as_ref() else {
            return Ok(Vec::new());
        };

        // Push the predicates Lance can evaluate natively (id, importance) down
        // into the scan; the remaining predicates (keyword JSON, time) are
        // applied in Rust via the shared filter to keep semantics identical.
        let mut pushdown: Vec<String> = Vec::new();
        if let Some(id) = filter.id {
            pushdown.push(format!("id = '{id}'"));
        }
        if let Some(min_imp) = filter.min_importance {
            pushdown.push(format!("importance >= {min_imp}"));
        }

        let mut scanner = ds.scan();
        if !pushdown.is_empty() {
            scanner
                .filter(&pushdown.join(" AND "))
                .map_err(map_lance)?;
        }
        let batch = scanner.try_into_batch().await.map_err(map_lance)?;
        if batch.num_rows() == 0 {
            return Ok(Vec::new());
        }

        let mut results = Vec::new();
        for obj in Self::objects_from(&batch)? {
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
        let guard = self.dataset.read().await;
        match guard.as_ref() {
            Some(ds) => ds.count_rows(None).await.map_err(map_lance),
            None => Ok(0),
        }
    }

    async fn list_ids(&self) -> Result<Vec<ObjectId>> {
        let guard = self.dataset.read().await;
        let Some(ds) = guard.as_ref() else {
            return Ok(Vec::new());
        };
        if ds.count_rows(None).await.map_err(map_lance)? == 0 {
            return Ok(Vec::new());
        }
        let mut scanner = ds.scan();
        scanner.project(&["id"]).map_err(map_lance)?;
        let batch = scanner.try_into_batch().await.map_err(map_lance)?;
        let id_col = batch
            .column_by_name("id")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>())
            .ok_or_else(|| KowitoError::Storage("missing id column".into()))?;
        let mut ids = Vec::with_capacity(id_col.len());
        for i in 0..id_col.len() {
            if let Ok(id) = ObjectId::parse_str(id_col.value(i)) {
                ids.push(id);
            }
        }
        Ok(ids)
    }
}

fn map_lance(e: lance::Error) -> KowitoError {
    KowitoError::Storage(format!("lance: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obj(seed: u128, content: &str, importance: f32) -> StoredObject {
        StoredObject {
            id: ObjectId::from_u128(seed),
            content: content.to_string(),
            metadata_json: "{}".to_string(),
            keywords_json: "[\"alpha\"]".to_string(),
            relationships_json: "[]".to_string(),
            embeddings_json: "{}".to_string(),
            version_history_json: "[]".to_string(),
            importance,
            created_at: "2024-01-01T00:00:00Z".to_string(),
            updated_at: "2024-01-01T00:00:00Z".to_string(),
        }
    }

    async fn temp_store() -> LanceStorage {
        let dir = std::env::temp_dir().join(format!("kowitodb-lance-{}", uuid::Uuid::new_v4()));
        LanceStorage::open(dir.to_string_lossy().to_string())
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn test_put_get_delete_roundtrip() {
        let store = temp_store().await;
        let o = obj(1, "hello lance", 0.8);
        let id = o.id;

        store.put(o).await.unwrap();
        let got = store.get(id).await.unwrap().expect("should exist");
        assert_eq!(got.content, "hello lance");
        assert_eq!(store.count().await.unwrap(), 1);

        // Upsert: same id, new content.
        store.put(obj(1, "updated", 0.9)).await.unwrap();
        let got = store.get(id).await.unwrap().unwrap();
        assert_eq!(got.content, "updated");
        assert_eq!(store.count().await.unwrap(), 1, "upsert must not duplicate");

        assert!(store.delete(id).await.unwrap());
        assert!(store.get(id).await.unwrap().is_none());
        assert!(!store.delete(id).await.unwrap());
    }

    #[tokio::test]
    async fn test_search_and_list() {
        let store = temp_store().await;
        store.put(obj(1, "a", 0.2)).await.unwrap();
        store.put(obj(2, "b", 0.9)).await.unwrap();

        let high = store
            .search(StorageFilter {
                min_importance: Some(0.5),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(high.len(), 1);
        assert_eq!(high[0].content, "b");

        let mut ids = store.list_ids().await.unwrap();
        ids.sort();
        assert_eq!(ids.len(), 2);
    }
}
