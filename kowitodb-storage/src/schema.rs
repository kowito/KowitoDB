use kowitodb_core::{ObjectId, Result, Timestamp};
use serde::{Deserialize, Serialize};

/// Materialized result returned by the storage engine after a query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredObject {
    pub id: ObjectId,
    pub content: String,
    pub metadata_json: String,
    pub keywords_json: String,
    pub relationships_json: String,
    pub embeddings_json: String,
    /// Version history as JSON (`[]` when none). `serde(default)` keeps records
    /// written before this field was added readable.
    #[serde(default = "empty_json_array")]
    pub version_history_json: String,
    pub importance: f32,
    pub created_at: String, // ISO-8601
    pub updated_at: String,
}

fn empty_json_array() -> String {
    "[]".to_string()
}

/// Search filter for the storage layer.
#[derive(Debug, Clone, Default)]
pub struct StorageFilter {
    pub id: Option<ObjectId>,
    pub keyword: Option<String>,
    pub metadata_key: Option<String>,
    pub metadata_value: Option<String>,
    pub min_importance: Option<f32>,
    pub created_after: Option<Timestamp>,
    pub created_before: Option<Timestamp>,
    pub limit: Option<usize>,
}

/// Test whether a stored object satisfies every predicate in `filter`
/// (excluding `limit`, which bounds result count rather than rows).
///
/// Shared by all backends so filter semantics stay identical regardless of how
/// much of the predicate was pushed down to the storage layer.
pub fn filter_matches(obj: &StoredObject, filter: &StorageFilter) -> bool {
    if let Some(ref target_id) = filter.id {
        if obj.id != *target_id {
            return false;
        }
    }
    if let Some(ref kw) = filter.keyword {
        let keywords: Vec<String> = serde_json::from_str(&obj.keywords_json).unwrap_or_default();
        if !keywords.iter().any(|k| k.contains(kw)) {
            return false;
        }
    }
    if let Some(min_imp) = filter.min_importance {
        if obj.importance < min_imp {
            return false;
        }
    }
    if let Some(ref after) = filter.created_after {
        if let Ok(parsed) = chrono::DateTime::parse_from_rfc3339(&obj.created_at) {
            if parsed < *after {
                return false;
            }
        }
    }
    if let Some(ref before) = filter.created_before {
        if let Ok(parsed) = chrono::DateTime::parse_from_rfc3339(&obj.created_at) {
            if parsed > *before {
                return false;
            }
        }
    }
    true
}

/// Defines operations the storage engine must support.
#[async_trait::async_trait]
pub trait StorageBackend: Send + Sync {
    /// Insert or update a stored object.
    async fn put(&self, obj: StoredObject) -> Result<()>;

    /// Retrieve an object by ID.
    async fn get(&self, id: ObjectId) -> Result<Option<StoredObject>>;

    /// Delete an object by ID.
    async fn delete(&self, id: ObjectId) -> Result<bool>;

    /// List objects matching a filter.
    async fn search(&self, filter: StorageFilter) -> Result<Vec<StoredObject>>;

    /// Count total objects.
    async fn count(&self) -> Result<usize>;

    /// Iterate over all object IDs.
    async fn list_ids(&self) -> Result<Vec<ObjectId>>;
}
