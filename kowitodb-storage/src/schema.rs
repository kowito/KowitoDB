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
