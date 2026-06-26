use std::collections::HashMap;

/// An embedding vector — a dense float array representing semantic meaning.
pub type Embedding = Vec<f32>;

/// A unique identifier for knowledge objects.
pub type ObjectId = uuid::Uuid;

/// A timestamp wrapper.
pub type Timestamp = chrono::DateTime<chrono::Utc>;

/// Metadata attached to a knowledge object — arbitrary key-value.
pub type Metadata = HashMap<String, serde_json::Value>;

/// Importance score (0.0 - 1.0) for prioritizing results.
pub type Importance = f32;

/// Named relationship between two knowledge objects.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Relationship {
    /// Type of relationship: "depends_on", "references", "contains", etc.
    pub relation_type: String,
    /// Target object ID.
    pub target_id: ObjectId,
    /// Optional weight (0.0 - 1.0).
    pub weight: Option<f32>,
}

/// A version entry recording a change to a knowledge object.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VersionEntry {
    /// Hash of the content at this version.
    pub content_hash: String,
    /// When this version was created.
    pub timestamp: Timestamp,
    /// Human-readable description of the change.
    pub description: Option<String>,
}
