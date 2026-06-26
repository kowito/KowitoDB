use serde::{Deserialize, Serialize};

use crate::types::*;

/// A Knowledge Object is the fundamental unit of storage in KowitoDB.
///
/// Unlike a vector database that stores only embeddings, each Knowledge Object
/// contains rich multimodal data:
/// - Raw content
/// - Embeddings (multiple, for different models or modalities)
/// - Arbitrary metadata
/// - Extracted keywords
/// - Named relationships to other objects
/// - Version history
/// - An importance score for prioritization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeObject {
    /// Unique identifier.
    pub id: ObjectId,

    /// The raw textual content of this knowledge object.
    pub content: String,

    /// One or more embedding vectors.
    /// Keys identify the embedding model or modality (e.g., "text-embedding-3-small").
    pub embeddings: std::collections::HashMap<String, Embedding>,

    /// Arbitrary key-value metadata (source URL, author, timestamp, etc.).
    pub metadata: Metadata,

    /// Extracted keywords for full-text search.
    pub keywords: Vec<String>,

    /// Named relationships to other knowledge objects (graph edges).
    pub relationships: Vec<Relationship>,

    /// History of versions of this object.
    pub version_history: Vec<VersionEntry>,

    /// Importance score (0.0 = low, 1.0 = critical).
    /// The planner can use this to prioritize results.
    pub importance: Importance,

    /// When this object was first created.
    pub created_at: Timestamp,

    /// When this object was last modified.
    pub updated_at: Timestamp,
}

impl KnowledgeObject {
    /// Create a new KnowledgeObject with sensible defaults.
    pub fn new(content: impl Into<String>) -> Self {
        let now = chrono::Utc::now();
        Self {
            id: uuid::Uuid::new_v4(),
            content: content.into(),
            embeddings: Default::default(),
            metadata: Default::default(),
            keywords: Vec::new(),
            relationships: Vec::new(),
            version_history: Vec::new(),
            importance: 0.5,
            created_at: now,
            updated_at: now,
        }
    }

    /// Attach an embedding for a given model name.
    pub fn with_embedding(mut self, model: impl Into<String>, embedding: Embedding) -> Self {
        self.embeddings.insert(model.into(), embedding);
        self
    }

    /// Set metadata key-value.
    pub fn with_metadata(
        mut self,
        key: impl Into<String>,
        value: impl Into<serde_json::Value>,
    ) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Add keywords.
    pub fn with_keywords(mut self, keywords: Vec<String>) -> Self {
        self.keywords = keywords;
        self
    }

    /// Add a relationship to another object.
    pub fn with_relationship(
        mut self,
        relation_type: impl Into<String>,
        target_id: ObjectId,
    ) -> Self {
        self.relationships.push(Relationship {
            relation_type: relation_type.into(),
            target_id,
            weight: None,
        });
        self
    }

    /// Set importance.
    pub fn with_importance(mut self, importance: Importance) -> Self {
        self.importance = importance.clamp(0.0, 1.0);
        self
    }

    /// Add a version entry to history.
    pub fn record_version(&mut self, description: Option<String>) {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(self.content.as_bytes());
        let hash = hex::encode(hasher.finalize());
        self.version_history.push(VersionEntry {
            content_hash: hash,
            timestamp: chrono::Utc::now(),
            description,
        });
        self.updated_at = chrono::Utc::now();
    }

    /// Convenience: get the primary embedding if one exists.
    pub fn primary_embedding(&self) -> Option<&Embedding> {
        self.embeddings.values().next()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_knowledge_object() {
        let obj = KnowledgeObject::new("Hello, KowitoDB!")
            .with_importance(0.8)
            .with_keywords(vec!["hello".into(), "test".into()])
            .with_metadata("source", "unit-test");
        assert!(!obj.id.is_nil());
        assert_eq!(obj.content, "Hello, KowitoDB!");
        assert_eq!(obj.importance, 0.8);
        assert_eq!(obj.keywords.len(), 2);
    }

    #[test]
    fn test_embedding_attachment() {
        let obj = KnowledgeObject::new("test")
            .with_embedding("text-embedding-3-small", vec![0.1, 0.2, 0.3]);
        assert_eq!(obj.embeddings.len(), 1);
        assert!(obj.primary_embedding().is_some());
    }

    #[test]
    fn test_versioning() {
        let mut obj = KnowledgeObject::new("v1");
        obj.record_version(Some("Initial".into()));
        assert_eq!(obj.version_history.len(), 1);
    }
}
