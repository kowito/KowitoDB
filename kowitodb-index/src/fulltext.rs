use std::path::Path;
use std::sync::Arc;

use kowitodb_core::{KowitoError, ObjectId, Result};
use parking_lot::RwLock;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::*;
use tantivy::{doc, Index, IndexReader, IndexWriter, ReloadPolicy};
use tracing::{debug, info};

/// Full-text search index backed by Tantivy.
///
/// Tantivy provides BM25 scoring, tokenization, and inverted-index search
/// comparable to Lucene. This is used for keyword queries.
pub struct FullTextIndex {
    index: Index,
    reader: IndexReader,
    #[allow(dead_code)]
    schema: Schema,
    #[allow(dead_code)]
    /// We need a writer for inserts; Tantivy requires a single writer.
    writer: Arc<RwLock<Option<IndexWriter>>>,
    /// Pre-allocated field handles.
    id_field: Field,
    content_field: Field,
    keywords_field: Field,
    metadata_field: Field,
}

impl FullTextIndex {
    /// Open an on-disk index at `path`, creating it if needed.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let mut schema_builder = Schema::builder();
        let id_field = schema_builder.add_text_field("id", STRING | STORED);
        let content_field = schema_builder.add_text_field("content", TEXT);
        let keywords_field = schema_builder.add_text_field("keywords", TEXT);
        let metadata_field = schema_builder.add_text_field("metadata", TEXT);
        let schema = schema_builder.build();

        let index_path = path.as_ref().join("tantivy");
        std::fs::create_dir_all(&index_path).map_err(KowitoError::Io)?;

        let index = if index_path.join("meta.json").exists() {
            Index::open_in_dir(&index_path)
                .map_err(|e| KowitoError::Index(format!("Failed to open Tantivy index: {}", e)))?
        } else {
            Index::create_in_dir(&index_path, schema.clone())
                .map_err(|e| KowitoError::Index(format!("Failed to create Tantivy index: {}", e)))?
        };

        let writer = index
            .writer(50_000_000) // 50 MB buffer
            .map_err(|e| KowitoError::Index(e.to_string()))?;

        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()
            .map_err(|e| KowitoError::Index(e.to_string()))?;

        info!("Full-text index opened at {:?}", index_path);

        Ok(Self {
            index,
            reader,
            schema,
            writer: Arc::new(RwLock::new(Some(writer))),
            id_field,
            content_field,
            keywords_field,
            metadata_field,
        })
    }

    /// Insert or update a document in the index.
    pub fn insert(
        &self,
        id: ObjectId,
        content: &str,
        keywords: &[String],
        metadata_json: &str,
    ) -> Result<()> {
        let mut writer_guard = self.writer.write();
        let writer = writer_guard
            .as_mut()
            .ok_or_else(|| KowitoError::Internal("FullTextIndex writer already closed".into()))?;

        // Delete any existing document with this ID
        let id_term = tantivy::Term::from_field_text(self.id_field, &id.to_string());
        writer.delete_term(id_term);

        // Insert new document
        // Build the full searchable text
        let full_text = format!("{} {}", content, keywords.join(" "));
        let _ = full_text; // Used implicitly via field tokenization
        writer
            .add_document(doc!(
                self.id_field => id.to_string(),
                self.content_field => content.to_string(),
                self.keywords_field => keywords.join(" "),
                self.metadata_field => metadata_json.to_string(),
            ))
            .map_err(|e| KowitoError::Index(e.to_string()))?;

        let _ = writer.commit();
        debug!("Full-text indexed object {}", id);
        Ok(())
    }

    /// Remove a document from the index.
    pub fn remove(&self, id: ObjectId) -> Result<()> {
        let mut writer_guard = self.writer.write();
        let writer = writer_guard
            .as_mut()
            .ok_or_else(|| KowitoError::Internal("FullTextIndex writer already closed".into()))?;
        let id_term = tantivy::Term::from_field_text(self.id_field, &id.to_string());
        writer.delete_term(id_term);
        let _ = writer.commit();
        Ok(())
    }

    /// Search the index and return top-k matching object IDs with BM25 scores.
    pub fn search(&self, query_str: &str, limit: usize) -> Result<Vec<(ObjectId, f32)>> {
        let reader = self.reader.searcher();

        let query_parser = QueryParser::for_index(
            &self.index,
            vec![self.content_field, self.keywords_field, self.metadata_field],
        );

        let query = query_parser
            .parse_query(query_str)
            .map_err(|e| KowitoError::Index(format!("Query parse error: {}", e)))?;

        let top_docs = reader
            .search(&query, &TopDocs::with_limit(limit))
            .map_err(|e| KowitoError::Index(e.to_string()))?;

        let mut results = Vec::with_capacity(top_docs.len());
        for (score, doc_address) in top_docs {
            let doc = reader
                .doc::<TantivyDocument>(doc_address)
                .map_err(|e| KowitoError::Index(e.to_string()))?;
            if let Some(id_str) = doc.get_first(self.id_field) {
                if let Some(id_text) = id_str.as_str() {
                    if let Ok(id) = uuid::Uuid::parse_str(id_text) {
                        results.push((id, score));
                    }
                }
            }
        }

        Ok(results)
    }

    /// Commit pending writes and reload the reader.
    pub fn commit(&self) -> Result<()> {
        let mut writer_guard = self.writer.write();
        if let Some(writer) = writer_guard.as_mut() {
            writer
                .commit()
                .map_err(|e| KowitoError::Index(e.to_string()))?;
        }
        // Force reader to pick up the commit
        self.reader
            .reload()
            .map_err(|e| KowitoError::Index(e.to_string()))?;
        Ok(())
    }
}
