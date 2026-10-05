use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;

use kowitodb_core::{KowitoError, ObjectId, Result};
use parking_lot::RwLock;
use tantivy::collector::{DocSetCollector, TopDocs};
use tantivy::query::{AllQuery, QueryParser};
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
    ///
    /// The change is staged in the writer; call [`Self::commit`] to make it
    /// durable and visible to searches. Batching several inserts before one
    /// commit is much cheaper than committing each.
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

        writer
            .add_document(doc!(
                self.id_field => id.to_string(),
                self.content_field => content.to_string(),
                self.keywords_field => keywords.join(" "),
                self.metadata_field => metadata_json.to_string(),
            ))
            .map_err(|e| KowitoError::Index(e.to_string()))?;

        debug!("Full-text staged object {}", id);
        Ok(())
    }

    /// Remove a document from the index.
    ///
    /// Like [`Self::insert`], the change is staged: it becomes durable and
    /// visible to searches after [`Self::commit`].
    pub fn remove(&self, id: ObjectId) -> Result<()> {
        let mut writer_guard = self.writer.write();
        let writer = writer_guard
            .as_mut()
            .ok_or_else(|| KowitoError::Internal("FullTextIndex writer already closed".into()))?;
        let id_term = tantivy::Term::from_field_text(self.id_field, &id.to_string());
        writer.delete_term(id_term);
        Ok(())
    }

    /// Search the index and return top-k matching object IDs with BM25 scores.
    ///
    /// `query_str` is treated as plain text, not Tantivy query syntax:
    /// punctuation (`:`, quotes, parentheses, `-`, ...) and the `AND`/`OR`/`NOT`
    /// operators are neutralised, so natural-language questions always search
    /// their words instead of failing to parse.
    pub fn search(&self, query_str: &str, limit: usize) -> Result<Vec<(ObjectId, f32)>> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        // The default tokenizer lowercases and splits on non-alphanumerics, so
        // this keeps every searchable term while removing query syntax.
        let query_str: String = query_str
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { ' ' })
            .collect::<String>()
            .to_lowercase();
        if query_str.trim().is_empty() {
            return Ok(Vec::new());
        }
        let reader = self.reader.searcher();

        let query_parser = QueryParser::for_index(
            &self.index,
            vec![self.content_field, self.keywords_field, self.metadata_field],
        );

        let (query, errors) = query_parser.parse_query_lenient(&query_str);
        if !errors.is_empty() {
            debug!("Lenient query parse of {:?}: {:?}", query_str, errors);
        }

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

    /// Ids of every live document in the index (used to reconcile the index
    /// against storage after an unclean shutdown).
    pub fn ids(&self) -> Result<HashSet<ObjectId>> {
        let searcher = self.reader.searcher();
        let addresses = searcher
            .search(&AllQuery, &DocSetCollector)
            .map_err(|e| KowitoError::Index(e.to_string()))?;
        let mut ids = HashSet::with_capacity(addresses.len());
        for address in addresses {
            let doc = searcher
                .doc::<TantivyDocument>(address)
                .map_err(|e| KowitoError::Index(e.to_string()))?;
            if let Some(id) = doc
                .get_first(self.id_field)
                .and_then(|v| v.as_str())
                .and_then(|s| uuid::Uuid::parse_str(s).ok())
            {
                ids.insert(id);
            }
        }
        Ok(ids)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn open_temp() -> FullTextIndex {
        let dir = std::env::temp_dir().join(format!("kowitodb-ft-{}", uuid::Uuid::new_v4()));
        FullTextIndex::open(&dir).unwrap()
    }

    #[test]
    fn staged_changes_become_visible_on_commit() {
        let idx = open_temp();
        let (a, b) = (uuid::Uuid::new_v4(), uuid::Uuid::new_v4());
        idx.insert(a, "rust vector database", &[], "{}").unwrap();
        idx.insert(b, "python notebook", &[], "{}").unwrap();
        idx.commit().unwrap();
        assert_eq!(idx.search("vector", 10).unwrap()[0].0, a);
        assert_eq!(idx.ids().unwrap(), HashSet::from([a, b]));

        idx.remove(a).unwrap();
        idx.commit().unwrap();
        assert!(idx.search("vector", 10).unwrap().is_empty());
        assert_eq!(idx.ids().unwrap(), HashSet::from([b]));
    }

    #[test]
    fn natural_language_queries_do_not_fail_to_parse() {
        let idx = open_temp();
        let a = uuid::Uuid::new_v4();
        idx.insert(a, "error timeout while connecting", &[], "{}")
            .unwrap();
        idx.commit().unwrap();
        for q in [
            "error: timeout",
            "what about \"timeout",
            "(timeout",
            "http://x/y",
            "NOT AND",
        ] {
            assert!(idx.search(q, 10).is_ok(), "query {q:?} failed");
        }
        assert_eq!(idx.search("error: timeout", 10).unwrap()[0].0, a);
        assert!(idx.search("timeout", 0).unwrap().is_empty());
    }
}
