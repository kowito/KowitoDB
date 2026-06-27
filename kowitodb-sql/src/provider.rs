//! DataFusion integration for KowitoDB.
//!
//! Exposes knowledge objects to the Apache DataFusion query engine through a
//! custom [`TableProvider`], so arbitrary SQL — projections, `WHERE`, `ORDER BY`,
//! `GROUP BY`, aggregates, `LIMIT` — executes over the live storage engine.
//!
//! This replaces the hand-rolled parser path (kept in [`crate::parser`] as a
//! lightweight fallback) with a real cost-based SQL engine.

use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

use datafusion::arrow::array::{ArrayRef, Float32Array, RecordBatch, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::catalog::Session;
use datafusion::datasource::{MemTable, TableProvider, TableType};
use datafusion::error::{DataFusionError, Result as DfResult};
use datafusion::execution::context::SessionContext;
use datafusion::logical_expr::Expr;
use datafusion::physical_plan::ExecutionPlan;

use kowitodb_storage::{StorageBackend, StorageFilter};

/// The table name(s) under which knowledge objects are exposed to SQL.
pub const TABLE_KNOWLEDGE: &str = "knowledge";
pub const TABLE_OBJECTS: &str = "objects";

/// A DataFusion [`TableProvider`] backed by the KowitoDB storage engine.
///
/// Each scan materializes the current set of stored objects into a single
/// Arrow [`RecordBatch`] and delegates execution to an in-memory plan. This
/// keeps the provider correct and version-robust; a future upgrade can stream
/// batches and push projection/filter predicates down into storage.
pub struct KnowledgeTableProvider {
    storage: Arc<dyn StorageBackend>,
    schema: SchemaRef,
}

impl std::fmt::Debug for KnowledgeTableProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KnowledgeTableProvider")
            .field("schema", &self.schema)
            .finish_non_exhaustive()
    }
}

impl KnowledgeTableProvider {
    /// Create a provider over the given storage backend.
    pub fn new(storage: Arc<dyn StorageBackend>) -> Self {
        Self {
            storage,
            schema: Self::arrow_schema(),
        }
    }

    /// The flat Arrow schema projected from a `StoredObject`.
    ///
    /// `metadata` and `keywords` are surfaced as their JSON string encodings so
    /// they remain queryable (e.g. `metadata LIKE '%"stage":"series_a"%'`)
    /// without committing to a fixed metadata schema.
    pub fn arrow_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("content", DataType::Utf8, false),
            Field::new("importance", DataType::Float32, false),
            Field::new("created_at", DataType::Utf8, false),
            Field::new("updated_at", DataType::Utf8, false),
            Field::new("keywords", DataType::Utf8, false),
            Field::new("metadata", DataType::Utf8, false),
        ]))
    }

    /// Read stored objects (optionally limited) and pack them into one batch.
    async fn build_batch(&self, limit: Option<usize>) -> DfResult<RecordBatch> {
        let objects = self
            .storage
            .search(StorageFilter {
                limit,
                ..Default::default()
            })
            .await
            .map_err(|e| DataFusionError::External(Box::new(e)))?;

        let mut ids = Vec::with_capacity(objects.len());
        let mut contents = Vec::with_capacity(objects.len());
        let mut importances = Vec::with_capacity(objects.len());
        let mut created = Vec::with_capacity(objects.len());
        let mut updated = Vec::with_capacity(objects.len());
        let mut keywords = Vec::with_capacity(objects.len());
        let mut metadata = Vec::with_capacity(objects.len());

        for obj in objects {
            ids.push(obj.id.to_string());
            contents.push(obj.content);
            importances.push(obj.importance);
            created.push(obj.created_at);
            updated.push(obj.updated_at);
            keywords.push(obj.keywords_json);
            metadata.push(obj.metadata_json);
        }

        let columns: Vec<ArrayRef> = vec![
            Arc::new(StringArray::from(ids)),
            Arc::new(StringArray::from(contents)),
            Arc::new(Float32Array::from(importances)),
            Arc::new(StringArray::from(created)),
            Arc::new(StringArray::from(updated)),
            Arc::new(StringArray::from(keywords)),
            Arc::new(StringArray::from(metadata)),
        ];

        RecordBatch::try_new(self.schema.clone(), columns).map_err(DataFusionError::from)
    }
}

#[async_trait::async_trait]
impl TableProvider for KnowledgeTableProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        // Snapshot current storage into an in-memory table and let DataFusion's
        // MemTable handle projection/limit; remaining filters are applied by a
        // FilterExec the optimizer inserts above this scan.
        //
        // Push the row limit into storage only when there are no filters — with
        // a WHERE clause, limiting before filtering would drop matching rows.
        let scan_limit = if filters.is_empty() { limit } else { None };
        let batch = self.build_batch(scan_limit).await?;
        let mem = MemTable::try_new(self.schema.clone(), vec![vec![batch]])?;
        mem.scan(state, projection, filters, limit).await
    }
}

/// A ready-to-query SQL session with the `knowledge` table registered.
pub struct SqlContext {
    ctx: SessionContext,
}

impl SqlContext {
    /// Build a session over the given storage and register the knowledge table
    /// (aliased as both `knowledge` and `objects`).
    pub fn new(storage: Arc<dyn StorageBackend>) -> DfResult<Self> {
        let ctx = SessionContext::new();
        ctx.register_table(
            TABLE_KNOWLEDGE,
            Arc::new(KnowledgeTableProvider::new(storage.clone())),
        )?;
        ctx.register_table(
            TABLE_OBJECTS,
            Arc::new(KnowledgeTableProvider::new(storage)),
        )?;
        Ok(Self { ctx })
    }

    /// Run a SQL query and collect the resulting record batches.
    pub async fn sql(&self, query: &str) -> DfResult<Vec<RecordBatch>> {
        let df = self.ctx.sql(query).await?;
        df.collect().await
    }

    /// Run a SQL query and return rows as ordered column-name → value maps.
    ///
    /// All values are stringified for transport-agnostic consumption; numeric
    /// and other typed columns are rendered via their Arrow display formatting.
    pub async fn query_rows(&self, query: &str) -> DfResult<Vec<HashMap<String, String>>> {
        let batches = self.sql(query).await?;
        let mut rows = Vec::new();

        for batch in &batches {
            let schema = batch.schema();
            for row_idx in 0..batch.num_rows() {
                let mut row = HashMap::with_capacity(schema.fields().len());
                for (col_idx, field) in schema.fields().iter().enumerate() {
                    let array = batch.column(col_idx);
                    let value = array_value_to_string(array, row_idx);
                    row.insert(field.name().clone(), value);
                }
                rows.push(row);
            }
        }

        Ok(rows)
    }

    /// Access the underlying DataFusion session (e.g. to register more tables).
    pub fn session(&self) -> &SessionContext {
        &self.ctx
    }
}

/// Render a single cell of an Arrow array as a string.
fn array_value_to_string(array: &ArrayRef, row: usize) -> String {
    use datafusion::arrow::util::display::{ArrayFormatter, FormatOptions};
    match ArrayFormatter::try_new(array.as_ref(), &FormatOptions::default()) {
        Ok(formatter) => formatter.value(row).to_string(),
        Err(_) => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kowitodb_storage::{StorageEngine, StoredObject};

    fn obj(id_seed: u128, content: &str, importance: f32, metadata: &str) -> StoredObject {
        StoredObject {
            id: uuid::Uuid::from_u128(id_seed),
            content: content.to_string(),
            metadata_json: metadata.to_string(),
            keywords_json: "[]".to_string(),
            relationships_json: "[]".to_string(),
            embeddings_json: "{}".to_string(),
            version_history_json: "[]".to_string(),
            importance,
            created_at: "2024-01-01T00:00:00Z".to_string(),
            updated_at: "2024-01-01T00:00:00Z".to_string(),
        }
    }

    async fn ctx_with_objects() -> SqlContext {
        let storage = Arc::new(StorageEngine::new_in_memory().unwrap());
        storage
            .put(obj(
                1,
                "Acme raised Series A",
                0.9,
                r#"{"stage":"series_a"}"#,
            ))
            .await
            .unwrap();
        storage
            .put(obj(
                2,
                "Globex raised Series B",
                0.4,
                r#"{"stage":"series_b"}"#,
            ))
            .await
            .unwrap();
        storage
            .put(obj(3, "Initech churned", 0.7, r#"{"stage":"series_a"}"#))
            .await
            .unwrap();
        SqlContext::new(storage).unwrap()
    }

    #[tokio::test]
    async fn test_projection_and_filter() {
        let ctx = ctx_with_objects().await;
        let rows = ctx
            .query_rows(
                "SELECT content FROM knowledge WHERE importance >= 0.5 ORDER BY importance DESC",
            )
            .await
            .unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0]["content"], "Acme raised Series A");
        assert_eq!(rows[1]["content"], "Initech churned");
        // Projection: only the requested column is present.
        assert!(!rows[0].contains_key("importance"));
    }

    #[tokio::test]
    async fn test_aggregate() {
        let ctx = ctx_with_objects().await;
        let rows = ctx
            .query_rows("SELECT COUNT(*) AS n, AVG(importance) AS avg_imp FROM objects")
            .await
            .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["n"], "3");
    }

    #[tokio::test]
    async fn test_metadata_json_like() {
        let ctx = ctx_with_objects().await;
        let rows = ctx
            .query_rows("SELECT id FROM knowledge WHERE metadata LIKE '%series_a%'")
            .await
            .unwrap();
        assert_eq!(rows.len(), 2);
    }
}
