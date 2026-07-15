//! `KowitoDBEngine` — sql methods (split from the former db.rs god-file).

use super::*;

impl KowitoDBEngine {
    /// Execute a SQL query against knowledge objects.
    ///
    /// Maps SQL WHERE clauses to the metadata, keyword, and time indexes.
    /// Results are loaded from storage with real content.
    pub async fn sql_query(&self, sql: &str) -> KResult<Vec<LoadedResult>> {
        let stmt = kowitodb_sql::parse_sql(sql)
            .map_err(|e| kowitodb_core::KowitoError::Planner(e.to_string()))?;

        let (where_clauses, limit) = match stmt {
            kowitodb_sql::SqlStatement::Select {
                where_clauses,
                limit,
                ..
            } => (where_clauses, limit),
        };

        let mut candidate_sets: Vec<Vec<ObjectId>> = Vec::new();

        for clause in &where_clauses {
            match clause {
                kowitodb_sql::WhereClause::MetadataEquals { key, value } => {
                    let ids = self.metadata_index.query_exact(key, value);
                    if !ids.is_empty() {
                        candidate_sets.push(ids);
                    }
                }
                kowitodb_sql::WhereClause::MetadataContains { key, substring } => {
                    let ids = self.metadata_index.query_contains(key, substring);
                    if !ids.is_empty() {
                        candidate_sets.push(ids);
                    }
                }
                kowitodb_sql::WhereClause::KeywordContains { substring } => {
                    // Use full-text search for keyword contains
                    if let Ok(results) = self.fulltext_index.search(substring, 100) {
                        if !results.is_empty() {
                            candidate_sets.push(results.into_iter().map(|(id, _)| id).collect());
                        }
                    }
                }
                kowitodb_sql::WhereClause::ContentContains { substring } => {
                    if let Ok(results) = self.fulltext_index.search(substring, 100) {
                        if !results.is_empty() {
                            candidate_sets.push(results.into_iter().map(|(id, _)| id).collect());
                        }
                    }
                }
                kowitodb_sql::WhereClause::CreatedAfter { timestamp } => {
                    // Parse timestamp to milliseconds
                    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(timestamp) {
                        let ids = self.time_index.after(dt.timestamp_millis());
                        if !ids.is_empty() {
                            candidate_sets.push(ids);
                        }
                    }
                }
                kowitodb_sql::WhereClause::CreatedBefore { timestamp } => {
                    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(timestamp) {
                        let ids = self.time_index.before(dt.timestamp_millis());
                        if !ids.is_empty() {
                            candidate_sets.push(ids);
                        }
                    }
                }
                _ => {
                    debug!("SQL clause not yet routed to index: {:?}", clause);
                }
            }
        }

        // Intersect candidate sets (AND semantics)
        let final_ids: Vec<ObjectId> = if candidate_sets.is_empty() {
            // No WHERE clauses: get all objects
            self.storage.list_ids().await?
        } else if candidate_sets.len() == 1 {
            candidate_sets.into_iter().next().unwrap()
        } else {
            // Intersect all sets
            let mut sets: Vec<std::collections::HashSet<ObjectId>> = candidate_sets
                .into_iter()
                .map(|v| v.into_iter().collect())
                .collect();
            let (first, rest) = sets.split_at_mut(1);
            first[0].retain(|id| rest.iter().all(|s| s.contains(id)));
            first[0].iter().copied().collect()
        };

        // Apply limit
        let final_ids: Vec<ObjectId> = if let Some(lim) = limit {
            final_ids.into_iter().take(lim).collect()
        } else {
            final_ids
        };

        // Load content for results
        let mut loaded = Vec::with_capacity(final_ids.len());
        for id in &final_ids {
            let content = if let Some(cached) = self.content_cache.get(id) {
                cached
            } else if let Ok(Some(stored)) = self.storage.get(*id).await {
                let val = stored.content.clone();
                self.content_cache.insert(*id, val.clone());
                val
            } else {
                format!("<Object {}>", id)
            };

            loaded.push(LoadedResult {
                id: *id,
                content,
                relevance_score: 1.0,
                retrieval_sources: vec!["sql".to_string()],
                metadata: HashMap::new(),
                importance: 0.5,
            });
        }

        Ok(loaded)
    }

    /// Execute arbitrary SQL through the DataFusion engine.
    ///
    /// Unlike [`Self::sql_query`] (which routes a small parsed subset to the
    /// native indexes), this runs the full DataFusion planner over the
    /// `knowledge` table provider — supporting projections, `ORDER BY`,
    /// `GROUP BY`, aggregates, and complex predicates. Rows are returned as
    /// ordered column-name → stringified-value maps.
    pub async fn sql_select(&self, sql: &str) -> KResult<Vec<HashMap<String, String>>> {
        // Read-only guard: this runs client- and LLM-generated SQL (the NL→SQL
        // feature) against DataFusion, which can otherwise CREATE tables,
        // `COPY ... TO` the filesystem, or read external files. Since the engine
        // ingests untrusted documents (RAG), reject anything that isn't a single
        // read-only SELECT/WITH statement before executing it.
        if !is_read_only_sql(sql) {
            return Err(kowitodb_core::KowitoError::Planner(
                "only a single read-only SELECT/WITH query is permitted".into(),
            ));
        }
        let ctx = kowitodb_sql::SqlContext::new(self.storage.clone())
            .map_err(|e| kowitodb_core::KowitoError::Planner(e.to_string()))?;
        ctx.query_rows(sql)
            .await
            .map_err(|e| kowitodb_core::KowitoError::Planner(e.to_string()))
    }

    /// Translate a natural-language question into a single SQL query over the
    /// `knowledge` table using the LLM. `None` when no client is configured.
    pub(crate) async fn nl_to_sql(&self, question: &str) -> Option<String> {
        let llm = self.llm_client.as_ref()?;
        let system = "Translate the user's question into ONE SQL query over a \
            table `knowledge` with columns: id (text), content (text), \
            importance (real), created_at (text), updated_at (text), keywords \
            (text), metadata (text, JSON). Use only SELECT. Output only the SQL \
            — no markdown fences, no explanation.";
        match llm.complete(system, question).await {
            Ok(sql) => Some(strip_sql_fence(&sql)),
            Err(e) => {
                debug!("NL→SQL translation failed: {e}");
                None
            }
        }
    }

    /// Answer an analytical/aggregational question by translating it to SQL via
    /// the LLM and executing it over the store. Returns `None` when no LLM
    /// client is configured (callers fall back to retrieval).
    pub async fn answer_with_sql(
        &self,
        question: &str,
    ) -> KResult<Option<Vec<HashMap<String, String>>>> {
        let Some(sql) = self.nl_to_sql(question).await else {
            return Ok(None);
        };
        debug!("NL→SQL: {sql}");
        Ok(Some(self.sql_select(&sql).await?))
    }
}
