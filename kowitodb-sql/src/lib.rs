//! SQL bridge for KowitoDB.
//!
//! Parses simple SQL queries and routes them to the appropriate indexes.
//! Supports SELECT with WHERE clauses on metadata, keywords, and time ranges.
//! Full DataFusion integration is the upgrade path.

use std::collections::HashMap;

use kowitodb_core::ObjectId;
use serde::{Deserialize, Serialize};

mod parser;

pub use parser::parse as parse_sql;

/// Parsed representation of a SQL-like query against knowledge objects.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SqlStatement {
    /// SELECT ... FROM knowledge [WHERE ...] [LIMIT n]
    Select {
        columns: Vec<SelectColumn>,
        where_clauses: Vec<WhereClause>,
        limit: Option<usize>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SelectColumn {
    All,
    Named(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WhereClause {
    /// metadata.key = 'value'
    MetadataEquals { key: String, value: String },
    /// metadata.key LIKE '%value%'
    MetadataContains { key: String, substring: String },
    /// keyword = 'value'
    KeywordEquals { value: String },
    /// keyword LIKE '%value%'
    KeywordContains { substring: String },
    /// content LIKE '%value%'
    ContentContains { substring: String },
    /// created_at > 'timestamp'
    CreatedAfter { timestamp: String },
    /// created_at < 'timestamp'
    CreatedBefore { timestamp: String },
    /// importance >= value
    ImportanceGe { value: f32 },
    /// importance <= value
    ImportanceLe { value: f32 },
}

#[derive(Debug, thiserror::Error)]
pub enum SqlError {
    #[error("Parse error: {0}")]
    Parse(String),
    #[error("Unsupported feature: {0}")]
    Unsupported(String),
}

/// Result of executing a SQL query against the engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SqlQueryResult {
    /// Matching object IDs.
    pub ids: Vec<ObjectId>,
    /// Column data (column name → list of values per row).
    pub columns: HashMap<String, Vec<String>>,
    /// Total rows.
    pub row_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_select_all() {
        let stmt = parse_sql("SELECT * FROM knowledge").unwrap();
        match stmt {
            SqlStatement::Select {
                columns,
                where_clauses,
                limit,
            } => {
                assert!(matches!(columns[0], SelectColumn::All));
                assert!(where_clauses.is_empty());
                assert!(limit.is_none());
            }
        }
    }

    #[test]
    fn test_parse_metadata_where() {
        let stmt = parse_sql("SELECT * FROM knowledge WHERE metadata.company = 'Acme'").unwrap();
        match stmt {
            SqlStatement::Select { where_clauses, .. } => {
                assert_eq!(where_clauses.len(), 1);
                match &where_clauses[0] {
                    WhereClause::MetadataEquals { key, value } => {
                        assert_eq!(key, "company");
                        assert_eq!(value, "Acme");
                    }
                    _ => panic!("wrong clause type"),
                }
            }
        }
    }

    #[test]
    fn test_parse_keyword_like() {
        let stmt = parse_sql("SELECT * FROM knowledge WHERE keyword LIKE '%enterprise%' LIMIT 10")
            .unwrap();
        match stmt {
            SqlStatement::Select {
                where_clauses,
                limit,
                ..
            } => {
                assert_eq!(limit, Some(10));
                match &where_clauses[0] {
                    WhereClause::KeywordContains { substring } => {
                        assert_eq!(substring, "enterprise");
                    }
                    _ => panic!("wrong clause type"),
                }
            }
        }
    }

    #[test]
    fn test_parse_multiple_where() {
        let stmt = parse_sql(
            "SELECT content FROM knowledge WHERE metadata.stage = 'series_a' AND importance >= 0.7 LIMIT 5",
        )
        .unwrap();
        match stmt {
            SqlStatement::Select {
                columns,
                where_clauses,
                limit,
            } => {
                assert!(matches!(columns[0], SelectColumn::Named(ref c) if c == "content"));
                assert_eq!(where_clauses.len(), 2);
                assert_eq!(limit, Some(5));
            }
        }
    }
}
