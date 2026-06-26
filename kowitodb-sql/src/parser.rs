//! Lightweight SQL parser for the KowitoDB SQL bridge.

use crate::{SelectColumn, SqlError, SqlStatement, WhereClause};

/// Parse a SQL string into a `SqlStatement`.
pub fn parse(sql: &str) -> Result<SqlStatement, SqlError> {
    let sql = sql.trim();
    let upper = sql.to_uppercase();

    if !upper.starts_with("SELECT") {
        return Err(SqlError::Parse("Query must start with SELECT".into()));
    }

    let (columns_str, rest) = split_keyword(&sql[6..], "FROM")?;
    let columns = parse_columns(columns_str.trim())?;
    let (table_rest, where_str, limit) = parse_rest(rest.trim())?;

    let table_name = table_rest.split_whitespace().next().unwrap_or("");
    if !table_name.eq_ignore_ascii_case("knowledge") && !table_name.eq_ignore_ascii_case("objects")
    {
        return Err(SqlError::Parse(format!(
            "Unknown table '{}'. Use 'knowledge' or 'objects'.",
            table_name
        )));
    }

    let where_clauses = if let Some(w) = where_str {
        parse_where_clauses(w.trim())?
    } else {
        Vec::new()
    };

    Ok(SqlStatement::Select {
        columns,
        where_clauses,
        limit,
    })
}

fn split_keyword<'a>(s: &'a str, keyword: &str) -> Result<(&'a str, &'a str), SqlError> {
    let upper = s.to_uppercase();
    let kw_upper = keyword.to_uppercase();
    let mut in_quote = false;
    let mut i = 0;
    let bytes = s.as_bytes();

    while i < bytes.len() {
        if bytes[i] == b'\'' {
            in_quote = !in_quote;
            i += 1;
            continue;
        }
        if !in_quote && i + keyword.len() <= upper.len() && upper[i..].starts_with(&kw_upper) {
            let after = i + keyword.len();
            let is_boundary =
                after >= bytes.len() || bytes[after].is_ascii_whitespace() || bytes[after] == b';';
            let before_ok = i == 0 || bytes[i - 1].is_ascii_whitespace();
            if is_boundary && before_ok {
                return Ok((s[..i].trim(), s[after..].trim()));
            }
        }
        i += 1;
    }

    Err(SqlError::Parse(format!("Expected keyword '{}'", keyword)))
}

fn parse_columns(s: &str) -> Result<Vec<SelectColumn>, SqlError> {
    if s == "*" {
        return Ok(vec![SelectColumn::All]);
    }
    let cols: Vec<SelectColumn> = s
        .split(',')
        .map(|c| {
            let col = c.trim().trim_matches('"').trim_matches('\'');
            SelectColumn::Named(col.to_string())
        })
        .collect();
    if cols.is_empty() {
        return Err(SqlError::Parse("No columns specified".into()));
    }
    Ok(cols)
}

fn parse_rest(s: &str) -> Result<(&str, Option<&str>, Option<usize>), SqlError> {
    let where_idx = find_keyword_boundary(s, "WHERE");
    let limit_idx = find_keyword_boundary(s, "LIMIT");
    let table_end = where_idx.unwrap_or(limit_idx.unwrap_or(s.len()));

    let where_str = if let Some(idx) = where_idx {
        let where_start = idx + 5;
        let where_end = limit_idx.unwrap_or(s.len());
        Some(&s[where_start..where_end])
    } else {
        None
    };

    let limit = if let Some(idx) = limit_idx {
        let limit_val = s[idx + 5..].trim();
        let limit_val = limit_val.split_whitespace().next().unwrap_or("");
        let limit_val = limit_val.trim_end_matches(';');
        Some(
            limit_val
                .parse::<usize>()
                .map_err(|_| SqlError::Parse(format!("Invalid LIMIT: {}", limit_val)))?,
        )
    } else {
        None
    };

    Ok((&s[..table_end], where_str, limit))
}

fn find_keyword_boundary(s: &str, keyword: &str) -> Option<usize> {
    let upper = s.to_uppercase();
    let kw_upper = keyword.to_uppercase();
    let mut in_quote = false;
    let bytes = s.as_bytes();

    for i in 0..bytes.len() {
        if bytes[i] == b'\'' {
            in_quote = !in_quote;
            continue;
        }
        if in_quote {
            continue;
        }
        if i + keyword.len() <= upper.len() && upper[i..].starts_with(&kw_upper) {
            let after = i + keyword.len();
            let is_boundary =
                after >= bytes.len() || bytes[after].is_ascii_whitespace() || bytes[after] == b';';
            let before_ok = i == 0 || bytes[i - 1].is_ascii_whitespace();
            if is_boundary && before_ok {
                return Some(i);
            }
        }
    }
    None
}

fn parse_where_clauses(s: &str) -> Result<Vec<WhereClause>, SqlError> {
    let parts = split_on_and(s);
    let mut clauses = Vec::new();
    for part in parts {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        clauses.push(parse_single_where(part)?);
    }
    Ok(clauses)
}

fn split_on_and(s: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut in_quote = false;
    let mut last = 0;
    let bytes = s.as_bytes();
    let upper = s.to_uppercase();

    for i in 0..bytes.len() {
        if bytes[i] == b'\'' {
            in_quote = !in_quote;
            continue;
        }
        if in_quote {
            continue;
        }
        if i + 3 <= upper.len() && &upper[i..i + 3] == "AND" {
            let after = i + 3;
            let is_boundary = after >= bytes.len() || bytes[after].is_ascii_whitespace();
            let before_ok = i == 0 || bytes[i - 1].is_ascii_whitespace();
            if is_boundary && before_ok {
                parts.push(&s[last..i]);
                last = after;
            }
        }
    }

    if last < s.len() {
        parts.push(&s[last..]);
    }
    parts
}

fn parse_single_where(s: &str) -> Result<WhereClause, SqlError> {
    let s = s.trim();
    let upper = s.to_uppercase();

    if upper.starts_with("METADATA.") {
        return parse_metadata_clause(&s[9..]);
    }
    if upper.starts_with("KEYWORD") {
        return parse_keyword_clause(s[7..].trim());
    }
    if upper.starts_with("CONTENT") {
        return parse_content_clause(s[7..].trim());
    }
    if upper.starts_with("IMPORTANCE") {
        return parse_importance_clause(s[10..].trim());
    }
    if upper.starts_with("CREATED_AT") {
        return parse_created_clause(s[10..].trim());
    }

    Err(SqlError::Parse(format!("Unsupported WHERE: {}", s)))
}

fn parse_metadata_clause(s: &str) -> Result<WhereClause, SqlError> {
    if let Some(idx) = find_operator(s, "=") {
        let key = s[..idx].trim().to_string();
        let value = extract_string_value(&s[idx + 1..])?;
        return Ok(WhereClause::MetadataEquals { key, value });
    }
    if let Some(idx) = find_operator(s, "LIKE") {
        let key = s[..idx].trim().to_string();
        let raw = s[idx + 4..].trim();
        let value = extract_string_value(raw)?;
        let clean = value.trim_matches('%').to_string();
        return Ok(WhereClause::MetadataContains {
            key,
            substring: clean,
        });
    }
    Err(SqlError::Parse(format!("Invalid metadata clause: {}", s)))
}

fn parse_keyword_clause(s: &str) -> Result<WhereClause, SqlError> {
    if let Some(idx) = find_operator(s, "=") {
        let value = extract_string_value(&s[idx + 1..])?;
        return Ok(WhereClause::KeywordEquals { value });
    }
    if let Some(idx) = find_operator(s, "LIKE") {
        let raw = s[idx + 4..].trim();
        let value = extract_string_value(raw)?;
        let clean = value.trim_matches('%').to_string();
        return Ok(WhereClause::KeywordContains { substring: clean });
    }
    Err(SqlError::Parse(format!("Invalid keyword clause: {}", s)))
}

fn parse_content_clause(s: &str) -> Result<WhereClause, SqlError> {
    if let Some(idx) = find_operator(s, "LIKE") {
        let raw = s[idx + 4..].trim();
        let value = extract_string_value(raw)?;
        let clean = value.trim_matches('%').to_string();
        return Ok(WhereClause::ContentContains { substring: clean });
    }
    Err(SqlError::Parse(format!("Invalid content clause: {}", s)))
}

fn parse_importance_clause(s: &str) -> Result<WhereClause, SqlError> {
    if s.starts_with(">=") {
        let val = s[2..]
            .trim()
            .parse::<f32>()
            .map_err(|_| SqlError::Parse("Invalid importance".into()))?;
        return Ok(WhereClause::ImportanceGe { value: val });
    }
    if s.starts_with("<=") {
        let val = s[2..]
            .trim()
            .parse::<f32>()
            .map_err(|_| SqlError::Parse("Invalid importance".into()))?;
        return Ok(WhereClause::ImportanceLe { value: val });
    }
    if s.starts_with('>') {
        let val = s[1..]
            .trim()
            .parse::<f32>()
            .map_err(|_| SqlError::Parse("Invalid importance".into()))?;
        return Ok(WhereClause::ImportanceGe { value: val });
    }
    if s.starts_with('<') {
        let val = s[1..]
            .trim()
            .parse::<f32>()
            .map_err(|_| SqlError::Parse("Invalid importance".into()))?;
        return Ok(WhereClause::ImportanceLe { value: val });
    }
    Err(SqlError::Parse(format!("Invalid importance clause: {}", s)))
}

fn parse_created_clause(s: &str) -> Result<WhereClause, SqlError> {
    if s.starts_with('>') {
        let value = extract_string_value(&s[1..])?;
        return Ok(WhereClause::CreatedAfter { timestamp: value });
    }
    if s.starts_with('<') {
        let value = extract_string_value(&s[1..])?;
        return Ok(WhereClause::CreatedBefore { timestamp: value });
    }
    Err(SqlError::Parse(format!("Invalid created_at clause: {}", s)))
}

fn find_operator(s: &str, op: &str) -> Option<usize> {
    let upper = s.to_uppercase();
    let op_upper = op.to_uppercase();
    let mut in_quote = false;
    let bytes = s.as_bytes();

    for i in 0..bytes.len() {
        if bytes[i] == b'\'' {
            in_quote = !in_quote;
            continue;
        }
        if in_quote {
            continue;
        }
        if i + op.len() <= upper.len() && upper[i..].starts_with(&op_upper) {
            return Some(i);
        }
    }
    None
}

fn extract_string_value(s: &str) -> Result<String, SqlError> {
    let s = s.trim();
    if s.starts_with('\'') {
        if let Some(end) = s[1..].find('\'') {
            return Ok(s[1..end + 1].to_string());
        }
        return Ok(s[1..].trim_end_matches('\'').to_string());
    }
    Ok(s.split_whitespace().next().unwrap_or("").to_string())
}
