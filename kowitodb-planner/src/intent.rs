use serde::{Deserialize, Serialize};

/// Classified intent of a natural-language query.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Intent {
    /// "Who", "What is", "Tell me about" — factual lookup.
    Factoid,
    /// "Compare X and Y", "difference between"
    Comparison,
    /// "List all", "Show me", "Find" — enumeration.
    Listing,
    /// "After X", "Before Y", "During Z" — time-bounded.
    Temporal,
    /// "Why", "How does" — explanatory.
    Explanation,
    /// "Summarize", "TLDR" — compression.
    Summary,
    /// "Which companies raised funding" — entity-driven.
    EntitySearch,
    /// "Code for", "Implement" — code-related.
    CodeSearch,
    /// "How many", "Count", "Average", "Total number of" — aggregational /
    /// structured analytics (favors broad structured recall over top-k semantic).
    Analytical,
    /// Default fallback.
    General,
}

/// Entities extracted from a query.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Entities {
    /// Named entities: company names, person names, product names.
    pub named: Vec<String>,
    /// Date references detected in the query.
    pub dates: Vec<String>,
    /// Keywords extracted for full-text search.
    pub keywords: Vec<String>,
    /// Metadata filters implied by the query (e.g., "source:web").
    pub metadata_filters: Vec<(String, String)>,
    /// Whether the query implies comparison between multiple items.
    pub is_comparison: bool,
    /// Whether the query references source code.
    pub is_code: bool,
}

/// Result of the intent analysis step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedIntent {
    pub intent: Intent,
    pub entities: Entities,
    /// Raw parsed question (cleaned).
    pub question: String,
    /// Confidence in the intent classification (0.0 - 1.0).
    pub confidence: f32,
}

/// Analyzes a natural-language query to detect intent and extract entities.
///
/// This is a rule-based implementation for Phase 1. In later phases, this
/// can be replaced with a learned classifier (small LM or fine-tuned model).
pub struct IntentAnalyzer;

impl IntentAnalyzer {
    pub fn new() -> Self {
        Self
    }

    /// Analyze a user question and produce a `DetectedIntent`.
    pub fn analyze(&self, question: &str) -> DetectedIntent {
        let cleaned = question.trim().to_string();
        let lower = cleaned.to_lowercase();

        // Detect intent via keyword rules
        let intent = self.classify_intent(&lower);

        // Extract entities
        let entities = self.extract_entities(&cleaned);

        // Compute rough confidence based on how many signals matched
        let confidence = self.compute_confidence(&intent, &entities);

        DetectedIntent {
            intent,
            entities,
            question: cleaned,
            confidence,
        }
    }

    fn classify_intent(&self, lower: &str) -> Intent {
        // Rule-based classification: check strongest signals first
        if has_term(lower, "compare")
            || has_term(lower, "difference")
            || has_term(lower, "versus")
            || has_term(lower, " vs ")
        {
            return Intent::Comparison;
        }
        if has_term(lower, "after")
            || has_term(lower, "before")
            || has_term(lower, "during")
            || has_term(lower, "since")
            || has_term(lower, "until")
            || has_term(lower, "between")
        {
            // Check if it's truly temporal vs just using the word casually
            if has_term(lower, "january")
                || has_term(lower, "february")
                || has_term(lower, "march")
                || has_year(lower)
                || has_term(lower, "q1")
                || has_term(lower, "q2")
                || has_term(lower, "q3")
                || has_term(lower, "q4")
                || has_term(lower, "series a")
                || has_term(lower, "series b")
                || has_term(lower, "series c")
            {
                return Intent::Temporal;
            }
        }
        if has_term(lower, "summarize")
            || has_term(lower, "tldr")
            || has_term(lower, "brief")
            || has_term(lower, "short")
        {
            return Intent::Summary;
        }
        // Aggregational / analytical queries — checked before Explanation so
        // "how many" doesn't fall through to "how"-style explanatory matching.
        if has_term(lower, "how many")
            || has_term(lower, "how much")
            || has_term(lower, "number of")
            || has_term(lower, "count of")
            || has_term(lower, "count the")
            || has_term(lower, "total number")
            || has_term(lower, "average")
            || has_term(lower, " avg ")
            || has_term(lower, "sum of")
            || has_term(lower, "most common")
            || has_term(lower, "least common")
            || has_term(lower, "percentage of")
            || has_term(lower, "what fraction")
            || has_term(lower, "group by")
        {
            return Intent::Analytical;
        }
        if has_term(lower, "why") || has_term(lower, "how does") || has_term(lower, "explain") {
            return Intent::Explanation;
        }
        if lower.starts_with("list")
            || has_term(lower, "show me")
            || has_term(lower, "find all")
            || has_term(lower, "all ")
            || has_term(lower, "every ")
        {
            return Intent::Listing;
        }
        if has_term(lower, "code")
            || has_term(lower, "function")
            || has_term(lower, "implement")
            || has_term(lower, "source")
            || has_term(lower, "bug")
            || has_term(lower, "error")
        {
            return Intent::CodeSearch;
        }
        if has_term(lower, "company")
            || has_term(lower, "companies")
            || has_term(lower, "startup")
            || has_term(lower, "funding")
            || has_term(lower, "raised")
            || has_term(lower, "founder")
            || has_term(lower, "investor")
            || has_term(lower, "invested")
            || has_term(lower, "enterprise")
            || has_term(lower, "customer")
        {
            return Intent::EntitySearch;
        }
        if lower.starts_with("who")
            || lower.starts_with("what")
            || lower.starts_with("where")
            || lower.starts_with("when")
        {
            return Intent::Factoid;
        }

        Intent::General
    }

    fn extract_entities(&self, text: &str) -> Entities {
        let lower = text.to_lowercase();
        let mut entities = Entities::default();

        // Extract potential named entities (capitalized words, not at sentence start)
        let words: Vec<&str> = text.split_whitespace().collect();
        for (i, word) in words.iter().enumerate() {
            let clean: String = word.chars().filter(|c| c.is_alphanumeric()).collect();
            if clean.len() > 1 && clean.chars().next().is_some_and(|c| c.is_uppercase()) && i > 0
            // skip first word (could be sentence start)
            {
                entities.named.push(clean);
            }
        }

        // Date detection
        let date_patterns = [
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
            "q1",
            "q2",
            "q3",
            "q4",
            "series a",
            "series b",
            "series c",
        ];
        for pat in &date_patterns {
            // "may" is far more often the verb than the month; only count it
            // next to a number ("May 2024", "3 May").
            if *pat == "may" && !may_is_month(&lower) {
                continue;
            }
            if has_term(&lower, pat) {
                entities.dates.push(pat.to_string());
            }
        }
        for year in years_in(&lower) {
            if !entities.dates.contains(&year) {
                entities.dates.push(year);
            }
        }

        // Comparison detection
        entities.is_comparison = has_term(&lower, "compare")
            || has_term(&lower, "versus")
            || has_term(&lower, " vs ")
            || has_term(&lower, "difference");

        // Code detection
        entities.is_code = has_term(&lower, "code")
            || has_term(&lower, "function")
            || has_term(&lower, "implement")
            || has_term(&lower, "source");

        // Extract generic keywords (remove stopwords)
        let stopwords: &[&str] = &[
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has",
            "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "can",
            "shall", "to", "of", "in", "for", "on", "with", "at", "by", "from", "as", "into",
            "through", "during", "before", "after", "about", "what", "which", "who", "whom",
            "whose", "where", "when", "why", "how", "tell", "show", "find", "list", "give", "and",
            "or", "not", "but", "if", "then", "else", "this", "that", "these", "those", "it",
            "its",
        ];

        entities.keywords = words
            .iter()
            .map(|w| {
                w.chars()
                    .filter(|c| c.is_alphanumeric())
                    .collect::<String>()
                    .to_lowercase()
            })
            .filter(|w| w.len() > 1 && !stopwords.contains(&w.as_str()))
            .collect();

        entities
    }

    fn compute_confidence(&self, intent: &Intent, entities: &Entities) -> f32 {
        let mut score: f32 = 0.5; // base

        if intent != &Intent::General {
            score += 0.2;
        }

        if !entities.named.is_empty() {
            score += 0.1;
        }
        if !entities.dates.is_empty() {
            score += 0.1;
        }
        if entities.keywords.len() > 2 {
            score += 0.1;
        }

        score.min(1.0)
    }
}

impl Default for IntentAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Whether `term` occurs in `text` as a whole word or phrase (a trailing
/// plural "s"/"es" is allowed), so "code" matches "code"/"codes" but not
/// "decode", and "all " doesn't match "small ". Edges of `term` that are not
/// alphanumeric (e.g. the spaces in `" vs "`) need no boundary.
fn has_term(text: &str, term: &str) -> bool {
    let is_word = |c: char| c.is_alphanumeric();
    let need_left = term.chars().next().is_some_and(is_word);
    let need_right = term.chars().last().is_some_and(is_word);
    let mut start = 0;
    while let Some(pos) = text[start..].find(term) {
        let at = start + pos;
        let end = at + term.len();
        let left_ok = !need_left || !text[..at].chars().next_back().is_some_and(is_word);
        let right_ok = !need_right || {
            let rest = &text[end..];
            let rest = rest
                .strip_prefix("es")
                .filter(|r| !r.chars().next().is_some_and(is_word))
                .or_else(|| rest.strip_prefix('s'))
                .unwrap_or(rest);
            !rest.chars().next().is_some_and(is_word)
        };
        if left_ok && right_ok {
            return true;
        }
        start = at + term.chars().next().map_or(1, char::len_utf8);
    }
    false
}

/// Four-digit years (1900–2100) mentioned in `text`.
fn years_in(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() == 4 && w.chars().all(|c| c.is_ascii_digit()))
        .filter(|w| w.parse::<u32>().is_ok_and(|y| (1900..=2100).contains(&y)))
        .map(str::to_string)
        .collect()
}

fn has_year(text: &str) -> bool {
    !years_in(text).is_empty()
}

/// "may" used as a month: directly next to a number ("may 2024", "3 may").
fn may_is_month(text: &str) -> bool {
    let words: Vec<&str> = text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect();
    let is_num = |w: &&str| w.chars().all(|c| c.is_ascii_digit());
    words.iter().enumerate().any(|(i, w)| {
        *w == "may" && (words.get(i + 1).is_some_and(is_num) || (i > 0 && is_num(&words[i - 1])))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_comparison() {
        let analyzer = IntentAnalyzer::new();
        let result = analyzer.analyze("Compare OpenAI with Anthropic");
        assert_eq!(result.intent, Intent::Comparison);
        assert!(result.entities.is_comparison);
    }

    #[test]
    fn test_classify_temporal() {
        let analyzer = IntentAnalyzer::new();
        let result = analyzer.analyze("Which enterprise customers renewed after January 2024?");
        assert_eq!(result.intent, Intent::Temporal);
        assert!(!result.entities.dates.is_empty());
    }

    #[test]
    fn test_classify_analytical() {
        let analyzer = IntentAnalyzer::new();
        assert_eq!(
            analyzer.analyze("How many customers churned?").intent,
            Intent::Analytical
        );
        assert_eq!(
            analyzer.analyze("What is the average deal size?").intent,
            Intent::Analytical
        );
        // "how does" must remain Explanation, not be captured by the analytical rule.
        assert_eq!(
            analyzer.analyze("How does replication work?").intent,
            Intent::Explanation
        );
    }

    #[test]
    fn test_classify_entity_search() {
        let analyzer = IntentAnalyzer::new();
        let result = analyzer.analyze("Which companies raised Series A funding?");
        assert_eq!(result.intent, Intent::EntitySearch);
    }

    #[test]
    fn test_terms_match_whole_words_only() {
        let analyzer = IntentAnalyzer::new();
        // "small " used to match the "all " listing rule.
        assert_ne!(
            analyzer
                .analyze("Which small startups raised funding?")
                .intent,
            Intent::Listing
        );
        // "shortage" used to match "short" → Summary.
        assert_ne!(
            analyzer.analyze("What caused the chip shortage?").intent,
            Intent::Summary
        );
        // "resources"/"decode" used to match "source"/"code" → CodeSearch.
        assert_ne!(
            analyzer
                .analyze("Where are the onboarding resources?")
                .intent,
            Intent::CodeSearch
        );
        assert_eq!(
            analyzer.analyze("Find the bugs in this function").intent,
            Intent::CodeSearch
        );
    }

    #[test]
    fn test_date_detection() {
        let analyzer = IntentAnalyzer::new();
        assert!(analyzer
            .analyze("What did the mayor say?")
            .entities
            .dates
            .is_empty());
        assert!(analyzer
            .analyze("May I ask a question?")
            .entities
            .dates
            .is_empty());
        let dates = analyzer
            .analyze("What happened in May 2019?")
            .entities
            .dates;
        assert!(dates.contains(&"may".to_string()) && dates.contains(&"2019".to_string()));
        assert_eq!(
            analyzer.analyze("Deals closed after April 2019").intent,
            Intent::Temporal
        );
    }
}
