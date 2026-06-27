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
        if lower.contains("compare")
            || lower.contains("difference")
            || lower.contains("versus")
            || lower.contains(" vs ")
        {
            return Intent::Comparison;
        }
        if lower.contains("after")
            || lower.contains("before")
            || lower.contains("during")
            || lower.contains("since")
            || lower.contains("until")
            || lower.contains("between")
        {
            // Check if it's truly temporal vs just using the word casually
            if lower.contains("january")
                || lower.contains("february")
                || lower.contains("march")
                || lower.contains("202")
                || lower.contains("q1")
                || lower.contains("q2")
                || lower.contains("q3")
                || lower.contains("q4")
                || lower.contains("series a")
                || lower.contains("series b")
                || lower.contains("series c")
            {
                return Intent::Temporal;
            }
        }
        if lower.contains("summarize")
            || lower.contains("tldr")
            || lower.contains("brief")
            || lower.contains("short")
        {
            return Intent::Summary;
        }
        // Aggregational / analytical queries — checked before Explanation so
        // "how many" doesn't fall through to "how"-style explanatory matching.
        if lower.contains("how many")
            || lower.contains("how much")
            || lower.contains("number of")
            || lower.contains("count of")
            || lower.contains("count the")
            || lower.contains("total number")
            || lower.contains("average")
            || lower.contains(" avg ")
            || lower.contains("sum of")
            || lower.contains("most common")
            || lower.contains("least common")
            || lower.contains("percentage of")
            || lower.contains("what fraction")
            || lower.contains("group by")
        {
            return Intent::Analytical;
        }
        if lower.contains("why") || lower.contains("how does") || lower.contains("explain") {
            return Intent::Explanation;
        }
        if lower.starts_with("list")
            || lower.contains("show me")
            || lower.contains("find all")
            || lower.contains("all ")
            || lower.contains("every ")
        {
            return Intent::Listing;
        }
        if lower.contains("code")
            || lower.contains("function")
            || lower.contains("implement")
            || lower.contains("source")
            || lower.contains("bug")
            || lower.contains("error")
        {
            return Intent::CodeSearch;
        }
        if lower.contains("company")
            || lower.contains("companies")
            || lower.contains("startup")
            || lower.contains("funding")
            || lower.contains("raised")
            || lower.contains("founder")
            || lower.contains("investor")
            || lower.contains("invested")
            || lower.contains("enterprise")
            || lower.contains("customer")
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
            "2020",
            "2021",
            "2022",
            "2023",
            "2024",
            "2025",
            "2026",
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
            if lower.contains(pat) {
                entities.dates.push(pat.to_string());
            }
        }

        // Comparison detection
        entities.is_comparison = lower.contains("compare")
            || lower.contains("versus")
            || lower.contains(" vs ")
            || lower.contains("difference");

        // Code detection
        entities.is_code = lower.contains("code")
            || lower.contains("function")
            || lower.contains("implement")
            || lower.contains("source");

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
}
