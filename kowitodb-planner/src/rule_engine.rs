use tracing::debug;

use super::intent::{DetectedIntent, Intent};

/// Action recommended by the rule engine for a specific retrieval step.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RetrievalAction {
    /// Search the vector index.
    VectorSearch,
    /// Search the full-text index.
    KeywordSearch,
    /// Traverse the knowledge graph.
    GraphTraverse,
    /// Apply metadata filters.
    MetadataFilter,
    /// Apply time-range filter.
    TimeFilter,
    /// Use code-specific embeddings.
    CodeSearch,
}

/// A rule matches a detected intent and produces a set of retrieval actions.
#[derive(Debug, Clone)]
struct Rule {
    name: &'static str,
    /// Predicate: does this rule fire for this intent?
    predicate: fn(&DetectedIntent) -> bool,
    /// Actions to execute when the rule fires.
    actions: Vec<RetrievalAction>,
}

/// Phase 1 rule engine for retrieval optimization.
///
/// This is a simple rule-based system as described in the architecture.
/// In later phases, this evolves into a learned optimizer that chooses
/// retrieval strategies based on historical query performance.
///
/// Rules follow the pattern from the idea document:
/// - IF contains date → Time Index
/// - IF contains "compare" → Dual retrieval
/// - IF contains company names → Entity graph
/// - IF contains source code → Code embedding
pub struct RuleEngine {
    rules: Vec<Rule>,
}

impl RuleEngine {
    pub fn new() -> Self {
        let rules = vec![
            Rule {
                name: "temporal",
                predicate: |di: &DetectedIntent| {
                    di.intent == Intent::Temporal || !di.entities.dates.is_empty()
                },
                actions: vec![RetrievalAction::TimeFilter, RetrievalAction::VectorSearch],
            },
            Rule {
                name: "comparison",
                predicate: |di: &DetectedIntent| {
                    di.intent == Intent::Comparison || di.entities.is_comparison
                },
                actions: vec![
                    RetrievalAction::VectorSearch,
                    RetrievalAction::KeywordSearch,
                    RetrievalAction::GraphTraverse,
                ],
            },
            Rule {
                name: "entity",
                predicate: |di: &DetectedIntent| {
                    di.intent == Intent::EntitySearch || !di.entities.named.is_empty()
                },
                actions: vec![
                    RetrievalAction::VectorSearch,
                    RetrievalAction::KeywordSearch,
                    RetrievalAction::GraphTraverse,
                    RetrievalAction::MetadataFilter,
                ],
            },
            Rule {
                name: "code",
                predicate: |di: &DetectedIntent| {
                    di.intent == Intent::CodeSearch || di.entities.is_code
                },
                actions: vec![RetrievalAction::CodeSearch, RetrievalAction::KeywordSearch],
            },
            Rule {
                name: "listing",
                predicate: |di: &DetectedIntent| di.intent == Intent::Listing,
                actions: vec![
                    RetrievalAction::VectorSearch,
                    RetrievalAction::KeywordSearch,
                ],
            },
            Rule {
                // Aggregational queries want broad structured recall (scan the
                // metadata/keyword space) rather than a tight top-k semantic hit.
                name: "analytical",
                predicate: |di: &DetectedIntent| di.intent == Intent::Analytical,
                actions: vec![
                    RetrievalAction::MetadataFilter,
                    RetrievalAction::KeywordSearch,
                    RetrievalAction::VectorSearch,
                ],
            },
            Rule {
                name: "factoid",
                predicate: |di: &DetectedIntent| di.intent == Intent::Factoid,
                actions: vec![
                    RetrievalAction::VectorSearch,
                    RetrievalAction::KeywordSearch,
                ],
            },
            // Default fallback: always do vector + keyword search
            Rule {
                name: "default",
                predicate: |_| true,
                actions: vec![
                    RetrievalAction::VectorSearch,
                    RetrievalAction::KeywordSearch,
                ],
            },
        ];

        Self { rules }
    }

    /// Evaluate all rules against a detected intent and return the
    /// ordered list of retrieval actions to execute.
    ///
    /// Multiple rules may fire; actions are deduplicated while preserving
    /// the order of first appearance.
    pub fn evaluate(&self, intent: &DetectedIntent) -> Vec<RetrievalAction> {
        let mut actions = Vec::new();
        let mut seen = std::collections::HashSet::new();

        for rule in &self.rules {
            if (rule.predicate)(intent) {
                debug!("Rule '{}' fired for intent {:?}", rule.name, intent.intent);
                for action in &rule.actions {
                    if seen.insert(action.clone()) {
                        actions.push(action.clone());
                    }
                }
            }
        }

        // Ensure we have at least one search action
        if actions.is_empty() {
            actions.push(RetrievalAction::VectorSearch);
        }

        debug!("Rule engine produced {} actions", actions.len());
        actions
    }
}

impl Default for RuleEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intent::IntentAnalyzer;

    #[test]
    fn test_temporal_rule() {
        let engine = RuleEngine::new();
        let analyzer = IntentAnalyzer::new();
        let intent = analyzer.analyze("after January 2024");
        let actions = engine.evaluate(&intent);
        assert!(actions.contains(&RetrievalAction::TimeFilter));
    }

    #[test]
    fn test_comparison_rule() {
        let engine = RuleEngine::new();
        let analyzer = IntentAnalyzer::new();
        let intent = analyzer.analyze("Compare OpenAI and Anthropic");
        let actions = engine.evaluate(&intent);
        assert!(actions.contains(&RetrievalAction::GraphTraverse));
    }

    #[test]
    fn test_analytical_rule() {
        let engine = RuleEngine::new();
        let analyzer = IntentAnalyzer::new();
        let intent = analyzer.analyze("How many deals closed last quarter?");
        let actions = engine.evaluate(&intent);
        assert!(actions.contains(&RetrievalAction::MetadataFilter));
    }

    #[test]
    fn test_default_rule_always_fires() {
        let engine = RuleEngine::new();
        let analyzer = IntentAnalyzer::new();
        let intent = analyzer.analyze("hello world");
        let actions = engine.evaluate(&intent);
        assert!(actions.len() >= 2);
    }
}
