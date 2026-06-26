use tracing::info;

use super::intent::{DetectedIntent, IntentAnalyzer};
use super::plan::{plan_from_actions, ExecutionPlan};
use super::rule_engine::RuleEngine;

/// The AI Query Planner — the core intelligence of KowitoDB.
///
/// This is the "moat" component. It takes a natural-language question
/// and produces an optimized execution plan that determines:
/// - Which indexes to query
/// - Which retrieval strategy to use
/// - How to merge and rerank results
/// - How to minimize cost
///
/// In Phase 1, this uses a rule engine. In later phases, it evolves
/// into a learned optimizer.
pub struct QueryPlanner {
    intent_analyzer: IntentAnalyzer,
    rule_engine: RuleEngine,
}

impl QueryPlanner {
    pub fn new() -> Self {
        info!("Initializing AI Query Planner");
        Self {
            intent_analyzer: IntentAnalyzer::new(),
            rule_engine: RuleEngine::new(),
        }
    }

    /// Analyze a user question and produce both the detected intent
    /// and the execution plan.
    pub fn plan(&self, question: &str) -> (DetectedIntent, ExecutionPlan) {
        // Step 1: Understand intent
        let intent = self.intent_analyzer.analyze(question);

        info!(
            "Intent detected: {:?} (confidence: {:.2})",
            intent.intent, intent.confidence
        );

        // Step 2: Apply rule engine to determine retrieval actions
        let actions = self.rule_engine.evaluate(&intent);

        // Step 3: Build execution plan
        let plan = plan_from_actions(
            &intent.question,
            &actions,
            &intent.entities.keywords,
            &intent.entities.metadata_filters,
        );

        info!(
            "Generated plan with {} steps (estimated cost: {:.2})",
            plan.steps.len(),
            plan.estimated_cost
        );

        (intent, plan)
    }

    /// Return a reference to the intent analyzer (for standalone intent
    /// analysis without full planning).
    pub fn analyze_intent(&self, question: &str) -> DetectedIntent {
        self.intent_analyzer.analyze(question)
    }
}

impl Default for QueryPlanner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plan_comparison_query() {
        let planner = QueryPlanner::new();
        let (_intent, plan) = planner.plan("Compare OpenAI and Anthropic");
        assert!(!plan.steps.is_empty());
        // Should include merge and context builder steps
        let step_types: Vec<_> = plan.steps.iter().map(|s| &s.step_type).collect();
        assert!(step_types.contains(&&crate::PlanStepType::Merge));
        assert!(step_types.contains(&&crate::PlanStepType::BuildContext));
    }

    #[test]
    fn test_plan_temporal_query() {
        let planner = QueryPlanner::new();
        let (_intent, plan) = planner.plan("Which customers renewed after January 2024?");
        let step_types: Vec<_> = plan.steps.iter().map(|s| &s.step_type).collect();
        assert!(step_types.contains(&&crate::PlanStepType::TimeFilter));
    }
}
