use serde::{Deserialize, Serialize};

use super::rule_engine::RetrievalAction;

/// A step in the query execution plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanStep {
    /// What kind of step this is.
    pub step_type: PlanStepType,
    /// Index to use (if applicable).
    pub index: Option<String>,
    /// Query string for this step.
    pub query: Option<String>,
    /// Number of results to return.
    pub limit: Option<usize>,
    /// Parameters for the step (metadata filters, time ranges, etc.).
    pub params: std::collections::HashMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlanStepType {
    /// Intent detection (already done, but recorded for trace).
    IntentDetection,
    /// Vector nearest-neighbor search.
    VectorSearch,
    /// Full-text keyword search.
    KeywordSearch,
    /// Graph traversal.
    GraphTraverse,
    /// Metadata filtering.
    MetadataFilter,
    /// Time-range filtering.
    TimeFilter,
    /// Result merging / fusion.
    Merge,
    /// Reranking of merged results.
    Rerank,
    /// Context building (assemble final context for LLM).
    BuildContext,
}

/// An execution plan for a single query.
///
/// This is analogous to a SQL query plan: a sequence of steps
/// the execution engine will perform to answer an `ai.ask()` call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionPlan {
    /// The question being answered.
    pub question: String,
    /// Ordered list of plan steps.
    pub steps: Vec<PlanStep>,
    /// Total estimated cost (arbitrary units, for cost optimization).
    pub estimated_cost: f32,
    /// Whether this plan uses cached results.
    pub uses_cache: bool,
}

impl ExecutionPlan {
    /// Create a new plan for a question.
    pub fn new(question: impl Into<String>) -> Self {
        Self {
            question: question.into(),
            steps: Vec::new(),
            estimated_cost: 0.0,
            uses_cache: false,
        }
    }

    /// Add a step to the plan.
    pub fn add_step(&mut self, step: PlanStep) {
        self.steps.push(step);
    }

    /// Print the plan in a human-readable format.
    pub fn explain(&self) -> String {
        let mut out = format!("Query Plan for: \"{}\"\n", self.question);
        out.push_str(&format!("Estimated cost: {:.2}\n", self.estimated_cost));
        out.push_str("Steps:\n");
        for (i, step) in self.steps.iter().enumerate() {
            out.push_str(&format!("  {}. {:?}", i + 1, step.step_type));
            if let Some(ref idx) = step.index {
                out.push_str(&format!(" [index={}]", idx));
            }
            if let Some(ref q) = step.query {
                out.push_str(&format!(" \"{}\"", q));
            }
            if let Some(limit) = step.limit {
                out.push_str(&format!(" limit={}", limit));
            }
            out.push('\n');
        }
        out
    }
}

/// Build an execution plan from a set of retrieval actions.
pub fn plan_from_actions(
    question: &str,
    actions: &[RetrievalAction],
    keywords: &[String],
    metadata_filters: &[(String, String)],
) -> ExecutionPlan {
    let mut plan = ExecutionPlan::new(question.to_string());
    let mut cost = 0.0;

    // Step 1: Intent detection (already done externally, but recorded)
    plan.add_step(PlanStep {
        step_type: PlanStepType::IntentDetection,
        index: None,
        query: Some(question.to_string()),
        limit: None,
        params: Default::default(),
    });

    for action in actions {
        match action {
            RetrievalAction::VectorSearch => {
                plan.add_step(PlanStep {
                    step_type: PlanStepType::VectorSearch,
                    index: Some("hnsw".into()),
                    query: Some(question.to_string()),
                    limit: Some(20),
                    params: Default::default(),
                });
                cost += 1.0;
            }
            RetrievalAction::KeywordSearch => {
                let query_str = keywords.join(" ");
                plan.add_step(PlanStep {
                    step_type: PlanStepType::KeywordSearch,
                    index: Some("tantivy".into()),
                    query: Some(query_str.clone()),
                    limit: Some(20),
                    params: Default::default(),
                });
                cost += 0.5;
            }
            RetrievalAction::GraphTraverse => {
                plan.add_step(PlanStep {
                    step_type: PlanStepType::GraphTraverse,
                    index: Some("graph".into()),
                    query: Some(question.to_string()),
                    limit: Some(10),
                    params: Default::default(),
                });
                cost += 1.5;
            }
            RetrievalAction::MetadataFilter => {
                let mut params = std::collections::HashMap::new();
                for (k, v) in metadata_filters {
                    params.insert(k.clone(), v.clone());
                }
                plan.add_step(PlanStep {
                    step_type: PlanStepType::MetadataFilter,
                    index: Some("metadata".into()),
                    query: None,
                    limit: None,
                    params,
                });
                cost += 0.3;
            }
            RetrievalAction::TimeFilter => {
                plan.add_step(PlanStep {
                    step_type: PlanStepType::TimeFilter,
                    index: Some("time".into()),
                    query: None,
                    limit: None,
                    params: Default::default(),
                });
                cost += 0.3;
            }
            RetrievalAction::CodeSearch => {
                plan.add_step(PlanStep {
                    step_type: PlanStepType::VectorSearch,
                    index: Some("code-hnsw".into()),
                    query: Some(question.to_string()),
                    limit: Some(10),
                    params: Default::default(),
                });
                cost += 1.0;
            }
        }
    }

    // Step: Merge results from all sources
    plan.add_step(PlanStep {
        step_type: PlanStepType::Merge,
        index: None,
        query: None,
        limit: None,
        params: Default::default(),
    });

    // Step: Rerank merged results
    plan.add_step(PlanStep {
        step_type: PlanStepType::Rerank,
        index: None,
        query: Some(question.to_string()),
        limit: Some(10),
        params: Default::default(),
    });

    // Step: Build context for LLM
    plan.add_step(PlanStep {
        step_type: PlanStepType::BuildContext,
        index: None,
        query: None,
        limit: None,
        params: Default::default(),
    });

    plan.estimated_cost = cost;
    plan
}
