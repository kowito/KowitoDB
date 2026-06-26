use std::sync::Arc;

use parking_lot::RwLock;
use tracing::info;

/// Tracks the estimated cost of query operations.
///
/// Models cost in terms of:
/// - Index lookups (cheap)
/// - Embedding API calls (medium)
/// - LLM context tokens (expensive)
///
/// This is the Cost Optimizer from the architecture — it enables
/// the planner to select cheaper strategies when quality is comparable.
#[derive(Debug, Clone)]
pub struct CostTracker {
    /// Cumulative estimated cost in USD.
    total_cost_usd: Arc<RwLock<f64>>,
    /// Cost model parameters.
    model: CostModel,
}

/// Pricing model for various operations.
///
/// Defaults approximate current market rates. These can be tuned
/// per deployment.
#[derive(Debug, Clone)]
pub struct CostModel {
    /// Cost per 1000 embedding API calls.
    pub embedding_cost_per_1k: f64,
    /// Cost per 1000 input tokens (LLM).
    pub llm_input_cost_per_1k_tokens: f64,
    /// Cost per 1000 output tokens (LLM).
    pub llm_output_cost_per_1k_tokens: f64,
    /// Cost per 1000 index lookups (essentially free compute).
    pub index_cost_per_1k: f64,
}

impl Default for CostModel {
    fn default() -> Self {
        Self {
            // ~OpenAI text-embedding-3-small pricing
            embedding_cost_per_1k: 0.00002,
            // ~GPT-4o-mini input pricing
            llm_input_cost_per_1k_tokens: 0.00015,
            // ~GPT-4o-mini output pricing
            llm_output_cost_per_1k_tokens: 0.0006,
            // Local index lookups are effectively free
            index_cost_per_1k: 0.0,
        }
    }
}

impl CostTracker {
    pub fn new() -> Self {
        Self {
            total_cost_usd: Arc::new(RwLock::new(0.0)),
            model: CostModel::default(),
        }
    }

    pub fn with_model(model: CostModel) -> Self {
        Self {
            total_cost_usd: Arc::new(RwLock::new(0.0)),
            model,
        }
    }

    /// Record an embedding API call.
    pub fn record_embedding_calls(&self, count: usize) {
        let cost = (count as f64 / 1000.0) * self.model.embedding_cost_per_1k;
        self.add_cost(cost);
        info!("Cost: {} embedding call(s) = ${:.6}", count, cost);
    }

    /// Record LLM input tokens consumed.
    pub fn record_llm_input_tokens(&self, tokens: usize) {
        let cost = (tokens as f64 / 1000.0) * self.model.llm_input_cost_per_1k_tokens;
        self.add_cost(cost);
        info!("Cost: {} LLM input tokens = ${:.6}", tokens, cost);
    }

    /// Record LLM output tokens consumed.
    pub fn record_llm_output_tokens(&self, tokens: usize) {
        let cost = (tokens as f64 / 1000.0) * self.model.llm_output_cost_per_1k_tokens;
        self.add_cost(cost);
        info!("Cost: {} LLM output tokens = ${:.6}", tokens, cost);
    }

    /// Record index lookups.
    pub fn record_index_lookups(&self, count: usize) {
        let cost = (count as f64 / 1000.0) * self.model.index_cost_per_1k;
        self.add_cost(cost);
    }

    /// Estimate the cost of a plan before executing it.
    /// Returns the estimated cost so the planner can choose cheaper alternatives.
    pub fn estimate_plan_cost(
        &self,
        embedding_calls: usize,
        index_lookups: usize,
        context_tokens: usize,
        output_tokens: usize,
    ) -> f64 {
        let emb_cost = (embedding_calls as f64 / 1000.0) * self.model.embedding_cost_per_1k;
        let idx_cost = (index_lookups as f64 / 1000.0) * self.model.index_cost_per_1k;
        let input_cost = (context_tokens as f64 / 1000.0) * self.model.llm_input_cost_per_1k_tokens;
        let output_cost =
            (output_tokens as f64 / 1000.0) * self.model.llm_output_cost_per_1k_tokens;
        emb_cost + idx_cost + input_cost + output_cost
    }

    /// Get the total accumulated cost.
    pub fn total_cost(&self) -> f64 {
        *self.total_cost_usd.read()
    }

    /// Reset the cost counter.
    pub fn reset(&self) {
        *self.total_cost_usd.write() = 0.0;
    }

    fn add_cost(&self, cost: f64) {
        *self.total_cost_usd.write() += cost;
    }
}

impl Default for CostTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cost_accumulation() {
        let tracker = CostTracker::new();
        tracker.record_embedding_calls(1);
        tracker.record_llm_input_tokens(2000);
        tracker.record_llm_output_tokens(500);

        let total = tracker.total_cost();
        assert!(total > 0.0);
    }

    #[test]
    fn test_estimate_plan_cost() {
        let tracker = CostTracker::new();
        let cost = tracker.estimate_plan_cost(1, 20, 5000, 500);

        // 1 embedding + 20 index lookups + 5000 context + 500 output tokens
        assert!(cost > 0.0);
        assert!(cost < 0.01); // Should be less than 1 cent for a single query
    }

    #[test]
    fn test_cost_reset() {
        let tracker = CostTracker::new();
        tracker.record_embedding_calls(1000);
        assert!(tracker.total_cost() > 0.0);
        tracker.reset();
        assert_eq!(tracker.total_cost(), 0.0);
    }
}
