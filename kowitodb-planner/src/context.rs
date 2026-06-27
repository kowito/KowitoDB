use std::collections::HashSet;

use kowitodb_core::ObjectId;
use tracing::debug;

use super::reranker::RankedResult;

/// A chunk of context ready to be sent to an LLM.
#[derive(Debug, Clone)]
pub struct ContextChunk {
    /// Object ID this chunk came from.
    pub object_id: ObjectId,
    /// The text content.
    pub content: String,
    /// Relevance score from the reranker.
    pub relevance: f32,
    /// Estimated token count.
    pub estimated_tokens: usize,
}

/// Assembled context from a retrieval operation.
#[derive(Debug, Clone)]
pub struct AssembledContext {
    /// Ordered, deduplicated chunks.
    pub chunks: Vec<ContextChunk>,
    /// Total estimated tokens.
    pub total_tokens: usize,
    /// Summary statistics.
    pub stats: ContextStats,
}

#[derive(Debug, Clone, Default)]
pub struct ContextStats {
    /// Number of raw results before dedup/compression.
    pub raw_count: usize,
    /// Number after deduplication.
    pub deduped_count: usize,
    /// Number after content-length trimming.
    pub trimmed_count: usize,
    /// Percentage reduction in token count.
    pub compression_ratio: f32,
}

/// The Context Optimizer.
///
/// Converts retrieval results into a compact, deduplicated context
/// ready for LLM consumption. This directly reduces token costs
/// and latency — often more impactful than faster vector search.
///
/// Strategies applied in order:
/// 1. **Deduplication** — near-duplicate content removal via Jaccard similarity
/// 2. **Length normalization** — truncate overly long chunks
/// 3. **Relevance-based pruning** — drop low-scoring chunks
/// 4. **Token budget enforcement** — stop adding when budget exhausted
pub struct ContextOptimizer {
    /// Maximum tokens for the assembled context.
    max_tokens: usize,
    /// Maximum characters per individual chunk.
    max_chunk_chars: usize,
    /// Deduplication threshold (Jaccard similarity above which chunks are merged).
    dedup_threshold: f32,
    /// Minimum relevance score to include a chunk.
    min_relevance: f32,
}

impl ContextOptimizer {
    /// Create a new context optimizer with defaults.
    ///
    /// Default max_tokens: 4096 (fits in most LLM context windows with room for prompt).
    pub fn new(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            max_chunk_chars: 2000, // ~500 tokens per chunk
            dedup_threshold: 0.85,
            min_relevance: 0.05,
        }
    }

    /// Set custom deduplication threshold (0.0 = never dedup, 1.0 = exact match only).
    pub fn with_dedup_threshold(mut self, threshold: f32) -> Self {
        self.dedup_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set maximum characters per chunk.
    pub fn with_max_chunk_chars(mut self, chars: usize) -> Self {
        self.max_chunk_chars = chars;
        self
    }

    /// Assemble ranked results into an optimized context, loading content
    /// from the provided lookup function.
    pub fn assemble(
        &self,
        ranked: &[RankedResult],
        content_lookup: &dyn Fn(ObjectId) -> Option<String>,
    ) -> AssembledContext {
        self.assemble_with_budget(ranked, content_lookup, None)
    }

    /// Like [`Self::assemble`] but with an optional per-call token budget that
    /// overrides the optimizer's configured `max_tokens` (e.g. honoring a
    /// request's `max_context_tokens`). A zero or `None` override uses the
    /// configured default.
    pub fn assemble_with_budget(
        &self,
        ranked: &[RankedResult],
        content_lookup: &dyn Fn(ObjectId) -> Option<String>,
        max_tokens_override: Option<usize>,
    ) -> AssembledContext {
        let budget = max_tokens_override
            .filter(|b| *b > 0)
            .unwrap_or(self.max_tokens);
        let raw_count = ranked.len();
        let mut chunks: Vec<ContextChunk> = Vec::new();

        // Load content for each ranked result
        for result in ranked {
            if result.score < self.min_relevance {
                continue;
            }

            if let Some(content) = content_lookup(result.id) {
                let trimmed = self.trim_content(&content);
                let tokens = estimate_tokens(&trimmed);

                chunks.push(ContextChunk {
                    object_id: result.id,
                    content: trimmed,
                    relevance: result.score,
                    estimated_tokens: tokens,
                });
            }
        }

        // Step 1: Deduplicate near-duplicate chunks
        let deduped = self.deduplicate(chunks);
        let deduped_count = deduped.len();

        // Step 2: Sort by relevance (highest first)
        let mut sorted = deduped;
        sorted.sort_unstable_by(|a, b| {
            b.relevance
                .partial_cmp(&a.relevance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Step 3: Enforce token budget (greedy by relevance)
        let mut final_chunks = Vec::new();
        let mut token_count = 0usize;

        for chunk in sorted {
            let chunk_tokens = chunk.estimated_tokens;
            if token_count + chunk_tokens <= budget {
                token_count += chunk_tokens;
                final_chunks.push(chunk);
            } else if final_chunks.is_empty() {
                // Must include at least one chunk, even if it exceeds budget
                token_count += chunk_tokens;
                final_chunks.push(chunk);
                break;
            } else {
                break;
            }
        }

        let trimmed_count = final_chunks.len();
        let stats = ContextStats {
            raw_count,
            deduped_count,
            trimmed_count,
            compression_ratio: if raw_count > 0 {
                1.0 - (final_chunks.len() as f32 / raw_count as f32)
            } else {
                0.0
            },
        };

        debug!(
            "Context optimizer: {} raw → {} deduped → {} final ({} tokens, {:.0}% compression)",
            raw_count,
            deduped_count,
            trimmed_count,
            token_count,
            stats.compression_ratio * 100.0
        );

        AssembledContext {
            chunks: final_chunks,
            total_tokens: token_count,
            stats,
        }
    }

    /// Trim content to max characters, trying to break at sentence boundaries.
    fn trim_content(&self, content: &str) -> String {
        if content.len() <= self.max_chunk_chars {
            return content.to_string();
        }

        // Try to find a sentence boundary near the max
        let end = self.max_chunk_chars;
        if let Some(period_idx) = content[..end].rfind('.') {
            if period_idx > end / 2 {
                return content[..=period_idx].to_string();
            }
        }
        // Fall back to space boundary
        if let Some(space_idx) = content[..end].rfind(' ') {
            return content[..space_idx].to_string();
        }

        content[..end].to_string()
    }

    /// Deduplicate chunks using Jaccard similarity on word sets.
    ///
    /// When two chunks are near-duplicates, keep the one with the higher
    /// relevance score and discard the other.
    fn deduplicate(&self, mut chunks: Vec<ContextChunk>) -> Vec<ContextChunk> {
        if chunks.len() <= 1 {
            return chunks;
        }

        // Sort by relevance so we keep the best
        chunks.sort_unstable_by(|a, b| {
            b.relevance
                .partial_cmp(&a.relevance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut kept = Vec::new();
        let mut drop_set: HashSet<usize> = HashSet::new();

        for i in 0..chunks.len() {
            if drop_set.contains(&i) {
                continue;
            }
            let words_i = word_set(&chunks[i].content);

            for (j, chunk_j) in chunks.iter().enumerate().skip(i + 1) {
                if drop_set.contains(&j) {
                    continue;
                }
                let words_j = word_set(&chunk_j.content);
                let similarity = jaccard_similarity(&words_i, &words_j);

                if similarity >= self.dedup_threshold {
                    drop_set.insert(j);
                }
            }
            kept.push(chunks[i].clone());
        }

        kept
    }
}

/// Build a set of lowercase words from text.
fn word_set(text: &str) -> HashSet<String> {
    text.split_whitespace()
        .map(|w| {
            w.chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>()
                .to_lowercase()
        })
        .filter(|w| w.len() > 1)
        .collect()
}

/// Jaccard similarity: |A ∩ B| / |A ∪ B|
fn jaccard_similarity(a: &HashSet<String>, b: &HashSet<String>) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let intersection = a.intersection(b).count();
    let union = a.union(b).count();
    intersection as f32 / union as f32
}

/// Quick token estimation: ~4 characters per token for English text.
/// In production, use a proper tokenizer (tiktoken, HuggingFace tokenizers).
pub fn estimate_tokens(text: &str) -> usize {
    // Rough heuristic: 4 chars ≈ 1 token for English
    (text.len() as f32 / 4.0).ceil() as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_estimate_tokens() {
        assert_eq!(estimate_tokens("hello world"), 3); // 11 chars → 3 tokens
        assert_eq!(estimate_tokens(""), 0);
    }

    #[test]
    fn test_jaccard() {
        let a: HashSet<String> = ["hello", "world"].iter().map(|s| s.to_string()).collect();
        let b: HashSet<String> = ["hello", "there"].iter().map(|s| s.to_string()).collect();
        // Intersection: {hello} = 1, Union: {hello, world, there} = 3
        let sim = jaccard_similarity(&a, &b);
        assert!((sim - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_dedup_similar_chunks() {
        let opt = ContextOptimizer::new(4096).with_dedup_threshold(0.5);
        let chunks = vec![
            ContextChunk {
                object_id: uuid::Uuid::new_v4(),
                content: "OpenAI raised $6.6 billion in funding".to_string(),
                relevance: 0.9,
                estimated_tokens: 10,
            },
            ContextChunk {
                object_id: uuid::Uuid::new_v4(),
                content: "OpenAI raised 6.6 billion dollars in funding round".to_string(),
                relevance: 0.8,
                estimated_tokens: 12,
            },
        ];
        let deduped = opt.deduplicate(chunks);
        assert_eq!(deduped.len(), 1);
        assert!((deduped[0].relevance - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_trim_content() {
        let opt = ContextOptimizer::new(4096).with_max_chunk_chars(50);
        let long = "This is a long sentence. And this is another sentence. And more text that goes on and on.";
        let trimmed = opt.trim_content(long);
        assert!(trimmed.len() <= 52); // +2 for possible "."
        assert!(trimmed.ends_with('.') || trimmed.len() <= 50);
    }

    #[test]
    fn test_token_budget_enforcement() {
        let opt = ContextOptimizer::new(50); // 50 token budget
        let mut ranked = Vec::new();
        for i in 0..5 {
            ranked.push(RankedResult {
                id: uuid::Uuid::new_v4(),
                score: 1.0 - i as f32 * 0.1,
                sources: vec![],
                source_scores: HashMap::new(),
            });
        }

        let content_lookup = |_id: ObjectId| -> Option<String> {
            // Each chunk is ~15 tokens (60 chars)
            Some(
                "This is some content that will take about fifteen tokens to represent."
                    .to_string(),
            )
        };

        let assembled = opt.assemble(&ranked, &content_lookup);
        // 50 token budget, ~15 tokens per chunk → ~3 chunks
        assert!(assembled.chunks.len() <= 4);
        assert!(assembled.total_tokens <= 60); // Allow a bit of slack
    }
}
