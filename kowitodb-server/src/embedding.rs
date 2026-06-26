use std::collections::HashMap;
use std::sync::Arc;

use kowitodb_core::Embedding;
use parking_lot::RwLock;
use tracing::debug;

/// Result of an embedding request.
#[derive(Debug, Clone)]
pub struct EmbeddingResult {
    /// The embedding vector.
    pub vector: Embedding,
    /// The model used.
    pub model: String,
    /// Number of tokens in the input.
    pub token_count: usize,
}

/// Abstract embedding client, made dyn-compatible via async_trait.
#[async_trait::async_trait]
pub trait EmbeddingClient: Send + Sync {
    /// Generate an embedding for the given text.
    async fn embed(&self, text: &str) -> Result<EmbeddingResult, EmbeddingError>;

    /// Batch-embed multiple texts.
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<EmbeddingResult>, EmbeddingError>;

    /// Get the model name.
    fn model_name(&self) -> &str;

    /// Get the embedding dimension.
    fn dimension(&self) -> usize;
}

#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
    #[error("API error: {0}")]
    Api(String),
    #[error("Rate limit exceeded")]
    RateLimit,
    #[error("Timeout")]
    Timeout,
    #[error("Model not found: {0}")]
    ModelNotFound(String),
}

/// A proxy embedding client for development and testing.
pub struct ProxyEmbeddingClient {
    model: String,
    dimension: usize,
    cache: Arc<RwLock<HashMap<String, Embedding>>>,
}

impl ProxyEmbeddingClient {
    pub fn new(model: impl Into<String>, dimension: usize) -> Self {
        Self {
            model: model.into(),
            dimension,
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

#[async_trait::async_trait]
impl EmbeddingClient for ProxyEmbeddingClient {
    async fn embed(&self, text: &str) -> Result<EmbeddingResult, EmbeddingError> {
        // Check cache
        {
            let cache = self.cache.read();
            if let Some(cached) = cache.get(text) {
                debug!("Embedding cache HIT for: {}", &text[..text.len().min(60)]);
                return Ok(EmbeddingResult {
                    vector: cached.clone(),
                    model: self.model.clone(),
                    token_count: text.split_whitespace().count(),
                });
            }
        }

        debug!("Embedding: \"{}\"...", &text[..text.len().min(60)]);
        let vector = text_to_embedding(text, self.dimension);

        {
            let mut cache = self.cache.write();
            cache.insert(text.to_string(), vector.clone());
        }

        Ok(EmbeddingResult {
            vector,
            model: self.model.clone(),
            token_count: text.split_whitespace().count(),
        })
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<EmbeddingResult>, EmbeddingError> {
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.embed(text).await?);
        }
        Ok(results)
    }

    fn model_name(&self) -> &str {
        &self.model
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}

/// Generate a deterministic pseudo-embedding from text.
pub fn text_to_embedding(text: &str, dim: usize) -> Embedding {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut vec = vec![0.0f32; dim];

    for (i, word) in text.split_whitespace().enumerate() {
        let mut hasher = DefaultHasher::new();
        word.hash(&mut hasher);
        let h = hasher.finish();
        let idx = (h as usize) % dim;
        vec[idx] += 1.0;
        vec[(idx + 1) % dim] += 0.5;
        vec[(idx + 7) % dim] += 0.3;
        vec[(idx + 13) % dim] += 0.2;
        vec[(idx.wrapping_add(i * 3)) % dim] += 0.15;
    }

    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        vec.iter_mut().for_each(|x| *x /= norm);
    }

    vec
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proxy_embedding_deterministic() {
        let client = ProxyEmbeddingClient::new("proxy", 128);
        let rt = tokio::runtime::Runtime::new().unwrap();

        let r1 = rt.block_on(client.embed("hello world")).unwrap();
        let r2 = rt.block_on(client.embed("hello world")).unwrap();

        assert_eq!(r1.vector, r2.vector);
        assert_eq!(r1.vector.len(), 128);
    }

    #[test]
    fn test_normalization() {
        let vec = text_to_embedding("test", 64);
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5 || vec.iter().all(|&x| x == 0.0));
    }
}
