//! OpenAI-compatible embedding API client.
//!
//! Calls any OpenAI-compatible embeddings endpoint (OpenAI, Azure,
//! local Ollama, vLLM, etc.). Caches results to minimize API costs.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use kowitodb_core::Embedding;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use crate::embedding::{EmbeddingClient, EmbeddingError, EmbeddingResult};

/// Configuration for an OpenAI-compatible embeddings endpoint.
#[derive(Debug, Clone)]
pub struct OpenAiConfig {
    /// Base URL (e.g., "https://api.openai.com/v1")
    pub base_url: String,
    /// API key (or "ollama" for local Ollama)
    pub api_key: String,
    /// Model name (e.g., "text-embedding-3-small")
    pub model: String,
    /// Request timeout in seconds.
    pub timeout_secs: u64,
    /// Max retries on failure.
    pub max_retries: usize,
}

impl OpenAiConfig {
    /// OpenAI default: text-embedding-3-small (1536 dimensions).
    pub fn openai(api_key: impl Into<String>) -> Self {
        Self {
            base_url: "https://api.openai.com/v1".into(),
            api_key: api_key.into(),
            model: "text-embedding-3-small".into(),
            timeout_secs: 30,
            max_retries: 3,
        }
    }

    /// Local Ollama (no API key needed, 768-dim for nomic-embed-text).
    pub fn ollama(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            api_key: "ollama".into(),
            model: model.into(),
            timeout_secs: 60,
            max_retries: 2,
        }
    }

    /// Build a config from environment variables, or `None` to use the proxy.
    ///
    /// Driven by `KOWITODB_EMBEDDING_PROVIDER`:
    /// - `openai`: requires `OPENAI_API_KEY` (or `KOWITODB_OPENAI_API_KEY`);
    ///   optional `KOWITODB_OPENAI_BASE_URL`, `KOWITODB_EMBEDDING_MODEL`.
    /// - `ollama`: optional `KOWITODB_OLLAMA_URL` (default
    ///   `http://localhost:11434/v1`) and `KOWITODB_EMBEDDING_MODEL`
    ///   (default `nomic-embed-text`).
    /// - anything else / unset: `None` (caller falls back to the proxy).
    pub fn from_env() -> Option<Self> {
        match std::env::var("KOWITODB_EMBEDDING_PROVIDER")
            .ok()?
            .to_lowercase()
            .as_str()
        {
            "openai" => {
                let key = std::env::var("OPENAI_API_KEY")
                    .or_else(|_| std::env::var("KOWITODB_OPENAI_API_KEY"))
                    .ok()?;
                let mut cfg = Self::openai(key);
                if let Ok(url) = std::env::var("KOWITODB_OPENAI_BASE_URL") {
                    cfg.base_url = url;
                }
                if let Ok(model) = std::env::var("KOWITODB_EMBEDDING_MODEL") {
                    cfg.model = model;
                }
                Some(cfg)
            }
            "ollama" => {
                let url = std::env::var("KOWITODB_OLLAMA_URL")
                    .unwrap_or_else(|_| "http://localhost:11434/v1".into());
                let model = std::env::var("KOWITODB_EMBEDDING_MODEL")
                    .unwrap_or_else(|_| "nomic-embed-text".into());
                Some(Self::ollama(url, model))
            }
            _ => None,
        }
    }
}

/// OpenAI-compatible embedding client with caching.
pub struct OpenAiEmbeddingClient {
    config: OpenAiConfig,
    client: reqwest::Client,
    /// Embedding dimension (populated on first successful call).
    dimension: Arc<RwLock<Option<usize>>>,
    /// Cache: text → embedding.
    cache: Arc<RwLock<HashMap<String, Embedding>>>,
}

impl OpenAiEmbeddingClient {
    pub fn new(config: OpenAiConfig) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()
            .expect("Failed to build HTTP client");

        Self {
            config,
            client,
            dimension: Arc::new(RwLock::new(None)),
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

#[async_trait::async_trait]
impl EmbeddingClient for OpenAiEmbeddingClient {
    async fn embed(&self, text: &str) -> Result<EmbeddingResult, EmbeddingError> {
        // Check cache
        {
            let cache = self.cache.read();
            if let Some(cached) = cache.get(text) {
                debug!("OpenAI embedding cache HIT");
                return Ok(EmbeddingResult {
                    vector: cached.clone(),
                    model: self.config.model.clone(),
                    token_count: text.split_whitespace().count(),
                });
            }
        }

        let embedding = self.call_api(text).await?;

        // Cache result
        {
            let mut cache = self.cache.write();
            cache.insert(text.to_string(), embedding.vector.clone());
        }

        Ok(embedding)
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<EmbeddingResult>, EmbeddingError> {
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.embed(text).await?);
        }
        Ok(results)
    }

    fn model_name(&self) -> &str {
        &self.config.model
    }

    fn dimension(&self) -> usize {
        self.dimension.read().unwrap_or(1536) // default to OpenAI small
    }
}

impl OpenAiEmbeddingClient {
    async fn call_api(&self, text: &str) -> Result<EmbeddingResult, EmbeddingError> {
        let url = format!("{}/embeddings", self.config.base_url);

        let body = EmbeddingRequest {
            model: self.config.model.clone(),
            input: text.to_string(),
        };

        let mut last_error = None;

        for attempt in 0..=self.config.max_retries {
            if attempt > 0 {
                tokio::time::sleep(Duration::from_millis(500 * attempt as u64)).await;
                debug!(
                    "OpenAI retry attempt {}/{}",
                    attempt, self.config.max_retries
                );
            }

            let mut req = self.client.post(&url).json(&body);

            // Set auth header (skip for Ollama)
            if self.config.api_key != "ollama" {
                req = req.bearer_auth(&self.config.api_key);
            }

            match req.send().await {
                Ok(resp) => {
                    if resp.status().is_success() {
                        let api_resp: EmbeddingResponse = resp
                            .json()
                            .await
                            .map_err(|e| EmbeddingError::Api(e.to_string()))?;

                        if let Some(data) = api_resp.data.into_iter().next() {
                            // Update dimension cache
                            let dim = data.embedding.len();
                            {
                                let mut d = self.dimension.write();
                                *d = Some(dim);
                            }

                            let tokens = api_resp
                                .usage
                                .as_ref()
                                .map(|u| u.prompt_tokens)
                                .unwrap_or(text.split_whitespace().count());

                            info!(
                                "OpenAI embedding: model={}, dim={}, tokens={}",
                                self.config.model, dim, tokens,
                            );

                            return Ok(EmbeddingResult {
                                vector: data.embedding,
                                model: self.config.model.clone(),
                                token_count: tokens,
                            });
                        }

                        return Err(EmbeddingError::Api("Empty response data".into()));
                    }

                    let status = resp.status();
                    let body = resp.text().await.unwrap_or_default();

                    if status.as_u16() == 429 {
                        warn!("OpenAI rate limited, retrying...");
                        last_error = Some(EmbeddingError::RateLimit);
                        continue;
                    }

                    return Err(EmbeddingError::Api(format!("HTTP {}: {}", status, body)));
                }
                Err(e) => {
                    if e.is_timeout() {
                        last_error = Some(EmbeddingError::Timeout);
                    } else {
                        last_error = Some(EmbeddingError::Api(e.to_string()));
                    }
                }
            }
        }

        Err(last_error.unwrap_or(EmbeddingError::Api("Max retries exceeded".into())))
    }
}

// ---- OpenAI API types ----

#[derive(Debug, Serialize)]
struct EmbeddingRequest {
    model: String,
    input: String,
}

#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
    usage: Option<UsageInfo>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

#[derive(Debug, Deserialize, Clone)]
struct UsageInfo {
    prompt_tokens: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openai_config_builder() {
        let config = OpenAiConfig::openai("sk-test-key");
        assert_eq!(config.base_url, "https://api.openai.com/v1");
        assert_eq!(config.model, "text-embedding-3-small");
    }

    #[test]
    fn test_ollama_config_builder() {
        let config = OpenAiConfig::ollama("http://localhost:11434", "nomic-embed-text");
        assert_eq!(config.api_key, "ollama");
        assert_eq!(config.model, "nomic-embed-text");
    }
}
