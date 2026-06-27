//! Generative LLM client used by the optional, pluggable AI features:
//! LLM-generated contextual retrieval, natural-language→SQL routing, and
//! Mem0-style memory consolidation.
//!
//! The [`LlmClient`] trait is the seam: the engine holds an `Option<Arc<dyn
//! LlmClient>>` and every feature degrades to its deterministic behavior when
//! no client is configured. An OpenAI-compatible implementation (OpenAI, Azure,
//! Ollama, vLLM, …) is provided; tests inject a mock so the *mechanism* is
//! verified without a live endpoint.

use async_trait::async_trait;

/// A minimal chat/completion interface: a system instruction plus a user
/// message in, a single text completion out.
#[async_trait]
pub trait LlmClient: Send + Sync {
    async fn complete(&self, system: &str, user: &str) -> anyhow::Result<String>;
}

/// Configuration for an OpenAI-compatible chat-completions endpoint.
#[derive(Debug, Clone)]
pub struct LlmConfig {
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    pub timeout_secs: u64,
    /// Upper bound on generated tokens (kept small — these are short helpers).
    pub max_tokens: u32,
}

impl LlmConfig {
    /// Build from the environment, or `None` to disable generative features.
    ///
    /// Driven by `KOWITODB_LLM_PROVIDER`:
    /// - `openai`: requires `OPENAI_API_KEY` (or `KOWITODB_OPENAI_API_KEY`);
    ///   optional `KOWITODB_LLM_BASE_URL`, `KOWITODB_LLM_MODEL`.
    /// - `ollama`: optional `KOWITODB_OLLAMA_URL`, `KOWITODB_LLM_MODEL`
    ///   (default `llama3.2`).
    /// - anything else / unset: `None`.
    pub fn from_env() -> Option<Self> {
        match std::env::var("KOWITODB_LLM_PROVIDER")
            .ok()?
            .to_lowercase()
            .as_str()
        {
            "openai" => {
                let api_key = std::env::var("OPENAI_API_KEY")
                    .or_else(|_| std::env::var("KOWITODB_OPENAI_API_KEY"))
                    .ok()?;
                Some(Self {
                    base_url: std::env::var("KOWITODB_LLM_BASE_URL")
                        .unwrap_or_else(|_| "https://api.openai.com/v1".into()),
                    api_key,
                    model: std::env::var("KOWITODB_LLM_MODEL")
                        .unwrap_or_else(|_| "gpt-4o-mini".into()),
                    timeout_secs: 30,
                    max_tokens: 512,
                })
            }
            "ollama" => Some(Self {
                base_url: std::env::var("KOWITODB_OLLAMA_URL")
                    .unwrap_or_else(|_| "http://localhost:11434/v1".into()),
                api_key: "ollama".into(),
                model: std::env::var("KOWITODB_LLM_MODEL").unwrap_or_else(|_| "llama3.2".into()),
                timeout_secs: 60,
                max_tokens: 512,
            }),
            _ => None,
        }
    }
}

/// OpenAI-compatible chat-completions client.
pub struct OpenAiLlmClient {
    config: LlmConfig,
    client: reqwest::Client,
}

impl OpenAiLlmClient {
    pub fn new(config: LlmConfig) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_secs))
            .build()
            .expect("failed to build HTTP client");
        Self { config, client }
    }
}

#[derive(serde::Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage<'a>>,
    max_tokens: u32,
    temperature: f32,
}

#[derive(serde::Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(serde::Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(serde::Deserialize)]
struct ChatChoice {
    message: ChatResponseMessage,
}

#[derive(serde::Deserialize)]
struct ChatResponseMessage {
    content: String,
}

#[async_trait]
impl LlmClient for OpenAiLlmClient {
    async fn complete(&self, system: &str, user: &str) -> anyhow::Result<String> {
        let req = ChatRequest {
            model: &self.config.model,
            messages: vec![
                ChatMessage {
                    role: "system",
                    content: system,
                },
                ChatMessage {
                    role: "user",
                    content: user,
                },
            ],
            max_tokens: self.config.max_tokens,
            temperature: 0.0,
        };
        let resp = self
            .client
            .post(format!("{}/chat/completions", self.config.base_url))
            .bearer_auth(&self.config.api_key)
            .json(&req)
            .send()
            .await?
            .error_for_status()?
            .json::<ChatResponse>()
            .await?;
        resp.choices
            .into_iter()
            .next()
            .map(|c| c.message.content.trim().to_string())
            .ok_or_else(|| anyhow::anyhow!("LLM returned no choices"))
    }
}

/// Build the configured LLM client, or `None` to keep generative features off.
pub fn from_env() -> Option<std::sync::Arc<dyn LlmClient>> {
    let cfg = LlmConfig::from_env()?;
    tracing::info!("LLM client enabled: {} ({})", cfg.model, cfg.base_url);
    Some(std::sync::Arc::new(OpenAiLlmClient::new(cfg)))
}

#[cfg(test)]
pub(crate) mod testing {
    use super::*;

    /// A deterministic mock that echoes a canned response, for verifying the
    /// generative-feature *mechanisms* without a live endpoint.
    pub struct MockLlm {
        pub response: String,
    }

    #[async_trait]
    impl LlmClient for MockLlm {
        async fn complete(&self, _system: &str, _user: &str) -> anyhow::Result<String> {
            Ok(self.response.clone())
        }
    }
}
