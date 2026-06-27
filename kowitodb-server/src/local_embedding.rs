//! Local embedding model via [Candle](https://github.com/huggingface/candle).
//!
//! Runs a BERT sentence-transformer (default `all-MiniLM-L6-v2`, 384-dim)
//! entirely on-device — no API key, no external service. The model is fetched
//! from the Hugging Face hub on first use and cached locally; subsequent runs
//! are offline.
//!
//! Enabled with the `local-embeddings` cargo feature and selected at runtime via
//! `KOWITODB_EMBEDDING_PROVIDER=local`. `KOWITODB_EMBEDDING_MODEL` overrides the
//! model id.

use std::collections::HashMap;

use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use kowitodb_core::Embedding;
use parking_lot::RwLock;
use tokenizers::Tokenizer;
use tracing::{debug, info};

use crate::embedding::{EmbeddingClient, EmbeddingError, EmbeddingResult};

/// Default sentence-transformer model (384-dim, small and fast).
pub const DEFAULT_LOCAL_MODEL: &str = "sentence-transformers/all-MiniLM-L6-v2";

/// Maximum input tokens (BERT context limit).
const MAX_TOKENS: usize = 512;

/// On-device BERT embedding client.
pub struct LocalEmbeddingClient {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    model_id: String,
    dimension: usize,
    cache: RwLock<HashMap<String, Embedding>>,
}

impl LocalEmbeddingClient {
    /// Load `model_id` from the HF hub (cached locally after the first fetch).
    pub fn load(model_id: &str) -> anyhow::Result<Self> {
        let device = Device::Cpu;
        let repo = hf_hub::api::sync::Api::new()?.model(model_id.to_string());

        let config_path = repo.get("config.json")?;
        let tokenizer_path = repo.get("tokenizer.json")?;
        let weights_path = repo.get("model.safetensors").map_err(|e| {
            anyhow::anyhow!("{model_id} has no model.safetensors (only PyTorch weights?): {e}")
        })?;

        let config: Config = serde_json::from_str(&std::fs::read_to_string(config_path)?)?;
        let dimension = config.hidden_size;
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DTYPE, &device)? };
        let model = BertModel::load(vb, &config)?;

        info!("Loaded local embedding model {model_id} (dim={dimension})");
        Ok(Self {
            model,
            tokenizer,
            device,
            model_id: model_id.to_string(),
            dimension,
            cache: RwLock::new(HashMap::new()),
        })
    }

    /// Run the model over one text and mean-pool + L2-normalize to a vector.
    fn embed_text(&self, text: &str) -> anyhow::Result<Embedding> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(anyhow::Error::msg)?;
        let ids = encoding.get_ids();
        let ids = &ids[..ids.len().min(MAX_TOKENS)];

        let input_ids = Tensor::new(ids, &self.device)?.unsqueeze(0)?;
        let token_type_ids = input_ids.zeros_like()?;

        // [1, seq, hidden]
        let hidden = self.model.forward(&input_ids, &token_type_ids, None)?;
        let (_b, seq, _h) = hidden.dims3()?;

        // Mean pooling over the sequence, then L2 normalization.
        let pooled = (hidden.sum(1)? / seq as f64)?; // [1, hidden]
        let norm = pooled.sqr()?.sum_keepdim(1)?.sqrt()?; // [1, 1]
        let normalized = pooled.broadcast_div(&norm)?;
        let vector = normalized.squeeze(0)?.to_vec1::<f32>()?;
        Ok(vector)
    }
}

#[async_trait::async_trait]
impl EmbeddingClient for LocalEmbeddingClient {
    async fn embed(&self, text: &str) -> Result<EmbeddingResult, EmbeddingError> {
        if let Some(cached) = self.cache.read().get(text) {
            return Ok(EmbeddingResult {
                vector: cached.clone(),
                model: self.model_id.clone(),
                token_count: text.split_whitespace().count(),
            });
        }

        debug!("Local embed: \"{}\"...", &text[..text.len().min(60)]);
        let vector = self
            .embed_text(text)
            .map_err(|e| EmbeddingError::Api(e.to_string()))?;
        self.cache.write().insert(text.to_string(), vector.clone());

        Ok(EmbeddingResult {
            vector,
            model: self.model_id.clone(),
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
        &self.model_id
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}
