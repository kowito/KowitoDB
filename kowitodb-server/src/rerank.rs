//! Cross-encoder reranking.
//!
//! A cross-encoder scores a `(query, document)` pair *jointly* (unlike the
//! bi-encoder embeddings, which encode each independently), giving much sharper
//! relevance — the step that, per Anthropic's Contextual Retrieval study, takes
//! the top-20 retrieval failure rate from a 49% to a 67% reduction.
//!
//! The [`CrossEncoder`] trait is always compiled so the engine can hold an
//! optional reranker; the on-device Candle implementation is behind the
//! `cross-encoder-rerank` feature (it downloads a model on first use).

use async_trait::async_trait;

/// Scores how relevant each document is to the query (higher = better).
#[async_trait]
pub trait CrossEncoder: Send + Sync {
    async fn rerank(&self, query: &str, documents: &[String]) -> Vec<f32>;
}

#[cfg(feature = "cross-encoder-rerank")]
mod candle_impl {
    use candle_core::{Device, IndexOp, Tensor};
    use candle_nn::{Linear, Module, VarBuilder};
    use candle_transformers::models::bert::{BertModel, Config, DTYPE};
    use tokenizers::Tokenizer;
    use tracing::info;

    use super::CrossEncoder;

    /// Default cross-encoder reranker model.
    pub const DEFAULT_RERANKER_MODEL: &str = "BAAI/bge-reranker-base";

    const MAX_TOKENS: usize = 512;

    /// On-device BERT cross-encoder (BERT encoder + a single-logit classifier
    /// head over the `[CLS]` token), in the standard `BertForSequenceClassification`
    /// layout (`bert.*` weights + a `classifier` head).
    pub struct CandleCrossEncoder {
        model: BertModel,
        classifier: Linear,
        tokenizer: Tokenizer,
        device: Device,
    }

    impl CandleCrossEncoder {
        /// Load `model_id` from the HF hub (cached locally after first fetch).
        pub fn load(model_id: &str) -> anyhow::Result<Self> {
            let device = Device::Cpu;
            let repo = hf_hub::api::sync::Api::new()?.model(model_id.to_string());
            let config_path = repo.get("config.json")?;
            let tokenizer_path = repo.get("tokenizer.json")?;
            let weights_path = repo.get("model.safetensors")?;

            let config: Config = serde_json::from_str(&std::fs::read_to_string(config_path)?)?;
            let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)?;
            let vb =
                unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DTYPE, &device)? };
            // BertForSequenceClassification: encoder under "bert", single-logit head.
            let model = BertModel::load(vb.pp("bert"), &config)?;
            let classifier = candle_nn::linear(config.hidden_size, 1, vb.pp("classifier"))?;

            info!("Loaded cross-encoder reranker {model_id}");
            Ok(Self {
                model,
                classifier,
                tokenizer,
                device,
            })
        }

        /// Relevance logit for one (query, document) pair.
        fn score_pair(&self, query: &str, doc: &str) -> anyhow::Result<f32> {
            let encoding = self
                .tokenizer
                .encode((query, doc), true)
                .map_err(anyhow::Error::msg)?;
            let ids = encoding.get_ids();
            let ids = &ids[..ids.len().min(MAX_TOKENS)];
            let type_ids = encoding.get_type_ids();
            let type_ids = &type_ids[..type_ids.len().min(MAX_TOKENS)];

            let input_ids = Tensor::new(ids, &self.device)?.unsqueeze(0)?;
            let token_type_ids = Tensor::new(type_ids, &self.device)?.unsqueeze(0)?;

            let hidden = self.model.forward(&input_ids, &token_type_ids, None)?; // [1, seq, h]
            let cls = hidden.i((0, 0))?.unsqueeze(0)?; // [1, h] — the [CLS] token
            let logit = self.classifier.forward(&cls)?; // [1, 1]
            Ok(logit.to_vec2::<f32>()?[0][0])
        }
    }

    #[async_trait::async_trait]
    impl CrossEncoder for CandleCrossEncoder {
        async fn rerank(&self, query: &str, documents: &[String]) -> Vec<f32> {
            documents
                .iter()
                .map(|doc| self.score_pair(query, doc).unwrap_or(f32::MIN))
                .collect()
        }
    }
}

#[cfg(feature = "cross-encoder-rerank")]
pub use candle_impl::{CandleCrossEncoder, DEFAULT_RERANKER_MODEL};
