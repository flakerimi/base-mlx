//! HuggingFace-style `config.json` parser. We only care about the
//! hyperparameters needed to build the forward pass; unknown fields are
//! tolerated. Quantization metadata is captured separately so the model
//! loader can pick between dense and quantized linear layers.

use crate::{Error, Result};
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    /// Architecture tag from HF (e.g. "qwen3"). Drives dispatch.
    #[serde(default)]
    pub model_type: String,

    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,

    /// GQA: kv heads can be fewer than attention heads. Defaults to
    /// `num_attention_heads` if absent (full MHA).
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,

    /// Per-head dimension. Most configs ship this; if missing we fall
    /// back to `hidden_size / num_attention_heads`.
    #[serde(default)]
    pub head_dim: Option<usize>,

    pub vocab_size: usize,

    #[serde(default = "default_rms_eps")]
    pub rms_norm_eps: f32,

    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,

    #[serde(default = "default_max_pos")]
    pub max_position_embeddings: usize,

    /// LM head shares the embedding matrix when true (Qwen3, Gemma).
    #[serde(default)]
    pub tie_word_embeddings: bool,

    /// Present when weights are quantized. mlx-lm writes *both*
    /// `quantization` and `quantization_config` with identical contents;
    /// `serde(alias)` would reject that as a duplicate, so we accept both
    /// independently and merge in `from_path` below.
    #[serde(default)]
    pub quantization: Option<QuantConfig>,
    #[serde(default, rename = "quantization_config")]
    pub quantization_alt: Option<QuantConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct QuantConfig {
    pub bits: u32,
    pub group_size: u32,
}

fn default_rms_eps() -> f32 {
    1e-6
}
fn default_rope_theta() -> f32 {
    10_000.0
}
fn default_max_pos() -> usize {
    4096
}

impl ModelConfig {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let raw = std::fs::read_to_string(path)?;
        let mut cfg: ModelConfig = serde_json::from_str(&raw)?;
        if cfg.quantization.is_none() {
            cfg.quantization = cfg.quantization_alt.take();
        }
        cfg.quantization_alt = None;
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<()> {
        if self.hidden_size == 0 || self.num_attention_heads == 0 {
            return Err(Error::InvalidConfig(
                "hidden_size and num_attention_heads must be non-zero".into(),
            ));
        }
        if self.hidden_size % self.num_attention_heads != 0 && self.head_dim.is_none() {
            return Err(Error::InvalidConfig(format!(
                "hidden_size ({}) not divisible by num_attention_heads ({}) and no explicit head_dim",
                self.hidden_size, self.num_attention_heads
            )));
        }
        Ok(())
    }

    pub fn kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    pub fn per_head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// Total query projection width: `num_attention_heads * head_dim`.
    pub fn q_dim(&self) -> usize {
        self.num_attention_heads * self.per_head_dim()
    }

    /// Total KV projection width: `kv_heads * head_dim`.
    pub fn kv_dim(&self) -> usize {
        self.kv_heads() * self.per_head_dim()
    }
}
