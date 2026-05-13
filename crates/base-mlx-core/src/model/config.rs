//! HuggingFace-style `config.json` parser. We only care about the
//! hyperparameters needed to build the forward pass; unknown fields are
//! tolerated. Quantization metadata is captured separately so the model
//! loader can pick between dense and quantized linear layers.

use crate::{Error, Result};
use serde::{Deserialize, Deserializer};
use serde_json::{Map, Value};
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

    #[serde(default)]
    pub bos_token_id: Option<u32>,

    #[serde(default, deserialize_with = "token_ids")]
    pub eos_token_id: Vec<u32>,

    #[serde(default)]
    pub pad_token_id: Option<u32>,

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

    /// Gemma 4 alternates sliding and full attention layers.
    #[serde(default)]
    pub layer_types: Vec<String>,
    #[serde(default)]
    pub rope_parameters: Option<Value>,
    #[serde(default)]
    pub global_head_dim: Option<usize>,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub num_kv_shared_layers: Option<usize>,
    #[serde(default)]
    pub hidden_size_per_layer_input: Option<usize>,
    #[serde(default)]
    pub vocab_size_per_layer_input: Option<usize>,
    #[serde(default)]
    pub final_logit_softcapping: Option<f32>,
    #[serde(default)]
    pub use_double_wide_mlp: bool,
    #[serde(default)]
    pub enable_moe_block: bool,
    #[serde(default)]
    pub attention_k_eq_v: bool,
    #[serde(default)]
    pub num_global_key_value_heads: Option<usize>,

    /// Granite hybrid/Mamba2 settings.
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default = "default_one")]
    pub embedding_multiplier: f32,
    #[serde(default = "default_one")]
    pub attention_multiplier: f32,
    #[serde(default = "default_one")]
    pub logits_scaling: f32,
    #[serde(default = "default_one")]
    pub residual_multiplier: f32,
    #[serde(default)]
    pub shared_intermediate_size: Option<usize>,
    #[serde(default)]
    pub num_local_experts: Option<usize>,
    #[serde(default)]
    pub num_experts_per_tok: Option<usize>,
    #[serde(default)]
    pub mamba_n_heads: Option<usize>,
    #[serde(default)]
    pub mamba_d_head: Option<usize>,
    #[serde(default)]
    pub mamba_d_state: Option<usize>,
    #[serde(default)]
    pub mamba_d_conv: Option<usize>,
    #[serde(default)]
    pub mamba_n_groups: Option<usize>,
    #[serde(default)]
    pub mamba_conv_bias: bool,
    #[serde(default)]
    pub mamba_proj_bias: bool,
    #[serde(default = "default_position_embedding_type")]
    pub position_embedding_type: String,
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
fn default_one() -> f32 {
    1.0
}
fn default_position_embedding_type() -> String {
    "rope".into()
}

impl ModelConfig {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let raw = std::fs::read_to_string(path)?;
        let value: Value = serde_json::from_str(&raw)?;
        let mut cfg: ModelConfig = if value
            .get("model_type")
            .and_then(Value::as_str)
            .is_some_and(|t| t == "gemma4")
        {
            serde_json::from_value(gemma4_text_config(value))?
        } else {
            serde_json::from_value(value)?
        };
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
        if !self.hidden_size.is_multiple_of(self.num_attention_heads) && self.head_dim.is_none() {
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

fn gemma4_text_config(root: Value) -> Value {
    let mut text = root
        .get("text_config")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();

    text.insert("model_type".into(), Value::String("gemma4".into()));
    copy_field(&root, &mut text, "quantization");
    copy_field(&root, &mut text, "quantization_config");
    copy_field(&root, &mut text, "eos_token_id");

    Value::Object(text)
}

fn copy_field(root: &Value, text: &mut Map<String, Value>, key: &str) {
    if let Some(v) = root.get(key) {
        text.insert(key.to_string(), v.clone());
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
enum TokenIds {
    One(u32),
    Many(Vec<u32>),
}

fn token_ids<'de, D>(deserializer: D) -> std::result::Result<Vec<u32>, D::Error>
where
    D: Deserializer<'de>,
{
    Ok(match Option::<TokenIds>::deserialize(deserializer)? {
        Some(TokenIds::One(id)) => vec![id],
        Some(TokenIds::Many(ids)) => ids,
        None => Vec::new(),
    })
}
