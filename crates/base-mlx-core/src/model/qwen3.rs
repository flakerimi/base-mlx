//! Qwen3 (dense, Instruct variant) — v1 reference architecture.
//!
//! Tensor layout from `mlx-community/Qwen3-*-4bit`:
//!   model.embed_tokens.weight                                (Embedding)
//!   model.layers.{i}.input_layernorm.weight                  (RMSNorm)
//!   model.layers.{i}.self_attn.q_proj.{weight,scales,biases} (QuantLinear) +
//!       (optional) q_proj.bias (none in Qwen3 — `attention_bias=false`)
//!   model.layers.{i}.self_attn.k_proj.{weight,scales,biases}
//!   model.layers.{i}.self_attn.v_proj.{weight,scales,biases}
//!   model.layers.{i}.self_attn.q_norm.weight                 (RMSNorm on per-head q)
//!   model.layers.{i}.self_attn.k_norm.weight                 (RMSNorm on per-head k)
//!   model.layers.{i}.self_attn.o_proj.{weight,scales,biases}
//!   model.layers.{i}.post_attention_layernorm.weight         (RMSNorm)
//!   model.layers.{i}.mlp.gate_proj.{weight,scales,biases}
//!   model.layers.{i}.mlp.up_proj.{weight,scales,biases}
//!   model.layers.{i}.mlp.down_proj.{weight,scales,biases}
//!   model.norm.weight                                        (final RMSNorm)
//!   lm_head.weight  — *missing* when `tie_word_embeddings=true`; share
//!                     `embed_tokens.weight`.
//!
//! GQA: q-head count = `num_attention_heads`, k/v-head count =
//! `num_key_value_heads`, head_dim = `head_dim` (128 for Qwen3-4B).
//! `q_norm`/`k_norm` are per-head RMSNorms applied after q/k projection
//! and *before* RoPE — a Qwen3-specific detail.
//!
//! Forward-pass implementation lands in the next pass once we've audited
//! mlx-rs's QuantizedLinear + RoPE shape conventions against an actual
//! load + single-token greedy decode.

use super::ModelConfig;

#[derive(Debug)]
pub struct Qwen3 {
    pub cfg: ModelConfig,
    // mlx-rs modules will be filled here: token embedding, layer stack,
    // final RMSNorm, optional LM head. Held back until the load path is
    // verified end-to-end against the LM Studio shards.
}

impl Qwen3 {
    /// Construct an *empty* model from a config. Weights are loaded
    /// separately via `load_weights` (not yet implemented).
    pub fn from_config(cfg: ModelConfig) -> Self {
        Self { cfg }
    }

    /// Expected weight count given the config — useful as a load-time
    /// sanity check before we try to use the model. Counts the named
    /// tensors per layer plus the global ones.
    ///
    /// Per-layer (Qwen3 quantized, attention_bias=false):
    ///   - input_layernorm: weight                              = 1
    ///   - self_attn.{q,k,v,o}_proj: weight + scales + biases   = 4 × 3 = 12
    ///   - self_attn.{q,k}_norm: weight                         = 2
    ///   - post_attention_layernorm: weight                     = 1
    ///   - mlp.{gate,up,down}_proj: weight + scales + biases    = 3 × 3 = 9
    ///   total per layer                                        = 25
    ///
    /// Global:
    ///   - embed_tokens.weight (+ scales + biases when quantized)
    ///   - final norm.weight
    ///   - lm_head when not tied
    pub fn expected_tensor_count(cfg: &ModelConfig) -> usize {
        let per_layer = 25;
        let mut global = 1 /* final norm */ + 1 /* embed_tokens.weight */;
        if cfg.quantization.is_some() {
            global += 2; // embed_tokens.scales + biases
        }
        if !cfg.tie_word_embeddings {
            // lm_head: weight (+ scales + biases when quantized)
            global += if cfg.quantization.is_some() { 3 } else { 1 };
        }
        cfg.num_hidden_layers * per_layer + global
    }
}
