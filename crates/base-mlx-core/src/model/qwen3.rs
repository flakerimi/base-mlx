//! Qwen3 (dense, Instruct variant).
//!
//! v1 implementation: full reprefill on every step (no KV cache yet).
//! Correctness first; cache + speculative decoding land as separate
//! milestones once a single forward pass produces sensible logits.

use super::ModelConfig;
use crate::{Error, Result};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::{Array, Dtype};
use std::collections::HashMap;
use std::path::Path;

// ─── Building blocks ────────────────────────────────────────────────────────

/// A quantized linear layer in MLX format: packed-uint32 weight + per-group
/// scales + biases. Forward = `quantized_matmul(x, W, s, b, transpose=true)`.
#[derive(Debug)]
struct QLinear {
    weight: Array,
    scales: Array,
    biases: Array,
    group_size: i32,
    bits: i32,
}

impl QLinear {
    fn forward(&self, x: &Array) -> Result<Array> {
        mlx_rs::ops::quantized_matmul(
            x,
            &self.weight,
            &self.scales,
            &self.biases,
            /* transpose */ true,
            self.group_size,
            self.bits,
        )
        .map_err(|e| Error::Inference(e.to_string()))
    }
}

#[derive(Debug)]
struct RmsNorm {
    weight: Array,
    eps: f32,
}

impl RmsNorm {
    fn forward(&self, x: &Array) -> Result<Array> {
        mlx_rs::fast::rms_norm(x, &self.weight, self.eps)
            .map_err(|e| Error::Inference(e.to_string()))
    }
}

/// Quantized embedding table. Lookup = dequantize selected rows.
#[derive(Debug)]
struct QEmbedding {
    weight: Array,
    scales: Array,
    biases: Array,
    group_size: i32,
    bits: i32,
}

impl QEmbedding {
    /// `tokens` is a 1-D int array of token ids.
    fn lookup(&self, tokens: &Array) -> Result<Array> {
        // `.index(&Array)` is infallible; integer-array indexing returns a
        // gather over the leading axis (one row per token id).
        let w_rows = self.weight.index(tokens);
        let s_rows = self.scales.index(tokens);
        let b_rows = self.biases.index(tokens);
        mlx_rs::ops::dequantize(&w_rows, &s_rows, &b_rows, self.group_size, self.bits)
            .map_err(emap)
    }

    /// Use the same quantized table as an LM head (`tie_word_embeddings`).
    /// `x` is `[seq, hidden]`; returns `[seq, vocab]`.
    fn lm_head(&self, x: &Array) -> Result<Array> {
        mlx_rs::ops::quantized_matmul(
            x,
            &self.weight,
            &self.scales,
            &self.biases,
            /* transpose */ true,
            self.group_size,
            self.bits,
        )
        .map_err(|e| Error::Inference(e.to_string()))
    }
}

#[derive(Debug)]
struct Qwen3Layer {
    input_norm: RmsNorm,
    q_proj: QLinear,
    k_proj: QLinear,
    v_proj: QLinear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    o_proj: QLinear,
    post_attn_norm: RmsNorm,
    gate_proj: QLinear,
    up_proj: QLinear,
    down_proj: QLinear,
}

// ─── Model ──────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct Qwen3 {
    pub cfg: ModelConfig,
    embed: QEmbedding,
    layers: Vec<Qwen3Layer>,
    norm: RmsNorm,
    // When `tie_word_embeddings` is true we reuse `embed`; otherwise a
    // dedicated LM head lives here. Quantized either way for MLX models.
    lm_head: Option<QLinear>,
}

impl Qwen3 {
    /// Number of tensors a config implies, used as a sanity check at
    /// `pull` / `inspect` time before we attempt a load.
    pub fn expected_tensor_count(cfg: &ModelConfig) -> usize {
        // Per layer: 4 norms (input + post-attn + q + k) × 1 + 7 quantized
        // projections (q/k/v/o + gate/up/down) × 3 fields (weight/scales/biases)
        //  = 4 + 21 = 25.
        let per_layer = 25;
        let mut global = 1 /* final norm */ + 1 /* embed_tokens.weight */;
        if cfg.quantization.is_some() {
            global += 2; // embed_tokens.scales + biases
        }
        if !cfg.tie_word_embeddings {
            global += if cfg.quantization.is_some() { 3 } else { 1 };
        }
        cfg.num_hidden_layers * per_layer + global
    }

    pub fn load(dir: &Path, cfg: ModelConfig) -> Result<Self> {
        let mut tensors = read_all_shards(dir)?;

        let q = cfg
            .quantization
            .as_ref()
            .ok_or_else(|| Error::ModelLoad("only quantized models supported in v1".into()))?;
        let bits = q.bits as i32;
        let group_size = q.group_size as i32;
        let eps = cfg.rms_norm_eps;

        // ── Embedding (quantized) ──
        let embed = QEmbedding {
            weight: take(&mut tensors, "model.embed_tokens.weight")?,
            scales: take(&mut tensors, "model.embed_tokens.scales")?,
            biases: take(&mut tensors, "model.embed_tokens.biases")?,
            group_size,
            bits,
        };

        // ── Layers ──
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{i}");
            layers.push(Qwen3Layer {
                input_norm: rms(&mut tensors, &format!("{p}.input_layernorm.weight"), eps)?,
                q_proj: qlin(&mut tensors, &format!("{p}.self_attn.q_proj"), bits, group_size)?,
                k_proj: qlin(&mut tensors, &format!("{p}.self_attn.k_proj"), bits, group_size)?,
                v_proj: qlin(&mut tensors, &format!("{p}.self_attn.v_proj"), bits, group_size)?,
                q_norm: rms(&mut tensors, &format!("{p}.self_attn.q_norm.weight"), eps)?,
                k_norm: rms(&mut tensors, &format!("{p}.self_attn.k_norm.weight"), eps)?,
                o_proj: qlin(&mut tensors, &format!("{p}.self_attn.o_proj"), bits, group_size)?,
                post_attn_norm: rms(
                    &mut tensors,
                    &format!("{p}.post_attention_layernorm.weight"),
                    eps,
                )?,
                gate_proj: qlin(&mut tensors, &format!("{p}.mlp.gate_proj"), bits, group_size)?,
                up_proj: qlin(&mut tensors, &format!("{p}.mlp.up_proj"), bits, group_size)?,
                down_proj: qlin(&mut tensors, &format!("{p}.mlp.down_proj"), bits, group_size)?,
            });
        }

        let norm = rms(&mut tensors, "model.norm.weight", eps)?;

        let lm_head = if cfg.tie_word_embeddings {
            None
        } else {
            Some(qlin(&mut tensors, "lm_head", bits, group_size)?)
        };

        // Anything left over is a config-or-mapping bug; flag it now.
        if !tensors.is_empty() {
            let leftover: Vec<_> = tensors.keys().take(5).cloned().collect();
            return Err(Error::ModelLoad(format!(
                "{} unmapped tensors remain (first 5: {:?})",
                tensors.len(),
                leftover
            )));
        }

        Ok(Self {
            cfg,
            embed,
            layers,
            norm,
            lm_head,
        })
    }

    /// Run the prompt through the model and return logits for the last
    /// position. Shape: `[vocab_size]`. No KV cache yet — every call
    /// reprefills the full sequence.
    pub fn forward(&self, tokens: &[u32]) -> Result<Array> {
        let head_dim = self.cfg.per_head_dim() as i32;
        let n_heads = self.cfg.num_attention_heads as i32;
        let kv_heads = self.cfg.kv_heads() as i32;
        let repeats = n_heads / kv_heads;
        let scale = 1.0_f32 / (head_dim as f32).sqrt();
        let seq_len = tokens.len() as i32;

        // tokens → int32 array
        let token_ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let tok_arr = Array::from_slice(&token_ids, &[seq_len]);

        // [seq, hidden]
        let mut x = self.embed.lookup(&tok_arr)?;

        // Causal mask: [seq, seq], lower triangle of 0s, above-diag = -inf.
        // SDPA expects [num_heads, seq_q, seq_k] or broadcastable; we build
        // [1, 1, seq, seq] so it broadcasts over batch and heads.
        let mask = causal_mask(seq_len, x.dtype())?;

        for layer in &self.layers {
            // ── Attention ──
            let h = layer.input_norm.forward(&x)?;

            let q = layer.q_proj.forward(&h)?; // [seq, n_heads*head_dim]
            let k = layer.k_proj.forward(&h)?; // [seq, kv_heads*head_dim]
            let v = layer.v_proj.forward(&h)?;

            // Reshape to [seq, n_heads, head_dim] and apply per-head RMSNorm
            // (Qwen3-specific: applied before RoPE).
            let q = q
                .reshape(&[seq_len, n_heads, head_dim])
                .map_err(emap)?;
            let k = k
                .reshape(&[seq_len, kv_heads, head_dim])
                .map_err(emap)?;
            let v = v
                .reshape(&[seq_len, kv_heads, head_dim])
                .map_err(emap)?;

            let q = layer.q_norm.forward(&q)?;
            let k = layer.k_norm.forward(&k)?;

            // Transpose to [heads, seq, head_dim] BEFORE RoPE so the
            // rotation is applied along the actual sequence axis.
            let q = q.transpose_axes(&[1, 0, 2]).map_err(emap)?;
            let k = k.transpose_axes(&[1, 0, 2]).map_err(emap)?;
            let v = v.transpose_axes(&[1, 0, 2]).map_err(emap)?;

            // RoPE on [heads, seq, head_dim]. offset=0 for a full prefill;
            // KV cache milestone will pass the prior length.
            let q = mlx_rs::fast::rope(&q, head_dim, false, self.cfg.rope_theta, 1.0, 0, None)
                .map_err(emap)?;
            let k = mlx_rs::fast::rope(&k, head_dim, false, self.cfg.rope_theta, 1.0, 0, None)
                .map_err(emap)?;

            // GQA: repeat kv heads along the heads axis (axis 0 now).
            let k = mlx_rs::ops::repeat_axis::<i32>(k, repeats, 0).map_err(emap)?;
            let v = mlx_rs::ops::repeat_axis::<i32>(v, repeats, 0).map_err(emap)?;

            // SDPA expects [batch, heads, seq, dim] — add a unit batch.
            let q = q.expand_dims(0).map_err(emap)?;
            let k = k.expand_dims(0).map_err(emap)?;
            let v = v.expand_dims(0).map_err(emap)?;

            let attn = mlx_rs::fast::scaled_dot_product_attention(&q, &k, &v, scale, &mask)
                .map_err(emap)?;

            // [1, heads, seq, dim] → [seq, heads*dim]
            let attn = attn
                .transpose_axes(&[0, 2, 1, 3])
                .map_err(emap)?
                .reshape(&[seq_len, n_heads * head_dim])
                .map_err(emap)?;

            let attn = layer.o_proj.forward(&attn)?;
            x = &x + &attn;

            // ── MLP (SwiGLU) ──
            let h = layer.post_attn_norm.forward(&x)?;
            let gate = layer.gate_proj.forward(&h)?;
            let up = layer.up_proj.forward(&h)?;
            let gate = mlx_rs::nn::silu(&gate).map_err(emap)?;
            let down = layer.down_proj.forward(&(&gate * &up))?;
            x = &x + &down;
        }

        let x = self.norm.forward(&x)?;

        let logits = match &self.lm_head {
            Some(head) => head.forward(&x)?,
            None => self.embed.lm_head(&x)?,
        };

        // Last token's logits: [vocab].
        Ok(logits.index((seq_len - 1, ..)))
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

fn emap(e: impl std::fmt::Display) -> Error {
    Error::Inference(e.to_string())
}

fn take(map: &mut HashMap<String, Array>, key: &str) -> Result<Array> {
    map.remove(key)
        .ok_or_else(|| Error::ModelLoad(format!("missing tensor: {key}")))
}

fn rms(map: &mut HashMap<String, Array>, key: &str, eps: f32) -> Result<RmsNorm> {
    Ok(RmsNorm {
        weight: take(map, key)?,
        eps,
    })
}

fn qlin(
    map: &mut HashMap<String, Array>,
    prefix: &str,
    bits: i32,
    group_size: i32,
) -> Result<QLinear> {
    Ok(QLinear {
        weight: take(map, &format!("{prefix}.weight"))?,
        scales: take(map, &format!("{prefix}.scales"))?,
        biases: take(map, &format!("{prefix}.biases"))?,
        group_size,
        bits,
    })
}

fn read_all_shards(dir: &Path) -> Result<HashMap<String, Array>> {
    let mut shards: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
        .collect();
    shards.sort();
    if shards.is_empty() {
        return Err(Error::ModelLoad(format!(
            "no .safetensors in {}",
            dir.display()
        )));
    }
    let mut out = HashMap::new();
    for shard in shards {
        let map = Array::load_safetensors(&shard)
            .map_err(|e| Error::ModelLoad(format!("load {}: {e}", shard.display())))?;
        out.extend(map);
    }
    Ok(out)
}

fn causal_mask(seq_len: i32, dtype: Dtype) -> Result<Array> {
    // Build a [seq, seq] mask with 0 on/below diag, -1e9 above.
    let ones = Array::ones::<f32>(&[seq_len, seq_len]).map_err(emap)?;
    let lower = mlx_rs::ops::tril(&ones, 0).map_err(emap)?;
    let neg_inf = Array::full::<f32>(&[seq_len, seq_len], &Array::from_f32(-1.0e9)).map_err(emap)?;
    // mask = lower==1 ? 0 : -1e9
    let zeros = Array::zeros::<f32>(&[seq_len, seq_len]).map_err(emap)?;
    let cond = lower.eq(&Array::from_f32(1.0)).map_err(emap)?;
    let m = mlx_rs::ops::r#where(&cond, &zeros, &neg_inf).map_err(emap)?;
    let m = m.as_dtype(dtype).map_err(emap)?;
    // [seq, seq] → [1, 1, seq, seq] for broadcasting in SDPA.
    let m = m.expand_dims(0).map_err(emap)?.expand_dims(0).map_err(emap)?;
    Ok(m)
}
