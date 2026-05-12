//! Qwen3 (dense, Instruct variant).
//!
//! v1 implementation: full reprefill on every step (no KV cache yet).
//! Correctness first; cache + speculative decoding land as separate
//! milestones once a single forward pass produces sensible logits.

use super::kernels::{mlp_block, qkv_block};
use super::ModelConfig;
use crate::{Error, Result};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::transforms::compile::compile;
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

/// Per-layer K/V cache. Each entry is pre-allocated to
/// `[kv_heads, KV_PREALLOC, head_dim]` once at first use; we write new
/// rows in place via `index_mut` rather than concatenating a fresh
/// Metal buffer every step. RoPE was already applied at the position
/// the tokens were first seen, so cached entries don't get re-rotated.
#[derive(Debug, Default)]
pub struct KvCache {
    pub k: Vec<mlx_rs::Array>,
    pub v: Vec<mlx_rs::Array>,
    pub seq_len: usize,
}

/// Max context we'll cache before erroring. 4096 covers the typical
/// chat / agent flow; we can bump or make configurable later.
pub const KV_PREALLOC: i32 = 4096;

impl KvCache {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn reset(&mut self) {
        self.k.clear();
        self.v.clear();
        self.seq_len = 0;
    }
    pub fn is_initialized(&self) -> bool {
        !self.k.is_empty()
    }

    /// Roll the cache back to the first `target_seq_len` positions.
    /// No-op if already at or below that length. Used by speculative
    /// decoding when the target rejects some draft tokens: both caches
    /// processed K positions, we keep the accepted prefix and drop the
    /// rest so the next iteration's verify input lines up.
    ///
    /// Each layer's k/v has shape `[kv_heads, seq, head_dim]`; we slice
    /// axis 1 to `[..target_seq_len]`. mlx-rs's slice is functional —
    /// it returns a new array sharing the underlying buffer, so this is
    /// cheap (no copy). Old views drop when the Vec slot is overwritten.
    pub fn truncate(&mut self, target_seq_len: usize) {
        if target_seq_len >= self.seq_len {
            return;
        }
        let t = target_seq_len as i32;
        for (k, v) in self.k.iter_mut().zip(self.v.iter_mut()) {
            // k/v shape: [kv_heads, seq, head_dim]. Slice axis 1 → [.., 0..t, ..].
            *k = k.index((.., 0..t, ..));
            *v = v.index((.., 0..t, ..));
        }
        self.seq_len = target_seq_len;
    }
}

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

    /// Run `tokens` through the model and return logits for the *last*
    /// position. Shape: `[vocab_size]`. `cache` is extended in place:
    ///   - First call (empty cache): full prefill on the whole prompt.
    ///   - Subsequent calls: pass *only the new tokens* (typically one
    ///     during greedy decode) — cached K/V handle the rest.
    pub fn forward(&self, tokens: &[u32], cache: &mut KvCache) -> Result<Array> {
        let all = self.forward_internal(tokens, cache)?;
        let seq_len = tokens.len() as i32;
        Ok(all.index((seq_len - 1, ..)))
    }

    /// Like `forward`, but returns logits at *every* input position.
    /// Shape: `[seq_len, vocab_size]`. Used by speculative decoding's
    /// verify step — we feed K candidate tokens at once and need each
    /// position's logits to compare against the draft's proposals.
    pub fn forward_multi(&self, tokens: &[u32], cache: &mut KvCache) -> Result<Array> {
        self.forward_internal(tokens, cache)
    }

    fn forward_internal(&self, tokens: &[u32], cache: &mut KvCache) -> Result<Array> {
        let head_dim = self.cfg.per_head_dim() as i32;
        let n_heads = self.cfg.num_attention_heads as i32;
        let kv_heads = self.cfg.kv_heads() as i32;
        let repeats = n_heads / kv_heads;
        let scale = 1.0_f32 / (head_dim as f32).sqrt();
        let seq_len = tokens.len() as i32;
        let offset = cache.seq_len as i32;

        // tokens → int32 array
        let token_ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let tok_arr = Array::from_slice(&token_ids, &[seq_len]);

        // [seq, hidden]
        let mut x = self.embed.lookup(&tok_arr)?;

        // Mask. Three regimes:
        //   - single-token decode (seq_len==1): no mask needed.
        //   - prefill on empty cache (offset==0, seq_len>1): square causal
        //     [seq, seq] mask.
        //   - multi-token forward on a non-empty cache (offset>0, seq_len>1):
        //     non-square [seq, offset+seq] mask. The first `offset` columns
        //     are 0 (each new query attends freely to all cached keys); the
        //     trailing `seq` columns are causal-triangular among the new
        //     tokens. This regime is hit by speculative decoding's verify
        //     step, where target ingests K candidate tokens after prefill.
        let mask = if seq_len > 1 {
            Some(causal_mask_offset(seq_len, offset, x.dtype())?)
        } else {
            None
        };

        let extending = cache.is_initialized();
        if extending && cache.k.len() != self.layers.len() {
            return Err(Error::Inference(format!(
                "cache layer count {} != model layers {}",
                cache.k.len(),
                self.layers.len()
            )));
        }

        // MLX op fusion: compile cache hits per shape × function-pointer
        // identity, so each layer's call lands on the same fused graph
        // after the first.
        let mut compiled_qkv = compile(qkv_block, true);
        let mut compiled_mlp = compile(mlp_block, true);

        for (li, layer) in self.layers.iter().enumerate() {
            // ── Attention: fused input_norm + q/k/v projections ──
            let qkv_inputs = [
                x.clone(),
                layer.input_norm.weight.clone(),
                layer.q_proj.weight.clone(),
                layer.q_proj.scales.clone(),
                layer.q_proj.biases.clone(),
                layer.k_proj.weight.clone(),
                layer.k_proj.scales.clone(),
                layer.k_proj.biases.clone(),
                layer.v_proj.weight.clone(),
                layer.v_proj.scales.clone(),
                layer.v_proj.biases.clone(),
            ];
            let qkv = compiled_qkv(&qkv_inputs).map_err(emap)?;
            let q = qkv[0].clone();
            let k = qkv[1].clone();
            let v = qkv[2].clone();

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

            // RoPE on [heads, seq, head_dim] with the right offset so
            // cached keys (rotated at their original positions) line up
            // with newly-rotated queries.
            let q = mlx_rs::fast::rope(&q, head_dim, false, self.cfg.rope_theta, 1.0, offset, None)
                .map_err(emap)?;
            let k = mlx_rs::fast::rope(&k, head_dim, false, self.cfg.rope_theta, 1.0, offset, None)
                .map_err(emap)?;

            // Extend or seed the per-layer K/V cache via in-place
            // slice_update against a pre-allocated [kv_heads, KV_PREALLOC,
            // head_dim] buffer. MLX's runtime planner reuses the input
            // buffer when its refcount is 1 — which is the case right
            // after we reassign `cache.k[li]` and the prior SDPA view has
            // dropped. Net effect: zero allocation per decode step, no
            // O(N) concat, no fragmentation. The per-step cost stops
            // growing with context length.
            let new_end = offset + seq_len;
            if new_end > KV_PREALLOC {
                return Err(Error::Inference(format!(
                    "KV cache exhausted: requested {new_end} positions, max {KV_PREALLOC}"
                )));
            }
            let kv_heads_dim = kv_heads;
            let head_dim_v = head_dim;
            let (k_cached, v_cached) = if extending {
                let kk = crate::mlx_ext::slice_update(
                    &cache.k[li],
                    &k,
                    &[0, offset, 0],
                    &[kv_heads_dim, new_end, head_dim_v],
                    &[1, 1, 1],
                )
                .map_err(|e| Error::Inference(e))?;
                let vv = crate::mlx_ext::slice_update(
                    &cache.v[li],
                    &v,
                    &[0, offset, 0],
                    &[kv_heads_dim, new_end, head_dim_v],
                    &[1, 1, 1],
                )
                .map_err(|e| Error::Inference(e))?;
                cache.k[li] = kk;
                cache.v[li] = vv;
                // Read view of the valid portion. The slice op is a
                // metadata-only view; SDPA reads through it without
                // copying. View drops at end of layer iteration so the
                // next step's slice_update sees refcount=1 on the cache.
                let kv = cache.k[li].index((.., 0..new_end, ..));
                let vv_view = cache.v[li].index((.., 0..new_end, ..));
                (kv, vv_view)
            } else {
                // First call this conversation: seed the cache by writing
                // the prefill into a fresh zero buffer of [kv_heads,
                // KV_PREALLOC, head_dim]. Buffer is allocated once and
                // reused for the whole conversation.
                let zeros_k = mlx_rs::ops::zeros_dtype(&[kv_heads_dim, KV_PREALLOC, head_dim_v], k.dtype())
                    .map_err(emap)?;
                let zeros_v = mlx_rs::ops::zeros_dtype(&[kv_heads_dim, KV_PREALLOC, head_dim_v], v.dtype())
                    .map_err(emap)?;
                let kk = crate::mlx_ext::slice_update(
                    &zeros_k,
                    &k,
                    &[0, 0, 0],
                    &[kv_heads_dim, seq_len, head_dim_v],
                    &[1, 1, 1],
                )
                .map_err(|e| Error::Inference(e))?;
                let vv = crate::mlx_ext::slice_update(
                    &zeros_v,
                    &v,
                    &[0, 0, 0],
                    &[kv_heads_dim, seq_len, head_dim_v],
                    &[1, 1, 1],
                )
                .map_err(|e| Error::Inference(e))?;
                cache.k.push(kk);
                cache.v.push(vv);
                let kv = cache.k[li].index((.., 0..seq_len, ..));
                let vv_view = cache.v[li].index((.., 0..seq_len, ..));
                (kv, vv_view)
            };

            // MLX SDPA handles GQA natively — q has n_heads, k/v have
            // kv_heads; the kernel handles the implicit head replication
            // without us materializing an expanded tensor. Avoids the
            // per-step kv-expansion copy.
            let _ = repeats;
            let q = q.expand_dims(0).map_err(emap)?;
            let k = k_cached.expand_dims(0).map_err(emap)?;
            let v = v_cached.expand_dims(0).map_err(emap)?;

            let attn = match &mask {
                Some(m) => mlx_rs::fast::scaled_dot_product_attention(&q, &k, &v, scale, m)
                    .map_err(emap)?,
                None => mlx_rs::fast::scaled_dot_product_attention(
                    &q,
                    &k,
                    &v,
                    scale,
                    Option::<mlx_rs::fast::ScaledDotProductAttentionMask<'_>>::None,
                )
                .map_err(emap)?,
            };

            // [1, heads, seq, dim] → [seq, heads*dim]
            let attn = attn
                .transpose_axes(&[0, 2, 1, 3])
                .map_err(emap)?
                .reshape(&[seq_len, n_heads * head_dim])
                .map_err(emap)?;

            let attn = layer.o_proj.forward(&attn)?;
            x = &x + &attn;

            // MLP fused: post_norm → gate/up_proj → silu·up → down_proj
            let mlp_inputs = [
                x.clone(),
                layer.post_attn_norm.weight.clone(),
                layer.gate_proj.weight.clone(),
                layer.gate_proj.scales.clone(),
                layer.gate_proj.biases.clone(),
                layer.up_proj.weight.clone(),
                layer.up_proj.scales.clone(),
                layer.up_proj.biases.clone(),
                layer.down_proj.weight.clone(),
                layer.down_proj.scales.clone(),
                layer.down_proj.biases.clone(),
            ];
            let mlp_out = compiled_mlp(&mlp_inputs).map_err(emap)?;
            x = &x + &mlp_out[0];
            let _ = li;
        }

        let x = self.norm.forward(&x)?;

        let logits = match &self.lm_head {
            Some(head) => head.forward(&x)?,
            None => self.embed.lm_head(&x)?,
        };

        cache.seq_len += seq_len as usize;
        // Full [seq, vocab] tensor — callers slice as needed.
        Ok(logits)
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

fn causal_mask_offset(seq_len: i32, offset: i32, dtype: Dtype) -> Result<Array> {
    // Build a [seq, offset+seq] mask. The first `offset` columns are 0
    // (each query attends freely to all cached keys); the trailing `seq`
    // columns are causal-triangular among the new tokens. Reduces to the
    // square [seq, seq] case when offset == 0.
    //
    // Trick: `tril(ones, k=offset)` over a [seq, offset+seq] grid sets 1
    // on/below the shifted diagonal — i.e. exactly the cells we want
    // unmasked. We then map 1→0 and 0→-1e9 for additive masking.
    let k_len = offset + seq_len;
    let ones = Array::ones::<f32>(&[seq_len, k_len]).map_err(emap)?;
    let allowed = mlx_rs::ops::tril(&ones, offset).map_err(emap)?;
    let zeros = Array::zeros::<f32>(&[seq_len, k_len]).map_err(emap)?;
    let neg_inf = Array::full::<f32>(&[seq_len, k_len], &Array::from_f32(-1.0e9)).map_err(emap)?;
    let cond = allowed.eq(&Array::from_f32(1.0)).map_err(emap)?;
    let m = mlx_rs::ops::r#where(&cond, &zeros, &neg_inf).map_err(emap)?;
    let m = m.as_dtype(dtype).map_err(emap)?;
    // [seq, k_len] → [1, 1, seq, k_len] for broadcasting in SDPA.
    let m = m.expand_dims(0).map_err(emap)?.expand_dims(0).map_err(emap)?;
    Ok(m)
}
