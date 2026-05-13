//! Gemma 4 text tower.
//!
//! Gemma 4 ships as a multimodal checkpoint, but chat generation only needs
//! the `language_model.*` tensors. This implementation mirrors mlx-lm's
//! Gemma4Text path for the dense 2B/4B family: per-layer input embeddings,
//! shared K/V layers, mixed sliding/full attention, GeGLU MLPs, and logit
//! soft-capping.

use super::ModelConfig;
use crate::{Error, Result};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::{Array, Dtype};
use std::collections::HashMap;
use std::path::Path;

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
            true,
            self.group_size,
            self.bits,
        )
        .map_err(emap)
    }
}

#[derive(Debug)]
struct RmsNorm {
    weight: Array,
    eps: f32,
}

impl RmsNorm {
    fn forward(&self, x: &Array) -> Result<Array> {
        mlx_rs::fast::rms_norm(x, &self.weight, self.eps).map_err(emap)
    }
}

#[derive(Debug)]
struct RmsNormNoScale {
    eps: f32,
}

impl RmsNormNoScale {
    fn forward(&self, x: &Array) -> Result<Array> {
        let sq = x.square().map_err(emap)?;
        let mean = mlx_rs::ops::mean_axis(&sq, -1, true).map_err(emap)?;
        let denom = (&mean + &Array::from_f32(self.eps)).rsqrt().map_err(emap)?;
        Ok(x * &denom)
    }
}

#[derive(Debug)]
struct QEmbedding {
    weight: Array,
    scales: Array,
    biases: Array,
    group_size: i32,
    bits: i32,
}

impl QEmbedding {
    fn lookup(&self, tokens: &Array) -> Result<Array> {
        let w_rows = self.weight.index(tokens);
        let s_rows = self.scales.index(tokens);
        let b_rows = self.biases.index(tokens);
        mlx_rs::ops::dequantize(&w_rows, &s_rows, &b_rows, self.group_size, self.bits).map_err(emap)
    }

    fn lm_head(&self, x: &Array) -> Result<Array> {
        mlx_rs::ops::quantized_matmul(
            x,
            &self.weight,
            &self.scales,
            &self.biases,
            true,
            self.group_size,
            self.bits,
        )
        .map_err(emap)
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum LayerType {
    Sliding,
    Full,
}

impl LayerType {
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "sliding_attention" => Ok(Self::Sliding),
            "full_attention" => Ok(Self::Full),
            other => Err(Error::InvalidConfig(format!(
                "unsupported Gemma4 layer type: {other}"
            ))),
        }
    }
}

#[derive(Debug)]
struct Gemma4Layer {
    layer_type: LayerType,
    has_kv: bool,
    previous_kv: usize,
    head_dim: i32,
    kv_heads: i32,
    input_norm: RmsNorm,
    q_proj: QLinear,
    k_proj: Option<QLinear>,
    v_proj: Option<QLinear>,
    q_norm: RmsNorm,
    k_norm: Option<RmsNorm>,
    v_norm: RmsNormNoScale,
    o_proj: QLinear,
    post_attn_norm: RmsNorm,
    pre_ffn_norm: RmsNorm,
    post_ffn_norm: RmsNorm,
    gate_proj: QLinear,
    up_proj: QLinear,
    down_proj: QLinear,
    per_layer_input_gate: QLinear,
    per_layer_projection: QLinear,
    post_per_layer_input_norm: RmsNorm,
    layer_scalar: Array,
}

#[derive(Debug, Default)]
pub struct KvCache {
    k: Vec<Option<Array>>,
    v: Vec<Option<Array>>,
    pub seq_len: usize,
}

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

    pub fn truncate(&mut self, target_seq_len: usize) {
        if target_seq_len >= self.seq_len {
            return;
        }
        let t = target_seq_len as i32;
        for (k, v) in self.k.iter_mut().zip(self.v.iter_mut()) {
            if let Some(kk) = k {
                *kk = kk.index((.., 0..t, ..));
            }
            if let Some(vv) = v {
                *vv = vv.index((.., 0..t, ..));
            }
        }
        self.seq_len = target_seq_len;
    }

    fn ensure_layers(&mut self, layers: usize) {
        if self.k.len() < layers {
            self.k.resize_with(layers, || None);
            self.v.resize_with(layers, || None);
        }
    }
}

#[derive(Debug)]
pub struct Gemma4 {
    pub cfg: ModelConfig,
    embed: QEmbedding,
    embed_per_layer: QEmbedding,
    per_layer_model_projection: QLinear,
    per_layer_projection_norm: RmsNorm,
    layers: Vec<Gemma4Layer>,
    norm: RmsNorm,
    embed_scale: f32,
    embed_per_layer_scale: f32,
    per_layer_projection_scale: f32,
    per_layer_input_scale: f32,
    sliding_rope_theta: f32,
    full_rope_freqs: Array,
    final_logit_softcapping: Option<f32>,
    sliding_window: i32,
}

impl Gemma4 {
    pub fn load(dir: &Path, cfg: ModelConfig) -> Result<Self> {
        if cfg.enable_moe_block {
            return Err(Error::ModelLoad(
                "Gemma4 MoE/A4B checkpoints are not supported yet; use dense E2B/E4B Gemma4 or LM Studio for this model".into(),
            ));
        }
        if cfg.attention_k_eq_v {
            return Err(Error::ModelLoad(
                "Gemma4 K-eq-V attention is not supported yet".into(),
            ));
        }
        if cfg.hidden_size_per_layer_input.unwrap_or(0) == 0 {
            return Err(Error::ModelLoad(
                "Gemma4 checkpoint has no per-layer input embeddings; this loader currently supports dense E2B/E4B only".into(),
            ));
        }

        let mut tensors = read_all_shards(dir)?;

        let q = cfg
            .quantization
            .as_ref()
            .ok_or_else(|| Error::ModelLoad("only quantized Gemma4 models supported".into()))?;
        let bits = q.bits as i32;
        let group_size = q.group_size as i32;
        let eps = cfg.rms_norm_eps;
        let prefix = "language_model.model";

        let embed = qembed(
            &mut tensors,
            &format!("{prefix}.embed_tokens"),
            bits,
            group_size,
        )?;
        let embed_per_layer = qembed(
            &mut tensors,
            &format!("{prefix}.embed_tokens_per_layer"),
            bits,
            group_size,
        )?;
        let per_layer_model_projection = qlin(
            &mut tensors,
            &format!("{prefix}.per_layer_model_projection"),
            bits,
            group_size,
        )?;
        let hidden_per_layer = cfg.hidden_size_per_layer_input.unwrap_or(256);
        let per_layer_projection_norm = rms(
            &mut tensors,
            &format!("{prefix}.per_layer_projection_norm.weight"),
            eps,
        )?;

        let layer_types = layer_types(&cfg)?;
        let first_kv_shared = cfg
            .num_kv_shared_layers
            .map(|n| cfg.num_hidden_layers.saturating_sub(n))
            .unwrap_or(cfg.num_hidden_layers);
        let previous_kvs = previous_kvs(&layer_types, first_kv_shared)?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let p = format!("{prefix}.layers.{i}");
            let layer_type = layer_types[i];
            let has_kv = i < first_kv_shared;
            let head_dim = match layer_type {
                LayerType::Sliding => cfg.per_head_dim(),
                LayerType::Full => cfg.global_head_dim.unwrap_or(cfg.per_head_dim()),
            } as i32;

            layers.push(Gemma4Layer {
                layer_type,
                has_kv,
                previous_kv: previous_kvs[i],
                head_dim,
                kv_heads: cfg.kv_heads() as i32,
                input_norm: rms(&mut tensors, &format!("{p}.input_layernorm.weight"), eps)?,
                q_proj: qlin(
                    &mut tensors,
                    &format!("{p}.self_attn.q_proj"),
                    bits,
                    group_size,
                )?,
                k_proj: if has_kv {
                    Some(qlin(
                        &mut tensors,
                        &format!("{p}.self_attn.k_proj"),
                        bits,
                        group_size,
                    )?)
                } else {
                    None
                },
                v_proj: if has_kv {
                    Some(qlin(
                        &mut tensors,
                        &format!("{p}.self_attn.v_proj"),
                        bits,
                        group_size,
                    )?)
                } else {
                    None
                },
                q_norm: rms(&mut tensors, &format!("{p}.self_attn.q_norm.weight"), eps)?,
                k_norm: if has_kv {
                    Some(rms(
                        &mut tensors,
                        &format!("{p}.self_attn.k_norm.weight"),
                        eps,
                    )?)
                } else {
                    None
                },
                v_norm: RmsNormNoScale { eps },
                o_proj: qlin(
                    &mut tensors,
                    &format!("{p}.self_attn.o_proj"),
                    bits,
                    group_size,
                )?,
                post_attn_norm: rms(
                    &mut tensors,
                    &format!("{p}.post_attention_layernorm.weight"),
                    eps,
                )?,
                pre_ffn_norm: rms(
                    &mut tensors,
                    &format!("{p}.pre_feedforward_layernorm.weight"),
                    eps,
                )?,
                post_ffn_norm: rms(
                    &mut tensors,
                    &format!("{p}.post_feedforward_layernorm.weight"),
                    eps,
                )?,
                gate_proj: qlin(
                    &mut tensors,
                    &format!("{p}.mlp.gate_proj"),
                    bits,
                    group_size,
                )?,
                up_proj: qlin(&mut tensors, &format!("{p}.mlp.up_proj"), bits, group_size)?,
                down_proj: qlin(
                    &mut tensors,
                    &format!("{p}.mlp.down_proj"),
                    bits,
                    group_size,
                )?,
                per_layer_input_gate: qlin(
                    &mut tensors,
                    &format!("{p}.per_layer_input_gate"),
                    bits,
                    group_size,
                )?,
                per_layer_projection: qlin(
                    &mut tensors,
                    &format!("{p}.per_layer_projection"),
                    bits,
                    group_size,
                )?,
                post_per_layer_input_norm: rms(
                    &mut tensors,
                    &format!("{p}.post_per_layer_input_norm.weight"),
                    eps,
                )?,
                layer_scalar: take(&mut tensors, &format!("{p}.layer_scalar"))?,
            });
        }

        let norm = rms(&mut tensors, &format!("{prefix}.norm.weight"), eps)?;
        let full_head_dim = cfg.global_head_dim.unwrap_or(cfg.per_head_dim());
        let full_partial = rope_f32(&cfg, "full_attention", "partial_rotary_factor", 0.25);
        let full_theta = rope_f32(&cfg, "full_attention", "rope_theta", 1_000_000.0);
        let sliding_theta = rope_f32(&cfg, "sliding_attention", "rope_theta", 10_000.0);

        Ok(Self {
            embed,
            embed_per_layer,
            per_layer_model_projection,
            per_layer_projection_norm,
            layers,
            norm,
            embed_scale: (cfg.hidden_size as f32).sqrt(),
            embed_per_layer_scale: (hidden_per_layer as f32).sqrt(),
            per_layer_projection_scale: (cfg.hidden_size as f32).powf(-0.5),
            per_layer_input_scale: 2.0_f32.powf(-0.5),
            sliding_rope_theta: sliding_theta,
            full_rope_freqs: proportional_rope_freqs(full_head_dim, full_partial, full_theta),
            final_logit_softcapping: cfg.final_logit_softcapping,
            sliding_window: cfg.sliding_window.unwrap_or(512) as i32,
            cfg,
        })
    }

    pub fn forward(&self, tokens: &[u32], cache: &mut KvCache) -> Result<Array> {
        self.forward_internal(tokens, cache, false)
    }

    pub fn forward_multi(&self, tokens: &[u32], cache: &mut KvCache) -> Result<Array> {
        self.forward_internal(tokens, cache, true)
    }

    fn forward_internal(
        &self,
        tokens: &[u32],
        cache: &mut KvCache,
        return_all_logits: bool,
    ) -> Result<Array> {
        let seq_len = tokens.len() as i32;
        let offset = cache.seq_len as i32;
        let new_end = offset + seq_len;
        if new_end > KV_PREALLOC {
            return Err(Error::Inference(format!(
                "KV cache exhausted: requested {new_end} positions, max {KV_PREALLOC}"
            )));
        }

        cache.ensure_layers(self.layers.len());

        let token_ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let tok_arr = Array::from_slice(&token_ids, &[seq_len]);

        let mut x = self.embed.lookup(&tok_arr)?;
        x = &x * &Array::from_f32(self.embed_scale);

        let per_layer_inputs = self.per_layer_inputs(&tok_arr, &x, seq_len)?;

        let full_mask = attention_mask(seq_len, offset, None, x.dtype())?;
        let sliding_mask = attention_mask(seq_len, offset, Some(self.sliding_window), x.dtype())?;

        let mut intermediates: Vec<Option<(Array, Array)>> = vec![None; self.layers.len()];

        for (li, layer) in self.layers.iter().enumerate() {
            let residual = x.clone();
            let h = layer.input_norm.forward(&x)?;
            let (attn, kv_pair) = self.attention(
                layer,
                li,
                &h,
                cache,
                &intermediates,
                match layer.layer_type {
                    LayerType::Sliding => sliding_mask.as_ref(),
                    LayerType::Full => full_mask.as_ref(),
                },
                seq_len,
                offset,
                new_end,
            )?;
            intermediates[li] = Some(kv_pair);
            let h = layer.post_attn_norm.forward(&attn)?;
            x = &residual + &h;

            let residual = x.clone();
            let h = layer.pre_ffn_norm.forward(&x)?;
            let gate = layer.gate_proj.forward(&h)?;
            let up = layer.up_proj.forward(&h)?;
            let activated = mlx_rs::nn::gelu_approximate(&gate).map_err(emap)?;
            let h = &activated * &up;
            let h = layer.down_proj.forward(&h)?;
            let h = layer.post_ffn_norm.forward(&h)?;
            x = &residual + &h;

            let residual = x.clone();
            let per_layer_input = per_layer_inputs.index((.., li as i32, ..));
            let gate = layer.per_layer_input_gate.forward(&x)?;
            let gate = mlx_rs::nn::gelu_approximate(&gate).map_err(emap)?;
            let gate = &gate * &per_layer_input;
            let gate = layer.per_layer_projection.forward(&gate)?;
            let gate = layer.post_per_layer_input_norm.forward(&gate)?;
            x = &residual + &gate;
            x = &x * &layer.layer_scalar;
        }

        let x = self.norm.forward(&x)?;
        let x = if return_all_logits {
            x
        } else {
            x.index((seq_len - 1, ..)).expand_dims(0).map_err(emap)?
        };

        let mut logits = self.embed.lm_head(&x)?;
        if let Some(cap) = self.final_logit_softcapping {
            logits = logit_softcap(&logits, cap)?;
        }

        cache.seq_len += seq_len as usize;
        if return_all_logits {
            Ok(logits)
        } else {
            Ok(logits.index((0, ..)))
        }
    }

    fn per_layer_inputs(&self, tok_arr: &Array, x: &Array, seq_len: i32) -> Result<Array> {
        let mut by_token = self.embed_per_layer.lookup(tok_arr)?;
        by_token = &by_token * &Array::from_f32(self.embed_per_layer_scale);
        by_token = by_token
            .reshape(&[
                seq_len,
                self.cfg.num_hidden_layers as i32,
                self.cfg.hidden_size_per_layer_input.unwrap_or(256) as i32,
            ])
            .map_err(emap)?;

        let mut projected = self.per_layer_model_projection.forward(x)?;
        projected = &projected * &Array::from_f32(self.per_layer_projection_scale);
        projected = projected
            .reshape(&[
                seq_len,
                self.cfg.num_hidden_layers as i32,
                self.cfg.hidden_size_per_layer_input.unwrap_or(256) as i32,
            ])
            .map_err(emap)?;
        projected = self.per_layer_projection_norm.forward(&projected)?;

        Ok(&(&projected + &by_token) * &Array::from_f32(self.per_layer_input_scale))
    }

    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        layer: &Gemma4Layer,
        li: usize,
        x: &Array,
        cache: &mut KvCache,
        intermediates: &[Option<(Array, Array)>],
        mask: Option<&Array>,
        seq_len: i32,
        offset: i32,
        new_end: i32,
    ) -> Result<(Array, (Array, Array))> {
        let n_heads = self.cfg.num_attention_heads as i32;
        let head_dim = layer.head_dim;
        let kv_heads = layer.kv_heads;

        let mut q = layer.q_proj.forward(x)?;
        q = q.reshape(&[seq_len, n_heads, head_dim]).map_err(emap)?;
        q = layer.q_norm.forward(&q)?;
        q = q.transpose_axes(&[1, 0, 2]).map_err(emap)?;
        q = self.apply_rope(layer.layer_type, &q, head_dim, offset)?;

        let (k_cached, v_cached) = if layer.has_kv {
            let k_proj = layer
                .k_proj
                .as_ref()
                .ok_or_else(|| Error::Inference("missing Gemma4 k_proj".into()))?;
            let v_proj = layer
                .v_proj
                .as_ref()
                .ok_or_else(|| Error::Inference("missing Gemma4 v_proj".into()))?;
            let k_norm = layer
                .k_norm
                .as_ref()
                .ok_or_else(|| Error::Inference("missing Gemma4 k_norm".into()))?;

            let mut k = k_proj.forward(x)?;
            let mut v = v_proj.forward(x)?;
            k = k.reshape(&[seq_len, kv_heads, head_dim]).map_err(emap)?;
            v = v.reshape(&[seq_len, kv_heads, head_dim]).map_err(emap)?;
            k = k_norm.forward(&k)?;
            v = layer.v_norm.forward(&v)?;
            k = k.transpose_axes(&[1, 0, 2]).map_err(emap)?;
            v = v.transpose_axes(&[1, 0, 2]).map_err(emap)?;
            k = self.apply_rope(layer.layer_type, &k, head_dim, offset)?;
            self.update_cache(cache, li, &k, &v, kv_heads, head_dim, new_end, seq_len)?
        } else {
            intermediates
                .get(layer.previous_kv)
                .and_then(Option::as_ref)
                .cloned()
                .ok_or_else(|| {
                    Error::Inference(format!(
                        "Gemma4 layer {li} missing shared kv from {}",
                        layer.previous_kv
                    ))
                })?
        };

        let q = q.expand_dims(0).map_err(emap)?;
        let k = k_cached.expand_dims(0).map_err(emap)?;
        let v = v_cached.expand_dims(0).map_err(emap)?;

        let attn = match mask {
            Some(m) => {
                mlx_rs::fast::scaled_dot_product_attention(&q, &k, &v, 1.0, m).map_err(emap)?
            }
            None => mlx_rs::fast::scaled_dot_product_attention(
                &q,
                &k,
                &v,
                1.0,
                Option::<mlx_rs::fast::ScaledDotProductAttentionMask<'_>>::None,
            )
            .map_err(emap)?,
        };

        let attn = attn
            .transpose_axes(&[0, 2, 1, 3])
            .map_err(emap)?
            .reshape(&[seq_len, n_heads * head_dim])
            .map_err(emap)?;
        let out = layer.o_proj.forward(&attn)?;

        Ok((out, (k_cached, v_cached)))
    }

    #[allow(clippy::too_many_arguments)]
    fn update_cache(
        &self,
        cache: &mut KvCache,
        li: usize,
        k: &Array,
        v: &Array,
        kv_heads: i32,
        head_dim: i32,
        new_end: i32,
        seq_len: i32,
    ) -> Result<(Array, Array)> {
        if cache.k[li].is_some() {
            let kk = crate::mlx_ext::slice_update(
                cache.k[li].as_ref().unwrap(),
                k,
                &[0, cache.seq_len as i32, 0],
                &[kv_heads, new_end, head_dim],
                &[1, 1, 1],
            )
            .map_err(Error::Inference)?;
            let vv = crate::mlx_ext::slice_update(
                cache.v[li].as_ref().unwrap(),
                v,
                &[0, cache.seq_len as i32, 0],
                &[kv_heads, new_end, head_dim],
                &[1, 1, 1],
            )
            .map_err(Error::Inference)?;
            cache.k[li] = Some(kk);
            cache.v[li] = Some(vv);
        } else {
            let zeros_k = mlx_rs::ops::zeros_dtype(&[kv_heads, KV_PREALLOC, head_dim], k.dtype())
                .map_err(emap)?;
            let zeros_v = mlx_rs::ops::zeros_dtype(&[kv_heads, KV_PREALLOC, head_dim], v.dtype())
                .map_err(emap)?;
            let kk = crate::mlx_ext::slice_update(
                &zeros_k,
                k,
                &[0, 0, 0],
                &[kv_heads, seq_len, head_dim],
                &[1, 1, 1],
            )
            .map_err(Error::Inference)?;
            let vv = crate::mlx_ext::slice_update(
                &zeros_v,
                v,
                &[0, 0, 0],
                &[kv_heads, seq_len, head_dim],
                &[1, 1, 1],
            )
            .map_err(Error::Inference)?;
            cache.k[li] = Some(kk);
            cache.v[li] = Some(vv);
        }

        let kk = cache.k[li].as_ref().unwrap().index((.., 0..new_end, ..));
        let vv = cache.v[li].as_ref().unwrap().index((.., 0..new_end, ..));
        Ok((kk, vv))
    }

    fn apply_rope(
        &self,
        layer_type: LayerType,
        x: &Array,
        head_dim: i32,
        offset: i32,
    ) -> Result<Array> {
        match layer_type {
            LayerType::Sliding => mlx_rs::fast::rope(
                x,
                head_dim,
                false,
                self.sliding_rope_theta,
                1.0,
                offset,
                None,
            )
            .map_err(emap),
            LayerType::Full => mlx_rs::fast::rope(
                x,
                head_dim,
                false,
                Option::<f32>::None,
                1.0,
                offset,
                &self.full_rope_freqs,
            )
            .map_err(emap),
        }
    }
}

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
    let weight = take(map, &format!("{prefix}.weight"))?;
    let scales = take(map, &format!("{prefix}.scales"))?;
    let biases = take(map, &format!("{prefix}.biases"))?;
    let bits = infer_bits(&weight, &scales, bits, group_size);
    Ok(QLinear {
        weight,
        scales,
        biases,
        group_size,
        bits,
    })
}

fn qembed(
    map: &mut HashMap<String, Array>,
    prefix: &str,
    bits: i32,
    group_size: i32,
) -> Result<QEmbedding> {
    let weight = take(map, &format!("{prefix}.weight"))?;
    let scales = take(map, &format!("{prefix}.scales"))?;
    let biases = take(map, &format!("{prefix}.biases"))?;
    let bits = infer_bits(&weight, &scales, bits, group_size);
    Ok(QEmbedding {
        weight,
        scales,
        biases,
        group_size,
        bits,
    })
}

fn infer_bits(weight: &Array, scales: &Array, fallback: i32, group_size: i32) -> i32 {
    let Some(&w_last) = weight.shape().last() else {
        return fallback;
    };
    let Some(&s_last) = scales.shape().last() else {
        return fallback;
    };
    if s_last <= 0 || group_size <= 0 {
        return fallback;
    }
    let numerator = w_last * 32;
    let denominator = s_last * group_size;
    if numerator % denominator == 0 {
        let bits = numerator / denominator;
        if matches!(bits, 2 | 3 | 4 | 6 | 8) {
            return bits;
        }
    }
    fallback
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
        for (key, value) in map {
            if key.starts_with("language_model.") {
                out.insert(key, value);
            }
        }
    }
    Ok(out)
}

fn layer_types(cfg: &ModelConfig) -> Result<Vec<LayerType>> {
    if cfg.layer_types.len() != cfg.num_hidden_layers {
        return Err(Error::InvalidConfig(format!(
            "Gemma4 expected {} layer_types, got {}",
            cfg.num_hidden_layers,
            cfg.layer_types.len()
        )));
    }
    cfg.layer_types
        .iter()
        .map(|s| LayerType::from_str(s))
        .collect()
}

fn previous_kvs(layer_types: &[LayerType], first_kv_shared: usize) -> Result<Vec<usize>> {
    let mut previous = Vec::with_capacity(layer_types.len());
    let mut sliding = None;
    let mut full = None;
    for (i, layer_type) in layer_types.iter().copied().enumerate() {
        if i < first_kv_shared {
            previous.push(i);
            match layer_type {
                LayerType::Sliding => sliding = Some(i),
                LayerType::Full => full = Some(i),
            }
        } else {
            let idx = match layer_type {
                LayerType::Sliding => sliding,
                LayerType::Full => full,
            }
            .ok_or_else(|| Error::InvalidConfig("Gemma4 shared layer lacks prior kv".into()))?;
            previous.push(idx);
        }
    }
    Ok(previous)
}

fn rope_f32(cfg: &ModelConfig, group: &str, key: &str, default: f32) -> f32 {
    cfg.rope_parameters
        .as_ref()
        .and_then(|v| v.get(group))
        .and_then(|v| v.get(key))
        .and_then(ValueExt::as_f32)
        .unwrap_or(default)
}

trait ValueExt {
    fn as_f32(&self) -> Option<f32>;
}

impl ValueExt for serde_json::Value {
    fn as_f32(&self) -> Option<f32> {
        self.as_f64().map(|v| v as f32)
    }
}

fn proportional_rope_freqs(head_dim: usize, partial_factor: f32, theta: f32) -> Array {
    let rotated = ((head_dim as f32 * partial_factor) as usize).min(head_dim);
    let mut freqs = Vec::with_capacity(head_dim / 2);
    for i in (0..rotated).step_by(2) {
        freqs.push(theta.powf(i as f32 / head_dim as f32));
    }
    while freqs.len() < head_dim / 2 {
        freqs.push(f32::INFINITY);
    }
    Array::from_slice(&freqs, &[freqs.len() as i32])
}

fn attention_mask(
    seq_len: i32,
    offset: i32,
    window: Option<i32>,
    dtype: Dtype,
) -> Result<Option<Array>> {
    let k_len = offset + seq_len;
    if seq_len == 1 && window.is_none_or(|w| k_len <= w) {
        return Ok(None);
    }

    let mut vals = Vec::with_capacity((seq_len * k_len) as usize);
    for q in 0..seq_len {
        let q_abs = offset + q;
        for k in 0..k_len {
            let causal = k <= q_abs;
            let in_window = window.is_none_or(|w| k > q_abs - w);
            vals.push(if causal && in_window { 0.0 } else { -1.0e9 });
        }
    }
    let mask = Array::from_slice(&vals, &[seq_len, k_len])
        .as_dtype(dtype)
        .map_err(emap)?;
    Ok(Some(mask))
}

fn logit_softcap(x: &Array, cap: f32) -> Result<Array> {
    let cap = Array::from_f32(cap);
    let scaled = x / &cap;
    let tanh = mlx_rs::ops::tanh(&scaled).map_err(emap)?;
    Ok(&tanh * &cap)
}
