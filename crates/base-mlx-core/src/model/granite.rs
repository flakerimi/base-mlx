//! IBM Granite MoE Hybrid (`granitemoehybrid`).
//!
//! The tiny Granite 4 checkpoint is a hybrid: most layers are Mamba2 SSM
//! blocks, with periodic attention layers and a routed MoE + shared MLP after
//! every block. This first implementation is decode-oriented: it steps one
//! token at a time through the recurrent SSM, which keeps the cache semantics
//! simple and matches generation's steady-state path.

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
struct QSwitchLinear {
    weight: Array,
    scales: Array,
    biases: Array,
    group_size: i32,
    bits: i32,
}

impl QSwitchLinear {
    fn concat_output(first: Self, second: Self) -> Result<Self> {
        if first.group_size != second.group_size || first.bits != second.bits {
            return Err(Error::ModelLoad(
                "cannot concatenate routed projections with different quantization".into(),
            ));
        }
        let group_size = first.group_size;
        let bits = first.bits;
        let weight =
            mlx_rs::ops::concatenate_axis(&[first.weight, second.weight], 1).map_err(emap)?;
        let scales =
            mlx_rs::ops::concatenate_axis(&[first.scales, second.scales], 1).map_err(emap)?;
        let biases =
            mlx_rs::ops::concatenate_axis(&[first.biases, second.biases], 1).map_err(emap)?;
        mlx_rs::transforms::eval([&weight, &scales, &biases]).map_err(emap)?;
        Ok(Self {
            weight,
            scales,
            biases,
            group_size,
            bits,
        })
    }

    fn forward(&self, x: &Array, indices: &Array) -> Result<Array> {
        crate::mlx_ext::gather_qmm(
            x,
            &self.weight,
            &self.scales,
            &self.biases,
            None,
            Some(indices),
            true,
            self.group_size,
            self.bits,
            false,
        )
        .map_err(Error::Inference)
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

    fn forward_gated(&self, x: &Array, gate: &Array) -> Result<Array> {
        let gate = mlx_rs::nn::silu(gate).map_err(emap)?;
        self.forward(&(x * &gate))
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

#[derive(Debug)]
struct SharedMlp {
    input: QLinear,
    output: QLinear,
    hidden: i32,
}

impl SharedMlp {
    fn forward(&self, x: &Array) -> Result<Array> {
        let parts = self
            .input
            .forward(x)?
            .split_axis(&[self.hidden], -1)
            .map_err(emap)?;
        let gate = mlx_rs::nn::silu(&parts[0]).map_err(emap)?;
        self.output.forward(&(&gate * &parts[1]))
    }
}

#[derive(Debug)]
struct SwitchGlu {
    gate_up_proj: QSwitchLinear,
    down_proj: QSwitchLinear,
    hidden: i32,
}

impl SwitchGlu {
    fn forward(&self, x: &Array, indices: &Array) -> Result<Array> {
        let x = x.expand_dims_axes(&[-2, -3]).map_err(emap)?;
        let parts = self
            .gate_up_proj
            .forward(&x, indices)?
            .split_axis(&[self.hidden], -1)
            .map_err(emap)?;
        let gate = mlx_rs::nn::silu(&parts[0]).map_err(emap)?;
        let hidden = &gate * &parts[1];
        let out = self.down_proj.forward(&hidden, indices)?;
        out.squeeze_axes(&[-2]).map_err(emap)
    }
}

#[derive(Debug)]
struct Moe {
    router: QLinear,
    switch_mlp: SwitchGlu,
    num_experts: i32,
    top_k: i32,
}

impl Moe {
    fn forward(&self, x: &Array) -> Result<Array> {
        let logits = self.router.forward(x)?;
        let kth = self.num_experts - self.top_k;
        let indices = mlx_rs::ops::argpartition_axis(&logits, kth, -1).map_err(emap)?;
        let indices = indices.index((.., kth..));
        let top_logits =
            mlx_rs::ops::indexing::take_along_axis(&logits, &indices, -1).map_err(emap)?;
        let gates = mlx_rs::ops::softmax_axis(&top_logits, -1, true).map_err(emap)?;
        let routed = self.switch_mlp.forward(x, &indices)?;
        let gates = gates.expand_dims(-1).map_err(emap)?;
        (&routed * &gates).sum_axis(-2, false).map_err(emap)
    }
}

#[derive(Debug)]
struct MambaMixer {
    in_proj: QLinear,
    conv_weight: Array,
    conv_bias: Option<Array>,
    norm: RmsNorm,
    out_proj: QLinear,
    dt_bias: Array,
    a: Array,
    d: Array,
    num_heads: i32,
    head_dim: i32,
    state_size: i32,
    conv_dim: i32,
    conv_kernel: i32,
    intermediate: i32,
}

impl MambaMixer {
    fn forward(&self, x: &Array, cache: &mut KvCache, layer_idx: usize) -> Result<Array> {
        let projected = self.in_proj.forward(x)?;
        let parts = projected
            .split_axis(&[self.intermediate, self.intermediate + self.conv_dim], -1)
            .map_err(emap)?;
        let gate = &parts[0];
        let conv_input = &parts[1];
        let dt = &parts[2];

        let conv_output = self.apply_conv(conv_input, cache, layer_idx)?;
        let conv_parts = conv_output
            .split_axis(
                &[self.intermediate, self.intermediate + self.state_size],
                -1,
            )
            .map_err(emap)?;
        let y = self.ssm_step(
            &conv_parts[0],
            &conv_parts[1],
            &conv_parts[2],
            dt,
            cache,
            layer_idx,
        )?;
        let y = self.norm.forward_gated(&y, gate)?;
        self.out_proj.forward(&y)
    }

    fn apply_conv(
        &self,
        conv_input: &Array,
        cache: &mut KvCache,
        layer_idx: usize,
    ) -> Result<Array> {
        let conv_input = conv_input.expand_dims(0).map_err(emap)?;
        let state = cache.conv_state(
            layer_idx,
            self.conv_kernel - 1,
            self.conv_dim,
            conv_input.dtype(),
        )?;
        if let Some(bias) = &self.conv_bias {
            let input = conv_input.reshape(&[self.conv_dim]).map_err(emap)?;
            let (y, new_state) = crate::mlx_ext::granite_conv1d_update(
                &input,
                &state,
                &self.conv_weight,
                bias,
                self.conv_dim,
                self.conv_kernel,
            )
            .map_err(Error::Inference)?;
            cache.conv[layer_idx] = Some(new_state);
            return y.reshape(&[1, self.conv_dim]).map_err(emap);
        }
        let padded = mlx_rs::ops::concatenate_axis(&[state, conv_input], 1).map_err(emap)?;
        cache.conv[layer_idx] = Some(padded.index((.., 1..self.conv_kernel, ..)));
        let padded = padded.index((0, .., ..));
        let mut y = (&padded * &self.conv_weight)
            .sum_axis(0, false)
            .map_err(emap)?
            .expand_dims(0)
            .map_err(emap)?;
        if let Some(bias) = &self.conv_bias {
            y = &y + bias;
        }
        let y = mlx_rs::nn::silu(&y).map_err(emap)?;
        Ok(y)
    }

    fn ssm_step(
        &self,
        hidden: &Array,
        b: &Array,
        c: &Array,
        dt: &Array,
        cache: &mut KvCache,
        layer_idx: usize,
    ) -> Result<Array> {
        let hidden = hidden
            .reshape(&[self.num_heads, self.head_dim])
            .map_err(emap)?;
        let b = b.reshape(&[self.state_size]).map_err(emap)?;
        let c = c.reshape(&[self.state_size]).map_err(emap)?;
        let dt = mlx_rs::nn::softplus(&(dt + &self.dt_bias)).map_err(emap)?;
        let dt = mlx_rs::ops::clip(&dt, (0.001f32, 100.0f32)).map_err(emap)?;
        let dt = dt.reshape(&[self.num_heads]).map_err(emap)?;

        let state = cache.ssm_state(
            layer_idx,
            self.num_heads,
            self.head_dim,
            self.state_size,
            hidden.dtype(),
        )?;
        let (mixed, new_state) = crate::mlx_ext::granite_ssm_update(
            &hidden,
            &self.a,
            &b,
            &c,
            &self.d,
            &dt,
            &state,
            self.num_heads,
            self.head_dim,
            self.state_size,
        )
        .map_err(Error::Inference)?;
        cache.ssm[layer_idx] = Some(new_state);

        mixed.reshape(&[1, self.intermediate]).map_err(emap)
    }
}

#[derive(Debug)]
struct AttentionBlock {
    q_proj: QLinear,
    k_proj: QLinear,
    v_proj: QLinear,
    o_proj: QLinear,
}

impl AttentionBlock {
    fn forward(
        &self,
        x: &Array,
        cache: &mut KvCache,
        layer_idx: usize,
        cfg: &ModelConfig,
        scale: f32,
    ) -> Result<Array> {
        let n_heads = cfg.num_attention_heads as i32;
        let kv_heads = cfg.kv_heads() as i32;
        let head_dim = cfg.per_head_dim() as i32;
        let offset = cache.seq_len as i32;
        let new_end = offset + 1;
        if new_end > KV_PREALLOC {
            return Err(Error::Inference(format!(
                "KV cache exhausted: requested {new_end} positions, max {KV_PREALLOC}"
            )));
        }

        let q = self
            .q_proj
            .forward(x)?
            .reshape(&[1, n_heads, head_dim])
            .map_err(emap)?
            .transpose_axes(&[1, 0, 2])
            .map_err(emap)?;
        let k = self
            .k_proj
            .forward(x)?
            .reshape(&[1, kv_heads, head_dim])
            .map_err(emap)?
            .transpose_axes(&[1, 0, 2])
            .map_err(emap)?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape(&[1, kv_heads, head_dim])
            .map_err(emap)?
            .transpose_axes(&[1, 0, 2])
            .map_err(emap)?;

        let (k_cached, v_cached) =
            cache.update_kv(layer_idx, &k, &v, kv_heads, head_dim, offset, new_end)?;
        let q = q.expand_dims(0).map_err(emap)?;
        let k = k_cached.expand_dims(0).map_err(emap)?;
        let v = v_cached.expand_dims(0).map_err(emap)?;
        let attn = mlx_rs::fast::scaled_dot_product_attention(
            &q,
            &k,
            &v,
            scale,
            Option::<mlx_rs::fast::ScaledDotProductAttentionMask<'_>>::None,
        )
        .map_err(emap)?;
        let attn = attn
            .transpose_axes(&[0, 2, 1, 3])
            .map_err(emap)?
            .reshape(&[1, n_heads * head_dim])
            .map_err(emap)?;
        self.o_proj.forward(&attn)
    }
}

#[derive(Debug)]
enum LayerBlock {
    Mamba(MambaMixer),
    Attention(AttentionBlock),
}

#[derive(Debug)]
struct GraniteLayer {
    block: LayerBlock,
    input_norm: RmsNorm,
    post_norm: RmsNorm,
    shared_mlp: SharedMlp,
    moe: Moe,
}

#[derive(Debug, Default)]
pub struct KvCache {
    k: Vec<Option<Array>>,
    v: Vec<Option<Array>>,
    conv: Vec<Option<Array>>,
    ssm: Vec<Option<Array>>,
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
        self.conv.clear();
        self.ssm.clear();
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
        // SSM state is recurrent and not cheaply reversible. Granite is not
        // used for speculative verification yet; reset Mamba state rather than
        // silently pretending it can be rolled back.
        for c in &mut self.conv {
            *c = None;
        }
        for s in &mut self.ssm {
            *s = None;
        }
        self.seq_len = target_seq_len;
    }

    fn ensure_layers(&mut self, layers: usize) {
        if self.k.len() < layers {
            self.k.resize_with(layers, || None);
            self.v.resize_with(layers, || None);
            self.conv.resize_with(layers, || None);
            self.ssm.resize_with(layers, || None);
        }
    }

    fn conv_state(&mut self, li: usize, len: i32, dim: i32, dtype: Dtype) -> Result<Array> {
        if let Some(state) = &self.conv[li] {
            return Ok(state.clone());
        }
        let state = mlx_rs::ops::zeros_dtype(&[1, len, dim], dtype).map_err(emap)?;
        self.conv[li] = Some(state.clone());
        Ok(state)
    }

    fn ssm_state(
        &mut self,
        li: usize,
        heads: i32,
        head_dim: i32,
        state_dim: i32,
        dtype: Dtype,
    ) -> Result<Array> {
        if let Some(state) = &self.ssm[li] {
            return Ok(state.clone());
        }
        let state = mlx_rs::ops::zeros_dtype(&[heads, head_dim, state_dim], dtype).map_err(emap)?;
        self.ssm[li] = Some(state.clone());
        Ok(state)
    }

    #[allow(clippy::too_many_arguments)]
    fn update_kv(
        &mut self,
        li: usize,
        k: &Array,
        v: &Array,
        kv_heads: i32,
        head_dim: i32,
        offset: i32,
        new_end: i32,
    ) -> Result<(Array, Array)> {
        if self.k[li].is_some() {
            let kk = crate::mlx_ext::slice_update(
                self.k[li].as_ref().unwrap(),
                k,
                &[0, offset, 0],
                &[kv_heads, new_end, head_dim],
                &[1, 1, 1],
            )
            .map_err(Error::Inference)?;
            let vv = crate::mlx_ext::slice_update(
                self.v[li].as_ref().unwrap(),
                v,
                &[0, offset, 0],
                &[kv_heads, new_end, head_dim],
                &[1, 1, 1],
            )
            .map_err(Error::Inference)?;
            self.k[li] = Some(kk);
            self.v[li] = Some(vv);
        } else {
            let zeros_k = mlx_rs::ops::zeros_dtype(&[kv_heads, KV_PREALLOC, head_dim], k.dtype())
                .map_err(emap)?;
            let zeros_v = mlx_rs::ops::zeros_dtype(&[kv_heads, KV_PREALLOC, head_dim], v.dtype())
                .map_err(emap)?;
            let kk = crate::mlx_ext::slice_update(
                &zeros_k,
                k,
                &[0, 0, 0],
                &[kv_heads, 1, head_dim],
                &[1, 1, 1],
            )
            .map_err(Error::Inference)?;
            let vv = crate::mlx_ext::slice_update(
                &zeros_v,
                v,
                &[0, 0, 0],
                &[kv_heads, 1, head_dim],
                &[1, 1, 1],
            )
            .map_err(Error::Inference)?;
            self.k[li] = Some(kk);
            self.v[li] = Some(vv);
        }
        let kk = self.k[li].as_ref().unwrap().index((.., 0..new_end, ..));
        let vv = self.v[li].as_ref().unwrap().index((.., 0..new_end, ..));
        Ok((kk, vv))
    }
}

#[derive(Debug)]
pub struct Granite {
    pub cfg: ModelConfig,
    embed: QEmbedding,
    layers: Vec<GraniteLayer>,
    norm: RmsNorm,
    lm_head: Option<QLinear>,
    embedding_mul: Array,
    residual_mul: Array,
    logits_scale_inv: Array,
}

impl Granite {
    pub fn load(dir: &Path, cfg: ModelConfig) -> Result<Self> {
        if !cfg.tie_word_embeddings {
            return Err(Error::ModelLoad(
                "Granite untied lm_head checkpoints are not supported yet".into(),
            ));
        }
        if cfg.position_embedding_type != "nope" {
            return Err(Error::ModelLoad(format!(
                "Granite position_embedding_type={} is not supported yet",
                cfg.position_embedding_type
            )));
        }
        if cfg.mamba_n_groups.unwrap_or(1) != 1 {
            return Err(Error::ModelLoad(
                "Granite Mamba groups > 1 are not supported yet".into(),
            ));
        }

        let mut tensors = read_all_shards(dir)?;
        let q = cfg
            .quantization
            .as_ref()
            .ok_or_else(|| Error::ModelLoad("only quantized Granite models supported".into()))?;
        let bits = q.bits as i32;
        let group_size = q.group_size as i32;
        let eps = cfg.rms_norm_eps;

        let embed = qembed(&mut tensors, "model.embed_tokens", bits, group_size)?;

        let layer_types = if cfg.layer_types.is_empty() {
            return Err(Error::InvalidConfig(
                "Granite config has no layer_types".into(),
            ));
        } else {
            cfg.layer_types.clone()
        };
        let mamba_heads = cfg.mamba_n_heads.unwrap_or(48) as i32;
        let mamba_head_dim = cfg.mamba_d_head.unwrap_or(64) as i32;
        let mamba_state = cfg.mamba_d_state.unwrap_or(128) as i32;
        let mamba_conv = cfg.mamba_d_conv.unwrap_or(4) as i32;
        let mamba_intermediate = mamba_heads * mamba_head_dim;
        let mamba_conv_dim = mamba_intermediate + 2 * mamba_state;
        let shared_hidden = cfg.shared_intermediate_size.unwrap_or(1024) as i32;
        let top_k = cfg.num_experts_per_tok.unwrap_or(6) as i32;
        let num_experts = cfg.num_local_experts.unwrap_or(64) as i32;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for (i, layer_type) in layer_types.iter().enumerate() {
            let p = format!("model.layers.{i}");
            let block = match layer_type.as_str() {
                "mamba" => LayerBlock::Mamba(MambaMixer {
                    in_proj: qlin(
                        &mut tensors,
                        &format!("{p}.mamba.in_proj"),
                        bits,
                        group_size,
                    )?,
                    conv_weight: take(&mut tensors, &format!("{p}.mamba.conv1d.weight"))?
                        .reshape(&[mamba_conv_dim, mamba_conv])
                        .map_err(emap)?
                        .transpose_axes(&[1, 0])
                        .map_err(emap)?,
                    conv_bias: if cfg.mamba_conv_bias {
                        Some(take(&mut tensors, &format!("{p}.mamba.conv1d.bias"))?)
                    } else {
                        None
                    },
                    norm: rms(&mut tensors, &format!("{p}.mamba.norm.weight"), eps)?,
                    out_proj: qlin(
                        &mut tensors,
                        &format!("{p}.mamba.out_proj"),
                        bits,
                        group_size,
                    )?,
                    dt_bias: take(&mut tensors, &format!("{p}.mamba.dt_bias"))?,
                    a: &(take(&mut tensors, &format!("{p}.mamba.A_log"))?
                        .exp()
                        .map_err(emap)?)
                        * &Array::from_f32(-1.0),
                    d: take(&mut tensors, &format!("{p}.mamba.D"))?
                        .reshape(&[mamba_heads, 1])
                        .map_err(emap)?,
                    num_heads: mamba_heads,
                    head_dim: mamba_head_dim,
                    state_size: mamba_state,
                    conv_dim: mamba_conv_dim,
                    conv_kernel: mamba_conv,
                    intermediate: mamba_intermediate,
                }),
                "attention" => LayerBlock::Attention(AttentionBlock {
                    q_proj: qlin(
                        &mut tensors,
                        &format!("{p}.self_attn.q_proj"),
                        bits,
                        group_size,
                    )?,
                    k_proj: qlin(
                        &mut tensors,
                        &format!("{p}.self_attn.k_proj"),
                        bits,
                        group_size,
                    )?,
                    v_proj: qlin(
                        &mut tensors,
                        &format!("{p}.self_attn.v_proj"),
                        bits,
                        group_size,
                    )?,
                    o_proj: qlin(
                        &mut tensors,
                        &format!("{p}.self_attn.o_proj"),
                        bits,
                        group_size,
                    )?,
                }),
                other => {
                    return Err(Error::InvalidConfig(format!(
                        "unsupported Granite layer type: {other}"
                    )));
                }
            };

            layers.push(GraniteLayer {
                block,
                input_norm: rms(&mut tensors, &format!("{p}.input_layernorm.weight"), eps)?,
                post_norm: rms(
                    &mut tensors,
                    &format!("{p}.post_attention_layernorm.weight"),
                    eps,
                )?,
                shared_mlp: SharedMlp {
                    input: qlin(
                        &mut tensors,
                        &format!("{p}.shared_mlp.input_linear"),
                        bits,
                        group_size,
                    )?,
                    output: qlin(
                        &mut tensors,
                        &format!("{p}.shared_mlp.output_linear"),
                        bits,
                        group_size,
                    )?,
                    hidden: shared_hidden,
                },
                moe: Moe {
                    router: qlin(
                        &mut tensors,
                        &format!("{p}.block_sparse_moe.router.layer"),
                        bits,
                        group_size,
                    )?,
                    switch_mlp: SwitchGlu {
                        gate_up_proj: QSwitchLinear::concat_output(
                            qswitch(
                                &mut tensors,
                                &format!("{p}.block_sparse_moe.switch_mlp.gate_proj"),
                                bits,
                                group_size,
                            )?,
                            qswitch(
                                &mut tensors,
                                &format!("{p}.block_sparse_moe.switch_mlp.up_proj"),
                                bits,
                                group_size,
                            )?,
                        )?,
                        down_proj: qswitch(
                            &mut tensors,
                            &format!("{p}.block_sparse_moe.switch_mlp.down_proj"),
                            bits,
                            group_size,
                        )?,
                        hidden: cfg.intermediate_size as i32,
                    },
                    num_experts,
                    top_k,
                },
            });
        }

        let norm = rms(&mut tensors, "model.norm.weight", eps)?;
        if !tensors.is_empty() {
            let leftover: Vec<_> = tensors.keys().take(5).cloned().collect();
            return Err(Error::ModelLoad(format!(
                "{} unmapped tensors remain (first 5: {:?})",
                tensors.len(),
                leftover
            )));
        }

        let embedding_mul = Array::from_f32(cfg.embedding_multiplier);
        let residual_mul = Array::from_f32(cfg.residual_multiplier);
        let logits_scale_inv = Array::from_f32(1.0 / cfg.logits_scaling);

        Ok(Self {
            cfg,
            embed,
            layers,
            norm,
            lm_head: None,
            embedding_mul,
            residual_mul,
            logits_scale_inv,
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
        if tokens.is_empty() {
            return Err(Error::Inference(
                "Granite forward called with no tokens".into(),
            ));
        }
        cache.ensure_layers(self.layers.len());
        let mut all = Vec::with_capacity(tokens.len());
        let mut last = None;
        for &token in tokens {
            let logits = self.forward_one(token, cache)?;
            if return_all_logits {
                all.push(logits.expand_dims(0).map_err(emap)?);
            } else {
                last = Some(logits);
            }
        }
        if return_all_logits {
            mlx_rs::ops::concatenate_axis(&all, 0).map_err(emap)
        } else {
            last.ok_or_else(|| Error::Inference("Granite produced no logits".into()))
        }
    }

    fn forward_one(&self, token: u32, cache: &mut KvCache) -> Result<Array> {
        let tok_arr = Array::from_slice(&[token as i32], &[1]);
        self.forward_one_array(&tok_arr, cache)
    }

    fn forward_one_array(&self, tok_arr: &Array, cache: &mut KvCache) -> Result<Array> {
        let mut x = self.embed.lookup(tok_arr)?;
        x = &x * &self.embedding_mul;

        for (li, layer) in self.layers.iter().enumerate() {
            let residual = x.clone();
            let h = layer.input_norm.forward(&x)?;
            let block_out = match &layer.block {
                LayerBlock::Mamba(mamba) => mamba.forward(&h, cache, li)?,
                LayerBlock::Attention(attn) => {
                    attn.forward(&h, cache, li, &self.cfg, self.cfg.attention_multiplier)?
                }
            };
            x = &residual + &(&block_out * &self.residual_mul);

            let residual = x.clone();
            let normed = layer.post_norm.forward(&x)?;
            let moe = layer.moe.forward(&normed)?;
            let shared = layer.shared_mlp.forward(&normed)?;
            let mlp = &moe + &shared;
            x = &residual + &(&mlp * &self.residual_mul);
        }

        let x = self.norm.forward(&x)?;
        let logits = match &self.lm_head {
            Some(head) => head.forward(&x)?,
            None => self.embed.lm_head(&x)?,
        };
        cache.seq_len += 1;
        let logits = &logits * &self.logits_scale_inv;
        Ok(logits.index((0, ..)))
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

fn qswitch(
    map: &mut HashMap<String, Array>,
    prefix: &str,
    bits: i32,
    group_size: i32,
) -> Result<QSwitchLinear> {
    let weight = take(map, &format!("{prefix}.weight"))?;
    let scales = take(map, &format!("{prefix}.scales"))?;
    let biases = take(map, &format!("{prefix}.biases"))?;
    let bits = infer_bits(&weight, &scales, bits, group_size);
    Ok(QSwitchLinear {
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
        out.extend(map);
    }
    Ok(out)
}
