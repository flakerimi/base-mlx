//! Compiled hot kernels for Qwen3 decode.
//!
//! These are free `fn(&[Array]) -> Vec<Array>` because `mlx_rs::compile`
//! requires `'static + Copy` — function pointers qualify, captures don't.
//! Each layer passes its weights into the same compiled function on
//! every step; MLX caches the fused graph by function-pointer identity
//! (plus input shapes) and hits that cache for every subsequent call.
//!
//! Hyperparameters baked in for Qwen3-4B-Instruct-2507-4bit:
//!   bits=4, group_size=64, rms_eps=1e-6
//! Other Qwen3 variants share these. When we add a different model
//! family we'll either parametrize or generate the kernels per-shape.

use mlx_rs::Array;

const BITS: i32 = 4;
const GROUP: i32 = 64;
const EPS: f32 = 1e-6;

/// QKV projection block.
///
/// Inputs: `[x, in_norm_w,
///           q_w, q_s, q_b,
///           k_w, k_s, k_b,
///           v_w, v_s, v_b]`
/// Returns: `[q, k, v]` with the model's full per-axis layout
/// (`[seq, n_heads * head_dim]` / `[seq, kv_heads * head_dim]`).
pub fn qkv_block(args: &[Array]) -> Vec<Array> {
    let x = &args[0];
    let in_norm_w = &args[1];
    let q_w = &args[2];
    let q_s = &args[3];
    let q_b = &args[4];
    let k_w = &args[5];
    let k_s = &args[6];
    let k_b = &args[7];
    let v_w = &args[8];
    let v_s = &args[9];
    let v_b = &args[10];

    let h = mlx_rs::fast::rms_norm(x, in_norm_w, EPS).expect("rms_norm");
    let q =
        mlx_rs::ops::quantized_matmul(&h, q_w, q_s, q_b, true, GROUP, BITS).expect("q_proj");
    let k =
        mlx_rs::ops::quantized_matmul(&h, k_w, k_s, k_b, true, GROUP, BITS).expect("k_proj");
    let v =
        mlx_rs::ops::quantized_matmul(&h, v_w, v_s, v_b, true, GROUP, BITS).expect("v_proj");
    vec![q, k, v]
}

/// SwiGLU MLP block.
///
/// Inputs: `[x, post_norm_w,
///           gate_w, gate_s, gate_b,
///           up_w, up_s, up_b,
///           down_w, down_s, down_b]`
/// Returns: `[down]` — the MLP output added to the residual outside.
pub fn mlp_block(args: &[Array]) -> Vec<Array> {
    let x = &args[0];
    let post_norm_w = &args[1];
    let gate_w = &args[2];
    let gate_s = &args[3];
    let gate_b = &args[4];
    let up_w = &args[5];
    let up_s = &args[6];
    let up_b = &args[7];
    let down_w = &args[8];
    let down_s = &args[9];
    let down_b = &args[10];

    let h = mlx_rs::fast::rms_norm(x, post_norm_w, EPS).expect("post_norm");
    let gate = mlx_rs::ops::quantized_matmul(&h, gate_w, gate_s, gate_b, true, GROUP, BITS)
        .expect("gate_proj");
    let up = mlx_rs::ops::quantized_matmul(&h, up_w, up_s, up_b, true, GROUP, BITS)
        .expect("up_proj");
    let activated = mlx_rs::nn::silu(&gate).expect("silu");
    let gated = &activated * &up;
    let down = mlx_rs::ops::quantized_matmul(&gated, down_w, down_s, down_b, true, GROUP, BITS)
        .expect("down_proj");
    vec![down]
}
