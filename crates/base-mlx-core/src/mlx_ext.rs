//! Extensions to mlx-rs we needed but the binding didn't expose.
//!
//! Built directly on top of `mlx-sys` (the raw C FFI to mlx-c), which is
//! the same surface mlx-rs uses internally. Anything we add here is a
//! candidate for upstreaming to mlx-rs — we keep the API shape close to
//! what mlx-rs would ship so a future migration is a find-and-replace.
//!
//! Why this exists at all: mlx-rs 0.25 marks `slice_update_device` as
//! `pub(crate)`, so we can't reach it from outside. The C function it
//! wraps (`mlx_slice_update`) is exactly what mlx-lm (Python) uses to
//! get *in-place* KV-cache writes — MLX's runtime planner reuses the
//! input buffer when its refcount is 1, eliding the copy. We need that
//! for the inference hot path; concat-axis allocates a fresh buffer per
//! decode step and dominates long-generation cost.

use mlx_rs::Array;
use mlx_sys as sys;
use std::ffi::CString;
use std::sync::OnceLock;

/// Update a contiguous slice of `src` in place (when MLX can prove the
/// input buffer is unique-referenced) with `update`. Returns a new
/// `Array` handle; on the in-place path the handle wraps the same
/// underlying buffer as `src`.
///
/// Semantics mirror `numpy`'s `src[tuple(slice(s, e, k) for s,e,k in …)] = update`:
///   * `start`, `stop`, `strides` are per-axis vectors describing the
///     slice. Lengths must equal `src.ndim()`.
///   * `update.shape()` must match the slice's effective shape under
///     the given strides.
///
/// The caller is responsible for not holding extra references to `src`
/// at the time of the call — otherwise MLX has to copy. Our `KvCache`
/// pre-allocates buffers and writes back via `*k = slice_update(k, …)`
/// to guarantee single-reference at the moment of the call.
pub fn slice_update(
    src: &Array,
    update: &Array,
    start: &[i32],
    stop: &[i32],
    strides: &[i32],
) -> Result<Array, String> {
    unsafe {
        let mut out = sys::mlx_array_new();
        let stream = default_gpu_stream();
        let rc = sys::mlx_slice_update(
            &mut out as *mut sys::mlx_array,
            src.as_ptr(),
            update.as_ptr(),
            start.as_ptr(),
            start.len(),
            stop.as_ptr(),
            stop.len(),
            strides.as_ptr(),
            strides.len(),
            stream,
        );
        if rc != 0 {
            sys::mlx_array_free(out);
            return Err(format!("mlx_slice_update failed: rc={}", rc));
        }
        Ok(Array::from_ptr(out))
    }
}

/// Expose MLX's routed quantized matmul. This is the primitive mlx-lm uses
/// for MoE expert projections: `rhs_indices` selects which expert matrix to
/// use for each token/top-k route without materializing all experts.
#[allow(clippy::too_many_arguments)]
pub fn gather_qmm(
    x: &Array,
    w: &Array,
    scales: &Array,
    biases: &Array,
    lhs_indices: Option<&Array>,
    rhs_indices: Option<&Array>,
    transpose: bool,
    group_size: i32,
    bits: i32,
    sorted_indices: bool,
) -> Result<Array, String> {
    unsafe {
        let mut out = sys::mlx_array_new();
        let stream = default_gpu_stream();
        let lhs = lhs_indices
            .map(Array::as_ptr)
            .unwrap_or_else(|| sys::mlx_array_new());
        let rhs = rhs_indices
            .map(Array::as_ptr)
            .unwrap_or_else(|| sys::mlx_array_new());
        let rc = sys::mlx_gather_qmm(
            &mut out as *mut sys::mlx_array,
            x.as_ptr(),
            w.as_ptr(),
            scales.as_ptr(),
            biases.as_ptr(),
            lhs,
            rhs,
            transpose,
            group_size,
            bits,
            sorted_indices,
            stream,
        );
        if lhs_indices.is_none() {
            sys::mlx_array_free(lhs);
        }
        if rhs_indices.is_none() {
            sys::mlx_array_free(rhs);
        }
        if rc != 0 {
            sys::mlx_array_free(out);
            return Err(format!("mlx_gather_qmm failed: rc={rc}"));
        }
        Ok(Array::from_ptr(out))
    }
}

/// Fused single-token Granite/Mamba2 SSM update.
///
/// This mirrors mlx-lm's decode-time Metal kernel for Mamba state updates. It
/// updates `[heads, head_dim, state]` recurrent state and produces the mixed
/// `[heads, head_dim]` output in one launch instead of materializing several
/// broadcast and reduction ops for every Mamba layer/token.
#[allow(clippy::too_many_arguments)]
pub fn granite_ssm_update(
    hidden: &Array,
    a: &Array,
    b: &Array,
    c: &Array,
    d: &Array,
    dt: &Array,
    state: &Array,
    heads: i32,
    head_dim: i32,
    state_dim: i32,
) -> Result<(Array, Array), String> {
    unsafe {
        let kernel = granite_ssm_kernel()?;
        let inputs = vector_array(&[hidden, a, b, c, d, dt, state])?;
        let mut outputs = sys::mlx_vector_array_new();
        let config = sys::mlx_fast_metal_kernel_config_new();
        let stream = default_gpu_stream();

        let out_shape = [heads, head_dim];
        let state_shape = [heads, head_dim, state_dim];
        let dtype = hidden.dtype() as sys::mlx_dtype;
        let mut rc = sys::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            out_shape.as_ptr(),
            out_shape.len(),
            dtype,
        );
        rc |= sys::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            state_shape.as_ptr(),
            state_shape.len(),
            dtype,
        );
        rc |= sys::mlx_fast_metal_kernel_config_set_grid(config, 32, head_dim, heads);
        rc |= sys::mlx_fast_metal_kernel_config_set_thread_group(config, 32, 8, 1);
        rc |= add_template_dtype(config, "T", dtype);
        rc |= add_template_int(config, "Dh", head_dim);
        rc |= add_template_int(config, "Ds", state_dim);

        if rc == 0 {
            rc = sys::mlx_fast_metal_kernel_apply(&mut outputs, kernel, inputs, config, stream);
        }

        sys::mlx_fast_metal_kernel_config_free(config);
        sys::mlx_vector_array_free(inputs);

        if rc != 0 {
            sys::mlx_vector_array_free(outputs);
            return Err(format!("granite_ssm_update failed: rc={rc}"));
        }
        if sys::mlx_vector_array_size(outputs) != 2 {
            let size = sys::mlx_vector_array_size(outputs);
            sys::mlx_vector_array_free(outputs);
            return Err(format!(
                "granite_ssm_update returned {size} outputs, expected 2"
            ));
        }

        let mixed = vector_array_get(outputs, 0)?;
        let new_state = vector_array_get(outputs, 1)?;
        sys::mlx_vector_array_free(outputs);
        Ok((mixed, new_state))
    }
}

/// Fused single-token depthwise conv cache update for Granite/Mamba2.
#[allow(clippy::too_many_arguments)]
pub fn granite_conv1d_update(
    input: &Array,
    state: &Array,
    weight: &Array,
    bias: &Array,
    dim: i32,
    kernel: i32,
) -> Result<(Array, Array), String> {
    unsafe {
        let metal_kernel = granite_conv1d_kernel()?;
        let inputs = vector_array(&[input, state, weight, bias])?;
        let mut outputs = sys::mlx_vector_array_new();
        let config = sys::mlx_fast_metal_kernel_config_new();
        let stream = default_gpu_stream();

        let out_shape = [dim];
        let state_shape = [1, kernel - 1, dim];
        let dtype = input.dtype() as sys::mlx_dtype;
        let mut rc = sys::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            out_shape.as_ptr(),
            out_shape.len(),
            dtype,
        );
        rc |= sys::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            state_shape.as_ptr(),
            state_shape.len(),
            dtype,
        );
        rc |= sys::mlx_fast_metal_kernel_config_set_grid(config, dim, 1, 1);
        rc |= sys::mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1);
        rc |= add_template_dtype(config, "T", dtype);
        rc |= add_template_int(config, "Dim", dim);
        rc |= add_template_int(config, "K", kernel);

        if rc == 0 {
            rc = sys::mlx_fast_metal_kernel_apply(
                &mut outputs,
                metal_kernel,
                inputs,
                config,
                stream,
            );
        }

        sys::mlx_fast_metal_kernel_config_free(config);
        sys::mlx_vector_array_free(inputs);

        if rc != 0 {
            sys::mlx_vector_array_free(outputs);
            return Err(format!("granite_conv1d_update failed: rc={rc}"));
        }
        if sys::mlx_vector_array_size(outputs) != 2 {
            let size = sys::mlx_vector_array_size(outputs);
            sys::mlx_vector_array_free(outputs);
            return Err(format!(
                "granite_conv1d_update returned {size} outputs, expected 2"
            ));
        }

        let out = vector_array_get(outputs, 0)?;
        let new_state = vector_array_get(outputs, 1)?;
        sys::mlx_vector_array_free(outputs);
        Ok((out, new_state))
    }
}

fn granite_ssm_kernel() -> Result<sys::mlx_fast_metal_kernel, String> {
    static KERNEL_CTX: OnceLock<usize> = OnceLock::new();
    let ctx = *KERNEL_CTX.get_or_init(|| unsafe { create_granite_ssm_kernel() });
    if ctx == 0 {
        return Err("failed to create Granite SSM Metal kernel".into());
    }
    Ok(sys::mlx_fast_metal_kernel {
        ctx: ctx as *mut std::ffi::c_void,
    })
}

unsafe fn create_granite_ssm_kernel() -> usize {
    const SOURCE: &str = r#"
        auto h_idx = thread_position_in_grid.z;
        auto d_idx = thread_position_in_grid.y;
        auto ds_idx = thread_position_in_threadgroup.x;
        constexpr int n_per_t = Ds / 32;

        auto x = X + h_idx * Dh;
        out += h_idx * Dh;
        auto i_state = state_in + h_idx * Dh * Ds;
        auto o_state = state_out + h_idx * Dh * Ds;

        auto dt_ = static_cast<float>(dt[h_idx]);
        auto dA = fast::exp(static_cast<float>(A[h_idx]) * dt_);

        float acc = 0.0;
        auto x_ = static_cast<float>(x[d_idx]);

        for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * ds_idx + i;
            auto idx = d_idx * Ds + s_idx;
            auto dB_by_x = x_ * dt_ * static_cast<float>(B[s_idx]);
            auto state = dA * static_cast<float>(i_state[idx]) + dB_by_x;
            o_state[idx] = static_cast<T>(state);
            acc += state * static_cast<float>(C[s_idx]);
        }
        acc = simd_sum(acc);
        if (thread_index_in_simdgroup == 0) {
            out[d_idx] = static_cast<T>(acc + x_ * static_cast<float>(D[h_idx]));
        }
    "#;

    let name = CString::new("granite_ssm_update").unwrap();
    let source = CString::new(SOURCE).unwrap();
    let header = CString::new("").unwrap();
    let input_names = match vector_string(&["X", "A", "B", "C", "D", "dt", "state_in"]) {
        Ok(v) => v,
        Err(_) => return 0,
    };
    let output_names = match vector_string(&["out", "state_out"]) {
        Ok(v) => v,
        Err(_) => {
            sys::mlx_vector_string_free(input_names);
            return 0;
        }
    };
    let kernel = sys::mlx_fast_metal_kernel_new(
        name.as_ptr(),
        input_names,
        output_names,
        source.as_ptr(),
        header.as_ptr(),
        true,
        false,
    );
    sys::mlx_vector_string_free(input_names);
    sys::mlx_vector_string_free(output_names);
    kernel.ctx as usize
}

fn granite_conv1d_kernel() -> Result<sys::mlx_fast_metal_kernel, String> {
    static KERNEL_CTX: OnceLock<usize> = OnceLock::new();
    let ctx = *KERNEL_CTX.get_or_init(|| unsafe { create_granite_conv1d_kernel() });
    if ctx == 0 {
        return Err("failed to create Granite conv1d Metal kernel".into());
    }
    Ok(sys::mlx_fast_metal_kernel {
        ctx: ctx as *mut std::ffi::c_void,
    })
}

unsafe fn create_granite_conv1d_kernel() -> usize {
    const SOURCE: &str = r#"
        auto dim_idx = thread_position_in_grid.x;
        constexpr int state_len = K - 1;

        float acc = static_cast<float>(X[dim_idx]) *
            static_cast<float>(W[(K - 1) * Dim + dim_idx]);

        for (int k = 0; k < state_len; ++k) {
            auto idx = k * Dim + dim_idx;
            auto v = static_cast<float>(state_in[idx]);
            acc += v * static_cast<float>(W[idx]);
            if (k + 1 < state_len) {
                state_out[idx] = state_in[(k + 1) * Dim + dim_idx];
            } else {
                state_out[idx] = static_cast<T>(X[dim_idx]);
            }
        }

        acc += static_cast<float>(bias[dim_idx]);
        out[dim_idx] = static_cast<T>(acc / (1.0f + fast::exp(-acc)));
    "#;

    let name = CString::new("granite_conv1d_update").unwrap();
    let source = CString::new(SOURCE).unwrap();
    let header = CString::new("").unwrap();
    let input_names = match vector_string(&["X", "state_in", "W", "bias"]) {
        Ok(v) => v,
        Err(_) => return 0,
    };
    let output_names = match vector_string(&["out", "state_out"]) {
        Ok(v) => v,
        Err(_) => {
            sys::mlx_vector_string_free(input_names);
            return 0;
        }
    };
    let kernel = sys::mlx_fast_metal_kernel_new(
        name.as_ptr(),
        input_names,
        output_names,
        source.as_ptr(),
        header.as_ptr(),
        true,
        false,
    );
    sys::mlx_vector_string_free(input_names);
    sys::mlx_vector_string_free(output_names);
    kernel.ctx as usize
}

unsafe fn vector_string(names: &[&str]) -> Result<sys::mlx_vector_string, String> {
    let vec = sys::mlx_vector_string_new();
    for name in names {
        let c_name = CString::new(*name).map_err(|e| e.to_string())?;
        let rc = sys::mlx_vector_string_append_value(vec, c_name.as_ptr());
        if rc != 0 {
            sys::mlx_vector_string_free(vec);
            return Err(format!("mlx_vector_string_append_value failed: rc={rc}"));
        }
    }
    Ok(vec)
}

unsafe fn vector_array(arrays: &[&Array]) -> Result<sys::mlx_vector_array, String> {
    let vec = sys::mlx_vector_array_new();
    for array in arrays {
        let rc = sys::mlx_vector_array_append_value(vec, array.as_ptr());
        if rc != 0 {
            sys::mlx_vector_array_free(vec);
            return Err(format!("mlx_vector_array_append_value failed: rc={rc}"));
        }
    }
    Ok(vec)
}

unsafe fn vector_array_get(vec: sys::mlx_vector_array, index: usize) -> Result<Array, String> {
    let mut out = sys::mlx_array_new();
    let rc = sys::mlx_vector_array_get(&mut out as *mut sys::mlx_array, vec, index);
    if rc != 0 {
        sys::mlx_array_free(out);
        return Err(format!("mlx_vector_array_get({index}) failed: rc={rc}"));
    }
    Ok(Array::from_ptr(out))
}

unsafe fn add_template_dtype(
    config: sys::mlx_fast_metal_kernel_config,
    name: &str,
    dtype: sys::mlx_dtype,
) -> i32 {
    let name = CString::new(name).unwrap();
    sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(config, name.as_ptr(), dtype)
}

unsafe fn add_template_int(
    config: sys::mlx_fast_metal_kernel_config,
    name: &str,
    value: i32,
) -> i32 {
    let name = CString::new(name).unwrap();
    sys::mlx_fast_metal_kernel_config_add_template_arg_int(config, name.as_ptr(), value)
}

fn default_gpu_stream() -> sys::mlx_stream {
    static STREAM_CTX: OnceLock<usize> = OnceLock::new();
    let ctx = *STREAM_CTX.get_or_init(|| unsafe { sys::mlx_default_gpu_stream_new().ctx as usize });
    sys::mlx_stream {
        ctx: ctx as *mut std::ffi::c_void,
    }
}
