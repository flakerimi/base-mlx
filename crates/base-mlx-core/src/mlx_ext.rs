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
        let stream = sys::mlx_default_gpu_stream_new();
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
        // Free the stream handle we just created. The op itself was
        // submitted async; freeing the handle doesn't cancel the op.
        sys::mlx_stream_free(stream);
        if rc != 0 {
            sys::mlx_array_free(out);
            return Err(format!("mlx_slice_update failed: rc={}", rc));
        }
        Ok(Array::from_ptr(out))
    }
}
