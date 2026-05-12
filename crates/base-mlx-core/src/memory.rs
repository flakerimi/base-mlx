//! Wrappers around MLX's process-wide memory controls.
//!
//! MLX keeps a free list of Metal buffers (one bucket per size) so it
//! can reuse allocations across steps. The free list is unbounded by
//! default and never shrinks, which during autoregressive decoding —
//! where every step has a slightly different KV-cache shape — leads to
//! steadily climbing RSS. These wrappers let us put a ceiling on the
//! cache and release it after each request.
//!
//! mlx-rs (0.25) does not bind the memory APIs, so we go through
//! mlx-sys directly. The C functions return non-zero on error; we map
//! that to a plain `Err`.

use mlx_sys as sys;

#[derive(Debug, thiserror::Error)]
#[error("mlx memory call failed (rc={0})")]
pub struct MemoryError(i32);

/// Release MLX's Metal buffer free list back to the OS. Safe to call
/// concurrently with idle threads; expected to be called between
/// generations, not in the middle of one.
pub fn clear_cache() -> Result<(), MemoryError> {
    let rc = unsafe { sys::mlx_clear_cache() };
    if rc == 0 {
        Ok(())
    } else {
        Err(MemoryError(rc))
    }
}

/// Cap the size of MLX's Metal buffer free list. Bytes above this are
/// returned to the OS instead of cached. Returns the previous limit.
pub fn set_cache_limit(limit_bytes: usize) -> Result<usize, MemoryError> {
    let mut prev: usize = 0;
    let rc = unsafe { sys::mlx_set_cache_limit(&mut prev as *mut usize, limit_bytes) };
    if rc == 0 {
        Ok(prev)
    } else {
        Err(MemoryError(rc))
    }
}

/// Bytes currently held in live arrays (not the free-list cache).
pub fn active_bytes() -> Result<usize, MemoryError> {
    let mut out: usize = 0;
    let rc = unsafe { sys::mlx_get_active_memory(&mut out as *mut usize) };
    if rc == 0 {
        Ok(out)
    } else {
        Err(MemoryError(rc))
    }
}

/// Bytes currently held in the Metal buffer free list.
pub fn cache_bytes() -> Result<usize, MemoryError> {
    let mut out: usize = 0;
    let rc = unsafe { sys::mlx_get_cache_memory(&mut out as *mut usize) };
    if rc == 0 {
        Ok(out)
    } else {
        Err(MemoryError(rc))
    }
}
