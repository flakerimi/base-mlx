//! Model loading + forward pass.
//!
//! Each supported architecture (qwen3, llama3, gemma3, …) gets its own
//! submodule that implements `Architecture`. The engine wires the right
//! architecture from the safetensors config + weights.

use crate::Result;

pub mod qwen3;

/// Minimal architecture trait. Concrete implementations own their own
/// weight layout, KV cache shape, and forward pass. Kept opaque on
/// purpose so each architecture can use mlx-rs in whatever way fits.
pub trait Architecture: Send + Sync {
    /// Architecture identifier (matches HF config `model_type`).
    fn id(&self) -> &'static str;

    /// Run one decode step. Returns next-token logits.
    /// Stub for now; full signature lands with the first real arch.
    fn step(&mut self, _token: u32) -> Result<Vec<f32>> {
        Err(crate::Error::Inference(
            "step() not implemented yet".into(),
        ))
    }
}
