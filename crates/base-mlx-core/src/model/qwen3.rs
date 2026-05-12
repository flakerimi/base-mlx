//! Qwen3-Instruct (dense) — the v1 reference architecture.
//!
//! Forward pass + KV cache land here. For now this is a stub; the actual
//! mlx-rs wiring follows once we've validated the workspace builds.

use super::Architecture;

pub struct Qwen3 {
    // Hyperparameters + tensors will live here. Intentionally empty for
    // the initial scaffold so `cargo check` stays green.
}

impl Qwen3 {
    pub fn placeholder() -> Self {
        Self {}
    }
}

impl Architecture for Qwen3 {
    fn id(&self) -> &'static str {
        "qwen3"
    }
}
