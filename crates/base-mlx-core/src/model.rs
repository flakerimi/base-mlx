//! Model loading + forward pass.
//!
//! Each supported architecture (qwen3, llama3, gemma3, …) gets its own
//! submodule that implements `Architecture`. The engine wires the right
//! architecture from the safetensors config + weights.

pub mod config;
pub mod qwen3;

pub use config::{ModelConfig, QuantConfig};
pub use qwen3::Qwen3;
