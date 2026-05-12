//! base-mlx-core
//!
//! Inference engine. Loads MLX-quantized transformer weights, runs forward
//! passes via mlx-rs, samples tokens, and serves a per-session generation
//! stream. Higher-level concerns (HTTP, MCP, app shell) live in sibling
//! crates.
//!
//! Public surface is intentionally small while the engine is in flux —
//! everything graduates out of `internal` as it stabilizes.

pub mod error;
pub mod model;
pub mod registry;
pub mod sampler;
pub mod session;
pub mod tokenizer;

pub use error::{Error, Result};
