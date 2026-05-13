//! Model loading + forward pass.
//!
//! Each supported architecture (qwen3, llama3, gemma3, …) gets its own
//! submodule that implements `Architecture`. The engine wires the right
//! architecture from the safetensors config + weights.

pub mod config;
pub mod gemma;
pub mod granite;
pub mod kernels;
pub mod qwen3;

use crate::{Error, Result};
use std::path::Path;

pub use config::{ModelConfig, QuantConfig};
pub use gemma::Gemma4;
pub use granite::Granite;
pub use qwen3::Qwen3;

#[derive(Debug)]
pub enum Architecture {
    Qwen3(Qwen3),
    Gemma4(Gemma4),
    Granite(Granite),
}

#[derive(Debug)]
pub enum ArchitectureCache {
    Qwen3(qwen3::KvCache),
    Gemma4(gemma::KvCache),
    Granite(granite::KvCache),
}

impl Architecture {
    pub fn load(dir: &Path, cfg: ModelConfig) -> Result<Self> {
        match cfg.model_type.as_str() {
            "qwen3" => Ok(Self::Qwen3(Qwen3::load(dir, cfg)?)),
            "gemma4" => Ok(Self::Gemma4(Gemma4::load(dir, cfg)?)),
            "granitemoehybrid" => Ok(Self::Granite(Granite::load(dir, cfg)?)),
            other => Err(Error::ModelLoad(format!("unsupported model_type: {other}"))),
        }
    }

    pub fn new_cache(&self) -> ArchitectureCache {
        match self {
            Self::Qwen3(_) => ArchitectureCache::Qwen3(qwen3::KvCache::new()),
            Self::Gemma4(_) => ArchitectureCache::Gemma4(gemma::KvCache::new()),
            Self::Granite(_) => ArchitectureCache::Granite(granite::KvCache::new()),
        }
    }

    pub fn forward(&self, tokens: &[u32], cache: &mut ArchitectureCache) -> Result<mlx_rs::Array> {
        match (self, cache) {
            (Self::Qwen3(model), ArchitectureCache::Qwen3(cache)) => model.forward(tokens, cache),
            (Self::Gemma4(model), ArchitectureCache::Gemma4(cache)) => model.forward(tokens, cache),
            (Self::Granite(model), ArchitectureCache::Granite(cache)) => {
                model.forward(tokens, cache)
            }
            _ => Err(Error::Inference("model/cache architecture mismatch".into())),
        }
    }

    pub fn forward_multi(
        &self,
        tokens: &[u32],
        cache: &mut ArchitectureCache,
    ) -> Result<mlx_rs::Array> {
        match (self, cache) {
            (Self::Qwen3(model), ArchitectureCache::Qwen3(cache)) => {
                model.forward_multi(tokens, cache)
            }
            (Self::Gemma4(model), ArchitectureCache::Gemma4(cache)) => {
                model.forward_multi(tokens, cache)
            }
            (Self::Granite(model), ArchitectureCache::Granite(cache)) => {
                model.forward_multi(tokens, cache)
            }
            _ => Err(Error::Inference("model/cache architecture mismatch".into())),
        }
    }

    pub fn warmup_token(&self) -> u32 {
        match self {
            Self::Qwen3(_) => 151643,
            Self::Gemma4(_) => 2,
            Self::Granite(_) => 100257,
        }
    }
}

impl ArchitectureCache {
    pub fn reset(&mut self) {
        match self {
            Self::Qwen3(cache) => cache.reset(),
            Self::Gemma4(cache) => cache.reset(),
            Self::Granite(cache) => cache.reset(),
        }
    }

    pub fn truncate(&mut self, target_seq_len: usize) {
        match self {
            Self::Qwen3(cache) => cache.truncate(target_seq_len),
            Self::Gemma4(cache) => cache.truncate(target_seq_len),
            Self::Granite(cache) => cache.truncate(target_seq_len),
        }
    }
}
