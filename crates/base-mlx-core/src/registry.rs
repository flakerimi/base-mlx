//! Model registry: pull weights from HuggingFace, cache to disk, list
//! installed models, hot-swap, idle-unload. v1 ships with three curated
//! entries (chat / code / embed); users can add custom ones later.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    /// Stable id used by the API (e.g. "qwen3-4b-instruct").
    pub id: String,
    /// Human-readable label shown in the menu bar.
    pub name: String,
    /// HuggingFace repo (e.g. "mlx-community/Qwen3-4B-Instruct-2507-4bit").
    pub hf_repo: String,
    /// What this model is for: chat | code | embed | vision | rerank | …
    pub role: ModelRole,
    /// Approximate working-set RAM in bytes. Used by the memory scheduler.
    pub ram_estimate: u64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ModelRole {
    Chat,
    Code,
    Embed,
    Vision,
    Rerank,
}

/// Default v1 catalog. Three models, curated.
pub fn default_catalog() -> Vec<ModelEntry> {
    vec![
        ModelEntry {
            id: "qwen3-4b-instruct".into(),
            name: "Qwen 3 4B Instruct".into(),
            hf_repo: "mlx-community/Qwen3-4B-Instruct-2507-4bit".into(),
            role: ModelRole::Chat,
            ram_estimate: 3_000_000_000,
        },
        ModelEntry {
            id: "qwen3-coder-30b".into(),
            name: "Qwen 3 Coder 30B MoE".into(),
            hf_repo: "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit".into(),
            role: ModelRole::Code,
            ram_estimate: 18_000_000_000,
        },
        ModelEntry {
            id: "nomic-embed-text".into(),
            name: "Nomic Embed Text v1.5".into(),
            hf_repo: "mlx-community/nomic-embed-text-v1.5".into(),
            role: ModelRole::Embed,
            ram_estimate: 200_000_000,
        },
    ]
}

/// Where pulled weights live on disk. Defaults to `~/.cache/base-mlx/models`.
pub fn cache_dir() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("base-mlx")
        .join("models")
}
