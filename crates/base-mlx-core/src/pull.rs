//! Pull MLX-quantized weights from HuggingFace.
//!
//! mlx-community repos all share a small set of files we care about:
//!   - `config.json`         — architecture hyperparameters
//!   - `tokenizer.json`      — HF tokenizer
//!   - `tokenizer_config.json` (chat template lives here)
//!   - `model.safetensors`   — either single-file or sharded as
//!     `model-00001-of-NNNNN.safetensors` + `model.safetensors.index.json`
//!
//! We download whatever's present. Missing optional files are not fatal —
//! the caller decides which ones it needs.
//!
//! Cache layout: `<cache>/base-mlx/models/<owner>--<repo>/`.

use crate::Result;
use std::path::{Path, PathBuf};
use tracing::{info, warn};

use crate::registry::cache_dir;

/// Directories searched when looking up a model that may already be on
/// disk. Order is important: the first hit wins. Adding LM Studio's
/// models tree lets us reuse pulls a user already made, so we don't
/// duplicate 2 GB of weights for the common case.
pub fn search_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    dirs.push(cache_dir());
    if let Some(home) = dirs::home_dir() {
        dirs.push(home.join(".lmstudio").join("models"));
    }
    dirs
}

/// Look for `hf_repo` in any known local cache. Returns the directory
/// containing `config.json` if found.
pub fn find_local(hf_repo: &str) -> Option<PathBuf> {
    let mangled = hf_repo.replace('/', "--");
    let (owner, name) = hf_repo.split_once('/').unwrap_or((hf_repo, hf_repo));

    for base in search_dirs() {
        for candidate in [
            base.join(&mangled),     // base-mlx layout
            base.join(owner).join(name), // LM Studio layout
        ] {
            if candidate.join("config.json").exists() {
                return Some(candidate);
            }
        }
    }
    None
}

/// Filenames worth attempting; absence of any single file is non-fatal so
/// we don't conflate "model has no chat template" with "download failed."
const CANDIDATE_FILES: &[&str] = &[
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "model.safetensors",
    "model.safetensors.index.json",
];

/// Where this repo's files live on disk after a successful pull.
pub fn repo_dir(hf_repo: &str) -> PathBuf {
    let mangled = hf_repo.replace('/', "--");
    cache_dir().join(mangled)
}

#[derive(Debug)]
pub struct PullReport {
    pub repo: String,
    pub dir: PathBuf,
    pub files: Vec<PathBuf>,
}

/// Download a HuggingFace repo (e.g. `mlx-community/Qwen3-4B-Instruct-2507-4bit`)
/// into the local cache. Pulls the common metadata files, plus either
/// `model.safetensors` (single-file) or every shard listed in
/// `model.safetensors.index.json` (sharded).
pub async fn pull(hf_repo: &str) -> Result<PullReport> {
    // Short-circuit if a complete copy already exists locally — saves
    // pulling 2 GB+ when LM Studio (or a previous run) already has it.
    if let Some(dir) = find_local(hf_repo) {
        info!(dir = %dir.display(), "found existing local copy; skipping download");
        let files: Vec<_> = std::fs::read_dir(&dir)
            .ok()
            .into_iter()
            .flatten()
            .filter_map(|e| e.ok().map(|e| e.path()))
            .collect();
        return Ok(PullReport {
            repo: hf_repo.to_string(),
            dir,
            files,
        });
    }

    let api = hf_hub::api::tokio::Api::new()
        .map_err(|e| crate::Error::ModelLoad(e.to_string()))?;
    let repo = api.model(hf_repo.to_string());

    let dest = repo_dir(hf_repo);
    std::fs::create_dir_all(&dest)?;

    let mut pulled = Vec::new();

    for f in CANDIDATE_FILES {
        match repo.get(f).await {
            Ok(src) => {
                let dst = dest.join(f);
                copy_into(&src, &dst)?;
                info!(file = %f, "pulled");
                pulled.push(dst);
            }
            Err(e) => {
                warn!(file = %f, error = %e, "skipped");
            }
        }
    }

    // If the model is sharded, the index.json names every shard. Pull each.
    let index_path = dest.join("model.safetensors.index.json");
    if index_path.exists() {
        let raw = std::fs::read_to_string(&index_path)?;
        let json: serde_json::Value = serde_json::from_str(&raw)?;
        if let Some(map) = json.get("weight_map").and_then(|v| v.as_object()) {
            let shards: std::collections::BTreeSet<String> = map
                .values()
                .filter_map(|v| v.as_str().map(String::from))
                .collect();
            for shard in shards {
                if dest.join(&shard).exists() {
                    continue;
                }
                match repo.get(&shard).await {
                    Ok(src) => {
                        let dst = dest.join(&shard);
                        copy_into(&src, &dst)?;
                        info!(file = %shard, "pulled shard");
                        pulled.push(dst);
                    }
                    Err(e) => warn!(file = %shard, error = %e, "shard failed"),
                }
            }
        }
    }

    Ok(PullReport {
        repo: hf_repo.to_string(),
        dir: dest,
        files: pulled,
    })
}

/// hf-hub returns a path inside its own cache. We mirror the file into
/// our cache via hard link (free) falling back to copy. Hardlink is safe
/// here because both paths live under the user's home filesystem.
fn copy_into(src: &Path, dst: &Path) -> std::io::Result<()> {
    if dst.exists() {
        return Ok(());
    }
    if let Some(parent) = dst.parent() {
        std::fs::create_dir_all(parent)?;
    }
    match std::fs::hard_link(src, dst) {
        Ok(()) => Ok(()),
        Err(_) => {
            std::fs::copy(src, dst)?;
            Ok(())
        }
    }
}
