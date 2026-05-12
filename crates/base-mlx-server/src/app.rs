use crate::state::{AppInner, AppState};
use anyhow::Result;
use axum::{
    extract::State,
    routing::{get, post},
    Json, Router,
};
use serde_json::{json, Value};
use std::net::SocketAddr;

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub addr: SocketAddr,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            // 11435 intentionally — Ollama owns 11434 on many machines.
            addr: "127.0.0.1:11435".parse().unwrap(),
        }
    }
}

pub async fn serve(cfg: ServerConfig) -> Result<()> {
    // Cap MLX's Metal free-list at 8 GiB. The default is unbounded and
    // grows unbounded too (one buffer per unique tensor shape) — during
    // long decode runs the process RSS climbs past 20 GB without this.
    // 1 GiB (our original cap) was too small for 1500-token KV cache
    // growth on a 4B model: every step's concat allocates a slightly
    // bigger buffer, the free-list spills, and the next request inherits
    // a fragmented heap. 8 GiB gives MLX room to reuse the full growth
    // curve without ever returning buffers to the OS mid-request, while
    // still leaving 24 GiB of unified memory for everything else on a
    // 32 GB machine.
    if let Err(e) = base_mlx_core::memory::set_cache_limit(16 << 30) {
        tracing::warn!(error = %e, "failed to cap MLX cache");
    }

    let state: AppState = AppInner::new();

    let app = Router::new()
        .route("/", get(root))
        .route("/v1/models", get(crate::openai::list_models))
        .route(
            "/v1/chat/completions",
            post(crate::openai::chat_completions),
        )
        .route("/v1/embeddings", post(crate::openai::embeddings))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(cfg.addr).await?;
    tracing::info!(addr = %cfg.addr, "base-mlx serving");
    axum::serve(listener, app).await?;
    Ok(())
}

async fn root(State(_state): State<AppState>) -> Json<Value> {
    Json(json!({
        "name": "base-mlx",
        "version": env!("CARGO_PKG_VERSION"),
        "endpoints": [
            "/v1/models",
            "/v1/chat/completions",
            "/v1/embeddings",
        ],
    }))
}
