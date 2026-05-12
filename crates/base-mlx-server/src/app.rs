use anyhow::Result;
use axum::{
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
    let app = Router::new()
        .route("/", get(root))
        .route("/v1/models", get(crate::openai::list_models))
        .route(
            "/v1/chat/completions",
            post(crate::openai::chat_completions),
        )
        .route("/v1/embeddings", post(crate::openai::embeddings));

    let listener = tokio::net::TcpListener::bind(cfg.addr).await?;
    tracing::info!(addr = %cfg.addr, "base-mlx serving");
    axum::serve(listener, app).await?;
    Ok(())
}

async fn root() -> Json<Value> {
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
