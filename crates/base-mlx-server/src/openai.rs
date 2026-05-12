//! OpenAI-compatible route handlers. Skeleton — real generation lands
//! once the engine forward-pass is wired in `base-mlx-core`.

use axum::{http::StatusCode, Json};
use base_mlx_core::registry;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

pub async fn list_models() -> Json<Value> {
    let data: Vec<_> = registry::default_catalog()
        .into_iter()
        .map(|m| {
            json!({
                "id": m.id,
                "object": "model",
                "owned_by": "base-mlx",
                "name": m.name,
                "role": format!("{:?}", m.role).to_lowercase(),
            })
        })
        .collect();
    Json(json!({ "object": "list", "data": data }))
}

#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    #[serde(default)]
    pub messages: Vec<Value>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub tools: Option<Vec<Value>>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
    #[serde(default)]
    pub response_format: Option<Value>,
}

#[derive(Debug, Serialize)]
struct ErrorBody {
    error: ErrorPayload,
}

#[derive(Debug, Serialize)]
struct ErrorPayload {
    message: String,
    #[serde(rename = "type")]
    kind: &'static str,
}

pub async fn chat_completions(
    Json(_req): Json<ChatRequest>,
) -> (StatusCode, Json<ErrorBody>) {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(ErrorBody {
            error: ErrorPayload {
                message: "engine not wired up yet — see ROADMAP.md".into(),
                kind: "not_implemented",
            },
        }),
    )
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingsRequest {
    pub model: String,
    #[serde(default)]
    pub input: Value,
}

pub async fn embeddings(
    Json(_req): Json<EmbeddingsRequest>,
) -> (StatusCode, Json<ErrorBody>) {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(ErrorBody {
            error: ErrorPayload {
                message: "engine not wired up yet — see ROADMAP.md".into(),
                kind: "not_implemented",
            },
        }),
    )
}
