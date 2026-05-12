//! OpenAI-compatible route handlers.
//!
//! Both `/v1/chat/completions` modes (one-shot + streaming) run entirely
//! inside `spawn_blocking` so the mutex around the loaded model never
//! crosses an `.await` (mlx-rs `Array` is `Send` but not `Sync`).

use crate::state::AppState;
use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive},
        IntoResponse, Response, Sse,
    },
    Json,
};
use base_mlx_core::chat_template::ChatMessage;
use base_mlx_core::engine::LoadedModel;
use base_mlx_core::registry;
use base_mlx_core::sampler::SamplingParams;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::convert::Infallible;
use std::time::SystemTime;
use uuid::Uuid;

// ─── /v1/models ─────────────────────────────────────────────────────────────

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

// ─── /v1/chat/completions ───────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    #[serde(default)]
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub tools: Option<Vec<Value>>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
    #[serde(default)]
    pub response_format: Option<Value>,
}

#[derive(Debug, Serialize)]
pub struct ErrorBody {
    pub error: ErrorPayload,
}

#[derive(Debug, Serialize)]
pub struct ErrorPayload {
    pub message: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
}

fn err(code: StatusCode, msg: impl Into<String>, kind: &'static str) -> Response {
    (
        code,
        Json(ErrorBody {
            error: ErrorPayload {
                message: msg.into(),
                kind,
            },
        }),
    )
        .into_response()
}

pub async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatRequest>,
) -> Response {
    if req.messages.is_empty() {
        return err(
            StatusCode::BAD_REQUEST,
            "messages is required and non-empty",
            "invalid_request",
        );
    }

    let params = SamplingParams {
        temperature: req.temperature.unwrap_or(0.7),
        top_p: req.top_p.unwrap_or(0.95),
        top_k: 64,
        repetition_penalty: 1.0,
        seed: req.seed,
        grammar: None,
    };
    let max_tokens = req.max_tokens.unwrap_or(512);

    if req.stream {
        stream_response(state, req.model, req.messages, params, max_tokens).await
    } else {
        oneshot_response(state, req.model, req.messages, params, max_tokens).await
    }
}

/// Take the mutex, load the model if needed, run `f` with a &LoadedModel.
/// Caller must already be on a blocking thread.
fn with_loaded<R>(
    state: &AppState,
    model_id: &str,
    f: impl FnOnce(&LoadedModel) -> R,
) -> Result<R, base_mlx_core::Error> {
    let mut slot = state.model.lock().expect("model lock poisoned");
    let need_load = match slot.as_ref() {
        Some((id, _)) => id != model_id,
        None => true,
    };
    if need_load {
        slot.take(); // free RAM before loading the next model
        let loaded = LoadedModel::load(model_id)?;
        *slot = Some((model_id.to_string(), loaded));
    }
    let (_, loaded) = slot.as_ref().expect("just inserted");
    Ok(f(loaded))
}

async fn oneshot_response(
    state: AppState,
    model_id: String,
    messages: Vec<ChatMessage>,
    params: SamplingParams,
    max_tokens: u32,
) -> Response {
    let model_id_for_payload = model_id.clone();
    let result = tokio::task::spawn_blocking(move || -> Result<_, base_mlx_core::Error> {
        with_loaded(&state, &model_id, |loaded| {
            let prompt = loaded.render_chat(&messages);
            let tokens = loaded.encode(&prompt)?;
            loaded.generate(&tokens, &params, max_tokens, |_, _| {})
        })?
    })
    .await;

    let gen = match result {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => {
            return err(
                StatusCode::INTERNAL_SERVER_ERROR,
                e.to_string(),
                "generation_failed",
            )
        }
        Err(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string(), "join_error"),
    };

    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    Json(json!({
        "id": format!("chatcmpl-{}", Uuid::new_v4().simple()),
        "object": "chat.completion",
        "created": now,
        "model": model_id_for_payload,
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": gen.text },
            "finish_reason": gen.finish_reason,
        }],
        "usage": {
            "prompt_tokens": gen.prompt_tokens,
            "completion_tokens": gen.completion_tokens,
            "total_tokens": gen.prompt_tokens + gen.completion_tokens,
        },
    }))
    .into_response()
}

async fn stream_response(
    state: AppState,
    model_id: String,
    messages: Vec<ChatMessage>,
    params: SamplingParams,
    max_tokens: u32,
) -> Response {
    let id = format!("chatcmpl-{}", Uuid::new_v4().simple());
    let created = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Result<Event, Infallible>>();

    // OpenAI-style "role: assistant" preamble.
    let preamble = json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_id,
        "choices": [{
            "index": 0,
            "delta": { "role": "assistant" },
            "finish_reason": null,
        }],
    });
    let _ = tx.send(Ok(Event::default().data(preamble.to_string())));

    let id2 = id.clone();
    let model_id2 = model_id.clone();
    tokio::task::spawn_blocking(move || {
        let res = with_loaded(&state, &model_id, |loaded| {
            let prompt = loaded.render_chat(&messages);
            let tokens = match loaded.encode(&prompt) {
                Ok(t) => t,
                Err(e) => {
                    push_error(&tx, &e);
                    return None;
                }
            };
            let tx_for_cb = tx.clone();
            let id3 = id2.clone();
            let model_id3 = model_id2.clone();
            let r = loaded.generate(&tokens, &params, max_tokens, move |piece, _| {
                let chunk = json!({
                    "id": id3,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_id3,
                    "choices": [{
                        "index": 0,
                        "delta": { "content": piece },
                        "finish_reason": null,
                    }],
                });
                let _ = tx_for_cb.send(Ok(Event::default().data(chunk.to_string())));
            });
            Some(r)
        });

        match res {
            Ok(Some(Ok(gen))) => {
                let stop = json!({
                    "id": id2,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_id2,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": gen.finish_reason,
                    }],
                    "usage": {
                        "prompt_tokens": gen.prompt_tokens,
                        "completion_tokens": gen.completion_tokens,
                        "total_tokens": gen.prompt_tokens + gen.completion_tokens,
                    },
                });
                let _ = tx.send(Ok(Event::default().data(stop.to_string())));
                let _ = tx.send(Ok(Event::default().data("[DONE]")));
            }
            Ok(Some(Err(e))) | Err(e) => push_error(&tx, &e),
            Ok(None) => {}
        }
    });

    let stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx);
    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

fn push_error(
    tx: &tokio::sync::mpsc::UnboundedSender<Result<Event, Infallible>>,
    e: &base_mlx_core::Error,
) {
    let _ = tx.send(Ok(Event::default()
        .data(json!({ "error": { "message": e.to_string() } }).to_string())));
}

// ─── /v1/embeddings (not wired in v1) ───────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct EmbeddingsRequest {
    pub model: String,
    #[serde(default)]
    pub input: Value,
}

pub async fn embeddings(Json(_req): Json<EmbeddingsRequest>) -> Response {
    err(
        StatusCode::NOT_IMPLEMENTED,
        "embeddings not wired in v1 — chat first",
        "not_implemented",
    )
}
