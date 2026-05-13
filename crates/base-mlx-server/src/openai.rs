//! OpenAI-compatible route handlers.
//!
//! Both `/v1/chat/completions` modes (one-shot + streaming) run entirely
//! inside `spawn_blocking` so the mutex around the loaded model never
//! crosses an `.await` (mlx-rs `Array` is `Send` but not `Sync`).

use crate::state::AppState;
use axum::{
    extract::{Query, State},
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
use std::time::{Instant, SystemTime};
use tracing::info;
use uuid::Uuid;

/// Log a one-line memory + throughput snapshot around a generation.
/// `tag` is "oneshot" or "stream"; `f` runs the actual generate call.
/// We sample MLX's active + cache bytes immediately before and after so
/// run-over-run drift is visible without extra tooling.
fn instrumented<F, T>(tag: &'static str, f: F) -> T
where
    F: FnOnce() -> T,
{
    let active_before = base_mlx_core::memory::active_bytes().unwrap_or(0);
    let cache_before = base_mlx_core::memory::cache_bytes().unwrap_or(0);
    let t0 = Instant::now();
    let out = f();
    let elapsed = t0.elapsed();
    let active_after = base_mlx_core::memory::active_bytes().unwrap_or(0);
    let cache_after = base_mlx_core::memory::cache_bytes().unwrap_or(0);
    info!(
        tag = tag,
        elapsed_ms = elapsed.as_millis() as u64,
        active_mb_before = active_before / 1024 / 1024,
        active_mb_after = active_after / 1024 / 1024,
        cache_mb_before = cache_before / 1024 / 1024,
        cache_mb_after = cache_after / 1024 / 1024,
        "gen done",
    );
    out
}

// ─── /v1/models ─────────────────────────────────────────────────────────────

pub async fn list_models(State(state): State<AppState>) -> Json<Value> {
    use base_mlx_core::pull;

    // Currently loaded model ids. Multiple may be resident (target +
    // draft for speculative decoding). Empty means first chat request
    // will trigger a load. We pick the first key for the legacy scalar
    // `loaded` field at the response root; per-entry `loaded:` flags
    // cover the full picture.
    let loaded_ids: Vec<String> = state
        .models
        .lock()
        .ok()
        .map(|g| g.keys().cloned().collect())
        .unwrap_or_default();
    let loaded_id: Option<String> = loaded_ids.first().cloned();

    // Enumerate every locally-available model — anything with a
    // `config.json` under our cache or LM Studio's models tree. Each
    // entry's `id` is the path-derived name (e.g. `Qwen3-4B-Instruct-2507-4bit`
    // or `mlx-community/Qwen3-4B-Instruct-2507-4bit`) so callers can hand
    // it back to chat completions and resolve cleanly via the fuzzy
    // matcher. We also surface the curated catalog ids when there's a
    // local hit for them — that gives consumers the friendly short
    // names too.
    let catalog = registry::default_catalog();
    let mut seen = std::collections::BTreeSet::<String>::new();
    let mut data = Vec::<Value>::new();

    // First: catalog entries that resolve to a local copy.
    for m in &catalog {
        if pull::find_local_exact(&m.hf_repo).is_some() {
            seen.insert(m.id.clone());
            let is_loaded = loaded_ids.iter().any(|x| x == &m.id);
            data.push(json!({
                "id": m.id,
                "object": "model",
                "owned_by": "base-mlx",
                "name": m.name,
                "role": format!("{:?}", m.role).to_lowercase(),
                "local": true,
                "loaded": is_loaded,
            }));
        }
    }

    // Then: every other on-disk model the catalog doesn't cover.
    for dir in pull::local_models() {
        // Compute a stable id from the path: `<parent>/<leaf>` when
        // hosted under an org dir (LM Studio layout), otherwise the
        // leaf directly.
        let leaf = dir.file_name().and_then(|s| s.to_str()).unwrap_or("");
        let parent = dir
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|s| s.to_str())
            .unwrap_or("");
        // Skip our own cache root names; only namespace when parent
        // looks like an HF owner (no spaces, contains alphanum).
        let id = if !parent.is_empty()
            && parent != "models"
            && parent.chars().any(|c| c.is_alphabetic())
            && !parent.starts_with('.')
        {
            format!("{parent}/{leaf}")
        } else {
            leaf.to_string()
        };
        if id.is_empty() || seen.contains(&id) {
            continue;
        }
        // Also skip if any catalog id already represents this dir.
        let already = catalog.iter().any(|m| {
            pull::find_local_exact(&m.hf_repo)
                .map(|d| d == dir)
                .unwrap_or(false)
                && seen.contains(&m.id)
        });
        if already {
            continue;
        }
        seen.insert(id.clone());
        let is_loaded = loaded_ids.iter().any(|x| x == &id);
        data.push(json!({
            "id": id,
            "object": "model",
            "owned_by": "base-mlx",
            "local": true,
            "loaded": is_loaded,
        }));
    }

    Json(json!({
        "object": "list",
        "data": data,
        "loaded": loaded_id,
    }))
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

/// Query-string knobs that don't fit the OpenAI JSON body. Right now
/// just `?spec=<draft_id>` to opt into speculative decoding per request.
#[derive(Debug, Default, Deserialize)]
pub struct ChatQuery {
    #[serde(default)]
    pub spec: Option<String>,
}

#[derive(Debug, Clone)]
struct GenerationOptions {
    tools: Option<Vec<Value>>,
    params: SamplingParams,
    max_tokens: u32,
    spec: Option<String>,
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
    Query(q): Query<ChatQuery>,
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

    // Soft enforcement of response_format. Real grammar-constrained
    // sampling lands in a separate milestone; for now we ask the model
    // for valid JSON via a system-prompt nudge, then validate and
    // optionally retry once. Schema is exposed in the nudge so the
    // model can shape its output even without enforced constraints.
    let messages = inject_response_format(req.messages.clone(), req.response_format.as_ref());
    let opts = GenerationOptions {
        tools: req.tools.clone(),
        params,
        max_tokens,
        spec: q.spec.clone(),
    };

    if req.stream {
        stream_response(state, req.model, messages, opts).await
    } else {
        oneshot_response(
            state,
            req.model,
            messages,
            opts,
            req.response_format.clone(),
        )
        .await
    }
}

fn inject_response_format(
    mut messages: Vec<ChatMessage>,
    response_format: Option<&Value>,
) -> Vec<ChatMessage> {
    let Some(rf) = response_format else {
        return messages;
    };
    let kind = rf.get("type").and_then(|v| v.as_str()).unwrap_or("");
    let nudge = match kind {
        "json_object" => Some(String::from(
            "\n\nIMPORTANT: respond with a single valid JSON object. No prose, no markdown fences.",
        )),
        "json_schema" => {
            let schema = rf
                .get("json_schema")
                .and_then(|j| j.get("schema"))
                .cloned()
                .unwrap_or(Value::Null);
            Some(format!(
                "\n\nIMPORTANT: respond with a single JSON value that conforms to this JSON Schema. No prose, no markdown fences.\nSchema:\n{}",
                serde_json::to_string(&schema).unwrap_or_else(|_| "{}".into())
            ))
        }
        _ => None,
    };
    let Some(nudge) = nudge else {
        return messages;
    };

    if let Some(sys) = messages.iter_mut().find(|m| m.role == "system") {
        sys.content.push_str(&nudge);
    } else {
        messages.insert(
            0,
            ChatMessage {
                role: "system".into(),
                content: format!("You are a helpful assistant.{nudge}"),
                tool_call_id: None,
                name: None,
                tool_calls: None,
            },
        );
    }
    messages
}

/// Take the mutex, load the model if needed, run `f` with a &LoadedModel.
/// Caller must already be on a blocking thread.
fn with_loaded<R>(
    state: &AppState,
    model_id: &str,
    f: impl FnOnce(&LoadedModel) -> R,
) -> Result<R, base_mlx_core::Error> {
    let mut map = state.models.lock().expect("model lock poisoned");
    if !map.contains_key(model_id) {
        let loaded = LoadedModel::load(model_id)?;
        map.insert(model_id.to_string(), loaded);
    }
    let loaded = map.get(model_id).expect("just inserted");
    Ok(f(loaded))
}

/// Same as `with_loaded` but yields both a target and a draft model.
/// Used by speculative decoding. Both are loaded if absent; both stay
/// resident across requests (no LRU yet).
fn with_target_and_draft<R>(
    state: &AppState,
    target_id: &str,
    draft_id: &str,
    f: impl FnOnce(&LoadedModel, &LoadedModel) -> R,
) -> Result<R, base_mlx_core::Error> {
    let mut map = state.models.lock().expect("model lock poisoned");
    if !map.contains_key(target_id) {
        map.insert(target_id.to_string(), LoadedModel::load(target_id)?);
    }
    if !map.contains_key(draft_id) {
        map.insert(draft_id.to_string(), LoadedModel::load(draft_id)?);
    }
    let target = map.get(target_id).expect("just inserted");
    let draft = map.get(draft_id).expect("just inserted");
    base_mlx_core::speculative::check_compatible(target, draft)?;
    Ok(f(target, draft))
}

async fn oneshot_response(
    state: AppState,
    model_id: String,
    messages: Vec<ChatMessage>,
    opts: GenerationOptions,
    response_format: Option<Value>,
) -> Response {
    let model_id_for_payload = model_id.clone();
    let tools_slice = opts.tools.clone();
    let result = tokio::task::spawn_blocking(move || -> Result<_, base_mlx_core::Error> {
        match opts.spec {
            Some(draft_id) => {
                with_target_and_draft(&state, &model_id, &draft_id, |target, draft| {
                    let prompt = target.render_chat(&messages, tools_slice.as_deref());
                    let tokens = target.encode(&prompt)?;
                    instrumented("oneshot-spec", || {
                        base_mlx_core::speculative::generate(
                            target,
                            draft,
                            &tokens,
                            &opts.params,
                            opts.max_tokens,
                            |_, _| {},
                        )
                    })
                })?
            }
            None => with_loaded(&state, &model_id, |loaded| {
                let prompt = loaded.render_chat(&messages, tools_slice.as_deref());
                let tokens = loaded.encode(&prompt)?;
                instrumented("oneshot", || {
                    loaded.generate_text(&tokens, &opts.params, opts.max_tokens)
                })
            })?,
        }
    })
    .await;

    let mut gen = match result {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => {
            return err(
                StatusCode::INTERNAL_SERVER_ERROR,
                e.to_string(),
                "generation_failed",
            )
        }
        Err(e) => {
            return err(
                StatusCode::INTERNAL_SERVER_ERROR,
                e.to_string(),
                "join_error",
            )
        }
    };

    // Post-hoc JSON validation when the caller asked for structured
    // output. On failure, return the raw text but flag it via a custom
    // header — a real retry pass would re-run with the error message
    // appended. Keeping v1 deterministic; retries are easy to add.
    let json_valid = match response_format
        .as_ref()
        .and_then(|r| r.get("type"))
        .and_then(|t| t.as_str())
    {
        Some("json_object") | Some("json_schema") => {
            Some(serde_json::from_str::<Value>(gen.text.trim()).is_ok())
        }
        _ => None,
    };
    if json_valid == Some(false) {
        // Heuristic cleanup: try to extract the first JSON object.
        if let Some(cleaned) = extract_first_json(&gen.text) {
            gen.text = cleaned;
        }
    }

    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let mut message = json!({ "role": "assistant" });
    if !gen.tool_calls.is_empty() {
        // OpenAI: when tool_calls is present, content can be null.
        message["content"] = Value::Null;
        message["tool_calls"] = serde_json::to_value(&gen.tool_calls).unwrap_or(Value::Null);
    } else {
        message["content"] = Value::String(gen.text);
    }

    Json(json!({
        "id": format!("chatcmpl-{}", Uuid::new_v4().simple()),
        "object": "chat.completion",
        "created": now,
        "model": model_id_for_payload,
        "choices": [{
            "index": 0,
            "message": message,
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

/// Heuristic: grab the first balanced JSON object from text. Bails out
/// if it can't find one. Used when a model returned JSON with stray
/// markdown fences or prose around it.
fn extract_first_json(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let mut start = None;
    let mut depth = 0i32;
    let mut in_str = false;
    let mut escape = false;
    for (i, &b) in bytes.iter().enumerate() {
        if escape {
            escape = false;
            continue;
        }
        if in_str {
            match b {
                b'\\' => escape = true,
                b'"' => in_str = false,
                _ => {}
            }
            continue;
        }
        match b {
            b'"' => in_str = true,
            b'{' => {
                if start.is_none() {
                    start = Some(i);
                }
                depth += 1;
            }
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    if let Some(s) = start {
                        return Some(text[s..=i].to_string());
                    }
                }
            }
            _ => {}
        }
    }
    None
}

async fn stream_response(
    state: AppState,
    model_id: String,
    messages: Vec<ChatMessage>,
    opts: GenerationOptions,
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
    let tools_for_task = opts.tools.clone();
    tokio::task::spawn_blocking(move || {
        // The streaming callback is identical between single-model and
        // spec-dec paths — build it once and clone the captured channel
        // into each closure.
        let has_tools = tools_for_task.as_ref().is_some_and(|t| !t.is_empty());
        let make_cb = || {
            let tx_for_cb = tx.clone();
            let id3 = id2.clone();
            let model_id3 = model_id2.clone();
            move |piece: &str, _tok: u32| {
                if has_tools {
                    return;
                }
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
            }
        };

        let res = match opts.spec {
            Some(draft_id) => {
                with_target_and_draft(&state, &model_id, &draft_id, |target, draft| {
                    let prompt = target.render_chat(&messages, tools_for_task.as_deref());
                    let tokens = match target.encode(&prompt) {
                        Ok(t) => t,
                        Err(e) => {
                            push_error(&tx, &e);
                            return None;
                        }
                    };
                    let cb = make_cb();
                    Some(instrumented("stream-spec", || {
                        base_mlx_core::speculative::generate(
                            target,
                            draft,
                            &tokens,
                            &opts.params,
                            opts.max_tokens,
                            cb,
                        )
                    }))
                })
            }
            None => with_loaded(&state, &model_id, |loaded| {
                let prompt = loaded.render_chat(&messages, tools_for_task.as_deref());
                let tokens = match loaded.encode(&prompt) {
                    Ok(t) => t,
                    Err(e) => {
                        push_error(&tx, &e);
                        return None;
                    }
                };
                let cb = make_cb();
                Some(instrumented("stream", || {
                    loaded.generate(&tokens, &opts.params, opts.max_tokens, cb)
                }))
            }),
        };

        match res {
            Ok(Some(Ok(gen))) => {
                // When tools came back, emit them in the final delta so
                // clients see a complete tool_calls array. Otherwise the
                // final delta is empty (content was already streamed).
                let final_delta = if !gen.tool_calls.is_empty() {
                    json!({
                        "tool_calls": serde_json::to_value(&gen.tool_calls).unwrap_or(Value::Null),
                    })
                } else if tools_for_task.as_ref().is_some_and(|t| !t.is_empty()) {
                    // Tools were offered but the model replied with
                    // plain content; deliver it all at once.
                    json!({ "content": gen.text })
                } else {
                    json!({})
                };
                let stop = json!({
                    "id": id2,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_id2,
                    "choices": [{
                        "index": 0,
                        "delta": final_delta,
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
    let _ = tx.send(Ok(
        Event::default().data(json!({ "error": { "message": e.to_string() } }).to_string())
    ));
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
