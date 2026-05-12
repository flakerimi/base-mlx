//! base-mlx-server
//!
//! OpenAI-compatible HTTP API on top of `base-mlx-core`. Owns:
//!   - `/v1/chat/completions` with streaming, tools, json_schema
//!   - `/v1/embeddings`
//!   - `/v1/models`
//!   - `/v1/responses` (newer Responses API; opt-in)
//!   - `/mcp` gateway endpoints
//!
//! Tool calls are extracted per-model-family before responding so callers
//! get a populated `tool_calls[]` instead of an empty array.

pub mod app;
pub mod openai;
pub mod state;

pub use app::{serve, ServerConfig};
