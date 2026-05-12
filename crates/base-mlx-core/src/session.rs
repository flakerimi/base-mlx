//! A generation session: prompt + sampling params + streaming output.
//!
//! Sessions are short-lived (one request) but share the underlying
//! per-model KV cache slab via the scheduler. Multi-request batching
//! and prefix caching are scheduler-level concerns.

use crate::sampler::SamplingParams;
use crate::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionRequest {
    pub model_id: String,
    pub prompt_tokens: Vec<u32>,
    pub max_new_tokens: u32,
    pub stop_token_ids: Vec<u32>,
    #[serde(default)]
    pub sampling: SamplingParams,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionChunk {
    pub token_id: u32,
    pub text: String,
    pub done: bool,
    pub finish_reason: Option<String>,
}

/// Trait the engine fulfils for the server crate. Implementations may be
/// in-process (mlx-rs forward pass) or out-of-process (for tests).
#[async_trait::async_trait]
pub trait Engine: Send + Sync {
    async fn stream(
        &self,
        req: SessionRequest,
    ) -> Result<futures::stream::BoxStream<'static, Result<SessionChunk>>>;
}
