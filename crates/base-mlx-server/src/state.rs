//! Shared server state. Holds a map of loaded models keyed by id, with
//! no eviction yet — a real LRU lands when we have more than two slots
//! to juggle. The motivating second slot is speculative decoding, where
//! a draft model lives alongside the target.
//!
//! We can't hand a `LoadedModel` around behind an `Arc` because mlx-rs
//! `Array` is `Send` but not `Sync` — wrapping in `Arc` would prevent
//! `AppState` itself from being `Sync` (axum requires it). So the map
//! lives directly inside a `Mutex`, and the only safe access pattern is
//! to lock + use + release inside `spawn_blocking`.

use base_mlx_core::engine::LoadedModel;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub type AppState = Arc<AppInner>;

pub struct AppInner {
    pub models: Mutex<HashMap<String, LoadedModel>>,
}

impl AppInner {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            models: Mutex::new(HashMap::new()),
        })
    }
}
