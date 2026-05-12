//! Shared server state. v1 holds at most one loaded model (the one
//! actively serving requests); future versions will keep a small LRU
//! set and page models in/out of RAM.
//!
//! We can't hand a `LoadedModel` around behind an `Arc` because mlx-rs
//! `Array` is `Send` but not `Sync` — wrapping in `Arc` would prevent
//! `AppState` itself from being `Sync` (axum requires it). So the model
//! lives directly inside a `Mutex`, and the only safe access pattern is
//! to lock + use + release inside `spawn_blocking`.

use base_mlx_core::engine::LoadedModel;
use std::sync::{Arc, Mutex};

pub type AppState = Arc<AppInner>;

pub struct AppInner {
    pub model: Mutex<Option<(String, LoadedModel)>>,
}

impl AppInner {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            model: Mutex::new(None),
        })
    }
}
