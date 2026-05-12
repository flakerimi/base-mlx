//! Sampling stack.
//!
//! Plain (temperature / top-p / top-k / repetition penalty), plus a
//! grammar-constrained mode for strict JSON schema enforcement. The
//! grammar engine is a Rust port of llama.cpp's GBNF that runs at
//! sampling time — schemas are *enforced* during decode, not validated
//! afterwards.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub repetition_penalty: f32,
    pub seed: Option<u64>,
    pub grammar: Option<Grammar>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.95,
            top_k: 64,
            repetition_penalty: 1.0,
            seed: None,
            grammar: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Grammar {
    /// GBNF source. Compiled once per request and run as a finite-state
    /// mask over the vocabulary at each decode step.
    Gbnf { source: String },
    /// JSON Schema. Compiled to GBNF internally.
    JsonSchema { schema: serde_json::Value },
    /// Regex. Compiled to GBNF internally.
    Regex { pattern: String },
}
