//! Thin wrapper over HuggingFace `tokenizers` so callers don't need a
//! direct dependency. Future: chat-template rendering (Jinja-lite) per
//! architecture so the server layer doesn't have to reimplement it.

use crate::{Error, Result};
use std::path::Path;

pub struct Tokenizer(tokenizers::Tokenizer);

impl Tokenizer {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(path.as_ref())
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        Ok(Self(inner))
    }

    pub fn encode(&self, text: &str, add_special: bool) -> Result<Vec<u32>> {
        let enc = self
            .0
            .encode(text, add_special)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        Ok(enc.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32], skip_special: bool) -> Result<String> {
        self.0
            .decode(ids, skip_special)
            .map_err(|e| Error::Tokenizer(e.to_string()))
    }
}
