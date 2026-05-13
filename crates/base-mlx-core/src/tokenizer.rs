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

#[derive(Debug, Default)]
pub struct Utf8TokenDecoder {
    pending: Vec<u32>,
    text: String,
}

impl Utf8TokenDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push<F: FnMut(&str, u32)>(
        &mut self,
        tokenizer: &Tokenizer,
        token_id: u32,
        on_piece: &mut F,
    ) -> Result<()> {
        self.pending.push(token_id);
        let decoded = tokenizer.decode(&self.pending, false)?;
        if decoded.ends_with('\u{FFFD}') {
            return Ok(());
        }
        if !decoded.is_empty() {
            on_piece(&decoded, token_id);
            self.text.push_str(&decoded);
        }
        self.pending.clear();
        Ok(())
    }

    pub fn finish<F: FnMut(&str, u32)>(
        mut self,
        tokenizer: &Tokenizer,
        on_piece: &mut F,
    ) -> Result<String> {
        if !self.pending.is_empty() {
            let decoded = tokenizer.decode(&self.pending, false)?;
            if !decoded.contains('\u{FFFD}') {
                on_piece(&decoded, 0);
                self.text.push_str(&decoded);
            }
            self.pending.clear();
        }
        Ok(self.text)
    }
}
