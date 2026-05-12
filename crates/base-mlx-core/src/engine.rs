//! High-level engine: a loaded model + tokenizer + a synchronous token
//! generator. The HTTP server owns one of these behind a Mutex per
//! model_id; concurrent requests serialize through it (until we add
//! continuous batching).

use crate::chat_template::{qwen3_chat, ChatMessage, ToolCall};
use crate::model::{ModelConfig, Qwen3};
use crate::model::qwen3::KvCache;
use crate::pull;
use crate::sampler::SamplingParams;
use crate::tokenizer::Tokenizer;
use crate::{Error, Result};

use mlx_rs::Array;

/// Qwen3 EOS tokens.
const EOS: &[u32] = &[151645 /* <|im_end|> */, 151643 /* <|endoftext|> */];

pub struct LoadedModel {
    pub id: String,
    pub repo: String,
    pub cfg: ModelConfig,
    pub tokenizer: Tokenizer,
    pub model: Qwen3,
}

impl LoadedModel {
    /// Load by registry id (e.g. `qwen3-4b-instruct`) or HF repo string.
    pub fn load(id_or_repo: &str) -> Result<Self> {
        let repo = resolve_repo(id_or_repo);
        let dir = pull::find_local(&repo).ok_or_else(|| {
            Error::ModelLoad(format!(
                "{repo} not found locally — pull it first (or place under ~/.lmstudio/models/…)",
            ))
        })?;
        let cfg = ModelConfig::from_path(dir.join("config.json"))?;
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))?;
        let model = Qwen3::load(&dir, cfg.clone())?;
        Ok(Self {
            id: id_or_repo.to_string(),
            repo,
            cfg,
            tokenizer,
            model,
        })
    }

    /// Render a chat with this model's template. `tools` is the raw
    /// OpenAI tools array (each entry is a `{type:"function",function:{...}}`
    /// object); we hand it straight to the template.
    pub fn render_chat(
        &self,
        msgs: &[ChatMessage],
        tools: Option<&[serde_json::Value]>,
    ) -> String {
        qwen3_chat(msgs, tools)
    }

    /// Encode a rendered prompt to token ids.
    pub fn encode(&self, prompt: &str) -> Result<Vec<u32>> {
        // We've already inserted special tokens via the template, so
        // tell the tokenizer not to add its own BOS/EOS.
        self.tokenizer.encode(prompt, false)
    }

    /// Greedy / sampled generation. Calls `on_token(piece, token_id)`
    /// for each produced token; returns the full text + a finish reason.
    pub fn generate<F: FnMut(&str, u32)>(
        &self,
        prompt_tokens: &[u32],
        params: &SamplingParams,
        max_new_tokens: u32,
        mut on_token: F,
    ) -> Result<GenerationResult> {
        let mut cache = KvCache::new();
        // Prefill.
        let mut logits = self.model.forward(prompt_tokens, &mut cache)?;
        let mut text = String::new();
        // Buffer of *all* generated token ids so far. We decode the
        // whole buffer each step and emit only the newly-completed
        // suffix — this collapses the multi-byte UTF-8 boundary bug
        // (single-token decode of a byte mid-codepoint returns `�`).
        let mut produced_ids: Vec<u32> = Vec::with_capacity(max_new_tokens as usize + 1);
        let mut produced = 0u32;
        let mut finish = "length";

        loop {
            let next = self.pick_next(&logits, params)?;
            produced += 1;
            if EOS.contains(&next) {
                finish = "stop";
                break;
            }
            produced_ids.push(next);
            let full = self.tokenizer.decode(&produced_ids, false)?;
            // Hold back the trailing fragment if it ends mid-codepoint —
            // tokenizers render incomplete bytes as `\u{FFFD}`. We slice
            // off any trailing replacement character so it can complete
            // on the next step.
            let emit_end = match full.rfind('\u{FFFD}') {
                Some(idx) if idx == full.len() - '\u{FFFD}'.len_utf8() => idx,
                _ => full.len(),
            };
            if emit_end > text.len() {
                let piece = &full[text.len()..emit_end];
                on_token(piece, next);
                text.push_str(piece);
            }
            if produced >= max_new_tokens {
                break;
            }
            logits = self.model.forward(&[next], &mut cache)?;
        }

        // Drain any final pending bytes (e.g. EOS landed mid-codepoint).
        if !produced_ids.is_empty() {
            let full = self.tokenizer.decode(&produced_ids, false)?;
            if full.len() > text.len() {
                let tail = &full[text.len()..];
                if !tail.contains('\u{FFFD}') {
                    on_token(tail, 0);
                    text.push_str(tail);
                }
            }
        }

        // Extract tool calls from the final text. If any are present we
        // bubble them up via `finish_reason: "tool_calls"` per the
        // OpenAI convention, and strip the markup from `text`.
        let (clean, tool_calls) = crate::chat_template::extract_tool_calls(&text);
        let finish_reason = if !tool_calls.is_empty() {
            "tool_calls".to_string()
        } else {
            finish.to_string()
        };

        // Release the KV cache (drops per-layer growing buffers held
        // by this request). The MLX Metal free-list is bounded by
        // `set_cache_limit` at server startup, so we *don't* clear it
        // here — clearing wipes the compile cache too, forcing kernel
        // recompilation on every subsequent request.
        cache.reset();
        drop(logits);

        Ok(GenerationResult {
            text: clean,
            prompt_tokens: prompt_tokens.len() as u32,
            completion_tokens: produced,
            finish_reason,
            tool_calls,
        })
    }

    fn pick_next(&self, logits: &Array, params: &SamplingParams) -> Result<u32> {
        // Greedy when temperature == 0 — fast, deterministic, no allocation.
        if params.temperature <= 0.0 {
            let argmax = mlx_rs::ops::indexing::argmax(logits, false)
                .map_err(|e| Error::Inference(e.to_string()))?;
            argmax.eval().map_err(|e| Error::Inference(e.to_string()))?;
            return Ok(argmax.as_slice::<u32>()[0]);
        }
        // Temperature + top-p sampling. Done on CPU after pulling logits
        // out of mlx — vocab=151k vectors are small enough that this is
        // not a bottleneck; we'll fuse it into MLX later.
        let logits_f32 = logits
            .as_dtype(mlx_rs::Dtype::Float32)
            .map_err(|e| Error::Inference(e.to_string()))?;
        logits_f32.eval().map_err(|e| Error::Inference(e.to_string()))?;
        let mut probs: Vec<f32> = logits_f32.as_slice::<f32>().to_vec();
        let t = params.temperature.max(1e-5);
        for p in probs.iter_mut() {
            *p /= t;
        }
        // softmax
        let max = probs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0_f32;
        for p in probs.iter_mut() {
            *p = (*p - max).exp();
            sum += *p;
        }
        for p in probs.iter_mut() {
            *p /= sum;
        }
        // Top-p truncation
        let mut indexed: Vec<(usize, f32)> =
            probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top_p = params.top_p.clamp(0.0, 1.0);
        let mut cum = 0.0_f32;
        let mut cutoff = indexed.len();
        for (i, (_, p)) in indexed.iter().enumerate() {
            cum += p;
            if cum >= top_p {
                cutoff = i + 1;
                break;
            }
        }
        let top: Vec<(usize, f32)> = indexed.into_iter().take(cutoff).collect();
        let renorm: f32 = top.iter().map(|(_, p)| p).sum();
        // Categorical draw
        use rand::Rng;
        let mut rng: rand::rngs::StdRng = match params.seed {
            Some(s) => {
                use rand::SeedableRng;
                rand::rngs::StdRng::seed_from_u64(s)
            }
            None => {
                use rand::SeedableRng;
                rand::rngs::StdRng::from_entropy()
            }
        };
        let r = rng.gen::<f32>() * renorm;
        let mut acc = 0.0_f32;
        for (idx, p) in &top {
            acc += *p;
            if acc >= r {
                return Ok(*idx as u32);
            }
        }
        Ok(top.last().map(|(i, _)| *i as u32).unwrap_or(0))
    }
}

#[derive(Debug, Clone)]
pub struct GenerationResult {
    pub text: String,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub finish_reason: String,
    /// Any tool calls extracted from the assistant output. When this is
    /// non-empty `text` has them stripped.
    pub tool_calls: Vec<ToolCall>,
}

fn resolve_repo(id_or_repo: &str) -> String {
    if id_or_repo.contains('/') {
        return id_or_repo.to_string();
    }
    crate::registry::default_catalog()
        .into_iter()
        .find(|m| m.id == id_or_repo)
        .map(|m| m.hf_repo)
        .unwrap_or_else(|| id_or_repo.to_string())
}
