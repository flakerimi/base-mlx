//! Speculative decoding: pair a fast draft model with a slower target.
//!
//! Algorithm (greedy variant — exact, no quality loss):
//!   1. Target prefills the prompt; first token is target's argmax.
//!   2. Draft prefills the prompt and ingests that first token.
//!   3. Per iteration: draft proposes K-1 tokens auto-regressively
//!      (one draft forward per token, cheap because the draft is small).
//!   4. Target runs ONE multi-token forward over [leader, d1..d_{K-1}],
//!      returning logits at all K positions.
//!   5. Compare target's argmax at position i to the i-th draft token.
//!      Accept the longest matching prefix; emit those tokens plus one
//!      "bonus" token from target's first disagreement (or the position
//!      after the last accepted draft if all accept).
//!   6. Rewind both KV caches to the accepted length, then continue.
//!
//! Per-iteration emitted tokens = (n_accepted + 1) where 0 ≤ n_accepted
//! ≤ K-1. Speedup vs single-target depends on (a) acceptance rate and
//! (b) the cost of a K-token verify forward versus K single-token
//! forwards. For Qwen3-4B + Qwen3-0.6B with K=4 and typical prose
//! acceptance ~0.7, we expect ~1.6× wall speedup.
//!
//! Currently greedy-only: when `params.temperature > 0` we fall back to
//! the single-target generate path. Temperature-aware spec-dec needs
//! rejection sampling; we'll add it once greedy is stable.

use crate::engine::{GenerationResult, LoadedModel};
use crate::model::qwen3::KvCache;
use crate::sampler::SamplingParams;
use crate::{Error, Result};
use mlx_rs::ops::indexing::{argmax_axis, IndexOp};
use mlx_rs::Array;
use tracing::info;

/// Qwen3 EOS tokens (mirrored from engine.rs).
const EOS: &[u32] = &[151645, 151643];

/// Number of speculative tokens per iteration. Sweet spot for 4B/0.6B
/// is 3-5; verify cost grows with K, acceptance rate caps the gain.
pub const SPEC_K: usize = 4;

/// Verify that draft and target share a vocabulary. Done once at
/// request time so we fail loud rather than emitting nonsense tokens.
pub fn check_compatible(target: &LoadedModel, draft: &LoadedModel) -> Result<()> {
    if target.cfg.vocab_size != draft.cfg.vocab_size {
        return Err(Error::ModelLoad(format!(
            "draft/target vocab mismatch: target={} draft={}",
            target.cfg.vocab_size, draft.cfg.vocab_size
        )));
    }
    Ok(())
}

/// Spec-dec generate. Falls back to single-target for non-greedy.
pub fn generate<F: FnMut(&str, u32)>(
    target: &LoadedModel,
    draft: &LoadedModel,
    prompt_tokens: &[u32],
    params: &SamplingParams,
    max_new_tokens: u32,
    mut on_token: F,
) -> Result<GenerationResult> {
        if params.temperature > 0.0 {
            // Greedy-only for now. Rejection sampling for temp>0 is a
            // separate change.
            return target.generate(prompt_tokens, params, max_new_tokens, on_token);
        }

        let mut target_cache = KvCache::new();
        let mut draft_cache = KvCache::new();
        let prompt_len = prompt_tokens.len();

        // ── Prefill ──
        let target_prefill = target.model.forward_multi(prompt_tokens, &mut target_cache)?;
        let _ = draft.model.forward_multi(prompt_tokens, &mut draft_cache)?;

        // First emitted token: target's argmax at the last prompt position.
        let leader_logits = target_prefill.index(((prompt_len as i32) - 1, ..));
        let mut leader = argmax_u32(&leader_logits)?;

        let mut emitted: Vec<u32> = Vec::with_capacity(max_new_tokens as usize + 1);
        let mut text = String::new();
        let mut finish = "length";

        // Emit leader unless it's already EOS.
        if EOS.contains(&leader) {
            return Ok(GenerationResult {
                text,
                prompt_tokens: prompt_len as u32,
                completion_tokens: 0,
                finish_reason: "stop".into(),
                tool_calls: vec![],
            });
        }
        emitted.push(leader);
        flush_emitted(target, &emitted, &mut text, &mut on_token)?;

        // Prime the draft with the leader so it can propose from the
        // position after it.
        let mut draft_logits = draft.model.forward(&[leader], &mut draft_cache)?;

        let mut total_proposed: u64 = 0;
        let mut total_accepted: u64 = 0;
        let mut iters: u64 = 0;
        // Per-iter timing aggregates. Lets us read off where the verify
        // bottleneck actually lives: if draft_ms dominates we picked the
        // wrong draft; if verify_ms stays high relative to a single-target
        // decode, the K-token forward isn't amortizing and the MLP/attn
        // kernels need a closer look.
        let mut total_draft_us: u64 = 0;
        let mut total_verify_us: u64 = 0;
        let mut total_accept_us: u64 = 0;

        // ── Iterate ──
        'outer: while (emitted.len() as u32) < max_new_tokens {
            // 1. Draft proposes K-1 tokens. d[0] = leader (already in
            //    target_cache will be the first verify token); d[1..K]
            //    are draft-proposed.
            let t_draft = std::time::Instant::now();
            let mut drafts: Vec<u32> = Vec::with_capacity(SPEC_K - 1);
            for _ in 0..(SPEC_K - 1) {
                let d = argmax_u32(&draft_logits)?;
                drafts.push(d);
                // Always advance the draft past the proposed token — that
                // keeps draft_cache.seq_len = prompt_len + emitted_so_far +
                // drafts.len() and makes the post-iter truncate trivial.
                // The trailing draft_logits is unused once we know target's
                // bonus, but the forward is cheap on a sub-1B model.
                draft_logits = draft.model.forward(&[d], &mut draft_cache)?;
                if EOS.contains(&d) {
                    break;
                }
            }
            total_draft_us += t_draft.elapsed().as_micros() as u64;

            // 2. Target verifies. Input = [leader, d1..d_{n-1}] where
            //    n = drafts.len() + 1 ≤ K. Returns logits at n positions.
            let mut verify_input: Vec<u32> = Vec::with_capacity(drafts.len() + 1);
            verify_input.push(leader);
            verify_input.extend(&drafts);
            let t_verify = std::time::Instant::now();
            let target_logits = target.model.forward_multi(&verify_input, &mut target_cache)?;
            // Force the verify forward to fully execute before we time the
            // accept loop. Without an eval here the time would smear into
            // the next .eval() (the batched argmax), making it hard to
            // attribute cost.
            target_logits
                .eval()
                .map_err(|e| Error::Inference(e.to_string()))?;
            total_verify_us += t_verify.elapsed().as_micros() as u64;

            // 3. Compare and accept. target_logits[i] predicts the token
            //    that should come after verify_input[i] — i.e., target's
            //    expected d_{i+1}. We accept draft tokens sequentially.
            let n = drafts.len(); // proposed
            total_proposed += n as u64;
            iters += 1;

            let t_accept = std::time::Instant::now();
            // Batched argmax: one MLX op + one .eval() sync produces all
            // (n + 1) target predictions at once.
            let target_preds = mlx_rs::ops::indexing::argmax_axis(&target_logits, 1, false)
                .map_err(|e| Error::Inference(e.to_string()))?;
            target_preds
                .eval()
                .map_err(|e| Error::Inference(e.to_string()))?;
            let preds_slice = target_preds.as_slice::<u32>();

            let mut n_accepted = 0usize;
            let mut bonus: u32 = 0;
            let mut found_disagreement = false;
            for i in 0..n {
                let target_pred = preds_slice[i];
                if target_pred == drafts[i] {
                    n_accepted += 1;
                } else {
                    bonus = target_pred;
                    found_disagreement = true;
                    break;
                }
            }
            if !found_disagreement {
                // All n drafts accepted; bonus = target's prediction at
                // position n (the slot after the last draft).
                bonus = preds_slice[n];
            }
            total_accepted += n_accepted as u64;
            total_accept_us += t_accept.elapsed().as_micros() as u64;

            // 4. Emit accepted drafts + bonus. EOS terminates without
            //    being emitted (matches engine.rs single-model behavior:
            //    EOS sets finish="stop" and breaks before the token is
            //    pushed to the decode buffer, so the marker text doesn't
            //    leak into the response).
            for &t in &drafts[..n_accepted] {
                if (emitted.len() as u32) >= max_new_tokens {
                    break 'outer;
                }
                if EOS.contains(&t) {
                    finish = "stop";
                    break 'outer;
                }
                emitted.push(t);
            }
            flush_emitted(target, &emitted, &mut text, &mut on_token)?;
            if (emitted.len() as u32) >= max_new_tokens {
                break 'outer;
            }
            if EOS.contains(&bonus) {
                finish = "stop";
                break 'outer;
            }
            emitted.push(bonus);
            flush_emitted(target, &emitted, &mut text, &mut on_token)?;

            // 5. Rewind caches.
            //    target processed (1 + n) tokens this iter (leader + drafts).
            //    We keep (1 + n_accepted) of them: leader + accepted drafts.
            //    The bonus is NOT in target's cache — it gets fed as next
            //    iter's leader (verify input position 0).
            target_cache.truncate(prompt_len + (emitted.len() - 1)); // emitted = prev + (1+n_accepted) drafts + 1 bonus; target should hold prev_emitted + 1 + n_accepted = emitted.len() - 1 (everything except the bonus).
            //    draft processed (n_accepted ... n) tokens (it stopped early
            //    if it hit EOS but typically advanced n-1 steps inside the
            //    propose loop, plus the priming forward for the leader at
            //    start-of-iter put it 1 ahead). Simplest correct accounting:
            //    just truncate draft to the same position as target plus
            //    one (because draft will hold the bonus after we feed it).
            draft_cache.truncate(prompt_len + (emitted.len() - 1));

            // 6. Set up next iteration: bonus becomes the new leader.
            //    Feed bonus to draft so it can propose from after-bonus.
            leader = bonus;
            draft_logits = draft.model.forward(&[leader], &mut draft_cache)?;
        }

        let acceptance = if total_proposed > 0 {
            total_accepted as f64 / total_proposed as f64
        } else {
            0.0
        };
        let mean = |us: u64| if iters > 0 { (us / iters) as f64 / 1000.0 } else { 0.0 };
        info!(
            iters = iters,
            proposed = total_proposed,
            accepted = total_accepted,
            acceptance = format!("{:.2}", acceptance),
            emitted = emitted.len(),
            draft_ms_mean = format!("{:.2}", mean(total_draft_us)),
            verify_ms_mean = format!("{:.2}", mean(total_verify_us)),
            accept_ms_mean = format!("{:.2}", mean(total_accept_us)),
            "spec-dec done",
        );

        // Extract tool calls (same convention as the single-model path).
        let (clean, tool_calls) = crate::chat_template::extract_tool_calls(&text);
        let finish_reason = if !tool_calls.is_empty() {
            "tool_calls".to_string()
        } else {
            finish.to_string()
        };

        target_cache.reset();
        draft_cache.reset();

        Ok(GenerationResult {
            text: clean,
            prompt_tokens: prompt_len as u32,
            completion_tokens: emitted.len() as u32,
            finish_reason,
            tool_calls,
        })
}

/// Pull a single u32 token id from a 1-D logits tensor via argmax.
fn argmax_u32(logits: &Array) -> Result<u32> {
    let am = argmax_axis(logits, 0, false).map_err(|e| Error::Inference(e.to_string()))?;
    am.eval().map_err(|e| Error::Inference(e.to_string()))?;
    Ok(am.as_slice::<u32>()[0])
}

/// Decode `emitted` so far, emit only the newly-completed UTF-8 suffix
/// via `on_token`. Mirrors the streaming convention in engine.rs::generate
/// so output formatting is identical between paths.
fn flush_emitted<F: FnMut(&str, u32)>(
    target: &LoadedModel,
    emitted: &[u32],
    text: &mut String,
    on_token: &mut F,
) -> Result<()> {
    let full = target.tokenizer.decode(emitted, false)?;
    // Hold back a trailing replacement char (mid-codepoint boundary).
    let emit_end = match full.rfind('\u{FFFD}') {
        Some(idx) if idx == full.len() - '\u{FFFD}'.len_utf8() => idx,
        _ => full.len(),
    };
    if emit_end > text.len() {
        let piece = &full[text.len()..emit_end];
        on_token(piece, *emitted.last().unwrap_or(&0));
        text.push_str(piece);
    }
    Ok(())
}
