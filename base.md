# base-mlx status

A status snapshot. Goal: open-source local LLM runtime on Apple Silicon that beats LM Studio. Hardware reference: M2-class, 32 GB unified memory. Reference model: `mlx-community/Qwen3-4B-Instruct-2507-4bit` (4-bit MLX quant, group_size=64).

---

## What we want

| target | source |
| --- | --- |
| Beat LM Studio's tok/s on the same weights, same hardware | LM Studio reference: ~35 tok/s sustained on Qwen3-4B-Instruct-2507-4bit |
| OpenAI-compatible HTTP API (`/v1/chat/completions`, `/v1/models`) | already done |
| Streaming SSE, tool calls, JSON mode | already done |
| Multi-model on disk; reuse LM Studio's `~/.lmstudio/models/` | already done |
| Open source. Viral-grade UX. Must shine | not yet, but the engine is the moat |

LM Studio reference numbers we're trying to pass:

- short prompt, 300-token output: 35 tok/s
- 1000-word story (~1500 tok output): ~35 tok/s steady state
- TTFT: ~1.16s

## Architecture (where the bits live)

```
crates/
  base-mlx-core/      forward pass, KV cache, sampler, tokenizer, speculative engine
    src/
      model/qwen3.rs        Qwen3 architecture, KvCache, forward + forward_multi
      model/kernels.rs      qkv_block / mlp_block — fused for mlx::compile
      engine.rs             LoadedModel; single-target greedy/sampled decode
      speculative.rs        SpeculativeEngine (target + draft, greedy verify loop)
      sampler.rs            temperature/top-p sampling (CPU-side currently)
      memory.rs             mlx-sys wrappers: cache_limit, active/cache bytes
      mlx_ext.rs            *our* mlx-rs extensions — direct mlx-sys bindings
                            for ops the upstream binding doesn't expose
                            (currently: slice_update)
      pull.rs               HF download + local model discovery
      registry.rs           catalog of curated model ids → HF repos
      tokenizer.rs          HF tokenizers crate wrapper
      chat_template.rs      Qwen3 chat template + tool_call extraction
  base-mlx-server/    axum HTTP server, OpenAI-compatible routes
    src/openai.rs           /v1/chat/completions, /v1/models, instrumentation
    src/state.rs            AppInner with HashMap<id, LoadedModel> (multi-model)
    src/app.rs              startup; MLX cache_limit set here
  base-mlx-cli/       single binary entry point (`base-mlx serve`)
```

## What works

| feature | status | numbers |
| --- | --- | --- |
| Qwen3-4B forward pass, 4-bit quant | ✓ correct | matches LM Studio outputs |
| Tokenizer + chat template | ✓ |  |
| UTF-8-safe streaming (multi-byte boundaries) | ✓ fixed earlier |  |
| KV cache (now: in-place via `mlx_ext::slice_update`) | ✓ | **32.75 tok/s steady single-model on 300-tok prompt** (was 21.3 before in-place) |
| Memory cap (8 GiB free-list, then 16 GiB) | ✓ | run-over-run stable; no leak; `active_mb` flat at 2160 across 5 runs |
| OpenAI chat completions (one-shot + streaming) | ✓ |  |
| Tool calls (function calling, `<tool_call>` markup) | ✓ |  |
| JSON / json_schema response_format | ✓ |  |
| Multi-model state (HashMap loaded) | ✓ | needed for target+draft simultaneous |
| Speculative decoding plumbing (`?spec=<draft_id>`) | ✓ correctness | `acceptance=1.00` on simple prompts; output bit-identical to single-target (greedy) |
| Memory + timing instrumentation per request | ✓ | `gen done tag=... elapsed_ms=... active_mb_*  cache_mb_*` |
| Model warmup on load | ✓ | first request no longer 2× slower |
| `/v1/models` shows on-disk + loaded state | ✓ | uses `find_local_exact` so catalog entries aren't false-positive fuzzy-matched |

## What doesn't work yet (or doesn't pay off)

### Speculative decoding is correct but doesn't accelerate

On the 300-token octopus story prompt:

| path | time | tok/s | notes |
| --- | --- | --- | --- |
| single-model (in-place KV) | 9.16s | **32.75** | new best |
| spec-dec (target=4B, draft=0.6B, K=4) | 12.20s | 24.6 | acceptance=1.00 (perfect) |

**Spec-dec is 25% slower than single-model despite 100% acceptance.**

Why: the K=4 verify forward through the 4B target costs roughly the same as 4 separate K=1 decodes — i.e., we're not getting batched-amortization across the K positions. Theory predicts K-token verify ≈ 1.5× K=1 cost on Apple Silicon if attention + MLP kernels amortize the batched matmul. We're seeing ~4× instead. So per iter we generate 4 tokens but spend ~4× the per-token cost → wash, or worse once you count draft forwards.

What this tells us:
- The attention kernel reads the KV cache once per layer regardless of K — but the **MLP** isn't amortizing across the batched seq dimension. MLP is the bottleneck per layer (intermediate=10240).
- Possibly the `qkv_block` / `mlp_block` fused kernels (`mlx::compile`) were compiled for seq=1 shape and the seq=4 shape is hitting a slower path. `shapeless=true` is set, but we should confirm cache hit empirically.
- Possibly per-iter Rust-side overhead (mask building, argmax_axis sync) is non-trivial relative to the actual MLX compute.

### Run-to-run variance under load

User reports 16 GiB cap shows one fast run (26.9 tok/s) then collapses (13.3 tok/s, then aborted runs). Hypothesis: spec-dec tests we ran in parallel were contending for the GPU. M-series has one GPU, no preemption; concurrent MLX work fights for SMs and memory bandwidth. Need to isolate the bench.

### Long-context throughput

Even at 32 tok/s short prompt, long generation (1500+ tok) is slower because each step's attention is O(N) over context. In-place KV fixed *allocation* growth; it doesn't fix *compute* growth. That's just the cost of dense attention.

## Why we did each thing (decision log)

- **Removed `clear_cache()` per request**: it wipes the MLX compile cache → forced kernel recompilation every request → 2.8 tok/s catastrophic regression. `set_cache_limit` alone bounds memory without trashing the compile cache.
- **Bumped free-list cap 1 GiB → 8 GiB → 16 GiB**: 1 GiB was too small for the 1500-tok KV churn — buffers spilled, fragmentation accumulated across requests, runs decayed. 8 GiB absorbed the churn; 16 GiB marginal beyond that.
- **`find_local_exact` for catalog entries in `/v1/models`**: fuzzy match was reporting `qwen3-coder-30b` and `nomic-embed-text` as locally available because they substring-matched the on-disk Qwen3-4B directory.
- **Warmup on `LoadedModel::load`**: dummy 2-token prefill + 1-token decode compiles the qkv/mlp kernels so the first real request doesn't pay 2× kernel-compilation cost.
- **`mlx_ext::slice_update`**: mlx-rs 0.25 marks `slice_update_device` as `pub(crate)`, blocking external use. mlx-lm Python uses the same C function (`mlx_slice_update`) to get in-place KV writes via MLX's runtime planner (refcount=1 elides the copy). We bound directly to `mlx-sys::mlx_slice_update` and got the in-place behavior — this is the change that took single-model from 21 to 32 tok/s.
- **Pre-allocated `[kv_heads, 4096, head_dim]` cache buffer**: writes go in-place via `slice_update`; SDPA reads a `[0..seq_len]` view. Buffer is ~600 MB total (per-conversation, all layers), allocated once.
- **`forward_multi` returns logits at all positions**: needed by spec-dec verify (compares target's argmax at K positions vs draft proposals). Single-model still uses `forward` which slices to last position.
- **Non-square causal mask for verify**: when extending an existing cache with a multi-token chunk (spec-dec verify, K=4), the mask is `[seq, offset+seq]` — first `offset` cols unmasked (attend to cached keys), trailing `seq` cols causal-triangular. Built with `tril(ones, k=offset)`.
- **Multi-model state (`HashMap`)**: needed to hold target + draft simultaneously without juggling locks.
- **Greedy-only spec-dec for now**: temperature > 0 falls back to single-target. Rejection sampling is a separate change.

## What's next (ranked by impact)

1. **Make K=4 verify actually cheap.** Two directions:
   - Profile per-iter time split: `draft_ms / verify_ms / accept_count`. If verify_ms stays high, attention or MLP isn't amortizing. (User explicitly asked for this — add it next.)
   - Check the `mlx::compile` cache: confirm `shapeless=true` actually shares the compiled kernel between seq=1 and seq=4. If not, force a shape-specific compile per regime.
2. **Bigger K** (K=6, K=8). Reduces fixed per-iter overhead amortization. Needs measurement once #1 is in.
3. **GPU sampling.** Currently `argmax`/`softmax`/`top-p` ship logits to CPU. For greedy + temperature it's small; matters more if we lean into sampling. Single sync per token doesn't help much, but every saved sync compounds with spec-dec's per-position argmax.
4. **Continuous batching.** Serve multiple chat requests on one GPU. LM Studio doesn't do this. Real moat for multi-user.
5. **Custom Metal kernel** for the per-token Qwen3-shaped attention. Beats MLX's stock SDPA on this specific shape. Months of work, real moat against everyone.
6. **Distilled 0.6B draft.** Fine-tune the draft to better mimic the 4B target. Acceptance 0.7 → 0.9 → another 1.3× on top of whatever spec-dec gives us once #1 fixes the verify cost.

## Reference numbers (current, May 12 2026)

Hardware: M2-class Apple Silicon, 32 GB unified memory.
Model: mlx-community/Qwen3-4B-Instruct-2507-4bit (target), mlx-community/Qwen3-0.6B-4bit (draft).
Prompt: "Write a 200-word story about an octopus." (32 prompt tokens, 300 completion tokens, temperature=0).

| build | path | tok/s | notes |
| --- | --- | --- | --- |
| pre in-place KV | single-model | 21.3 | concat per step |
| pre in-place KV | spec-dec K=4 | 20.9 | wash with single-model |
| in-place KV | single-model | **32.75** | the win |
| in-place KV | spec-dec K=4 | 24.6 | verify cost still dominates; 100% acceptance |
| LM Studio (mlx-lm) | reference | ~35 | what we're chasing |

Gap to LM Studio: **~2 tok/s** (6%). Closing it requires either spec-dec actually paying (needs verify-cost fix) or further per-step optimizations on single-model. We're closer than we've ever been.

## How to run

```bash
# build
cargo build --release -p base-mlx-cli

# serve
RUST_LOG=info ./target/release/base-mlx serve
# listens on 127.0.0.1:11435, OpenAI-compatible

# normal chat
curl -sN -X POST localhost:11435/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"qwen3-4b-instruct","temperature":0,"max_tokens":300,
       "messages":[{"role":"user","content":"hi"}]}'

# speculative decoding (greedy only)
curl -sN -X POST 'localhost:11435/v1/chat/completions?spec=mlx-community/Qwen3-0.6B-4bit' \
  -H 'content-type: application/json' \
  -d '{"model":"qwen3-4b-instruct","temperature":0,"max_tokens":300,
       "messages":[{"role":"user","content":"hi"}]}'
```

Watch `/tmp/base-mlx.log` for `gen done` (per-request) and `spec-dec done` (per spec request: iters / proposed / accepted / acceptance ratio).
