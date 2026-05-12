# base-mlx

A local LLM runtime for Apple Silicon. Rust, MLX backend, OpenAI-compatible HTTP API. One binary.

```
single-model decode (Qwen3-4B-Instruct-2507-4bit, 300 tokens, greedy)

base-mlx       32.75 tok/s
LM Studio      ~35    tok/s
```

The gap is closing. The point of this project is to close it in the open.

## Why this exists

Local inference on Apple Silicon currently splits into:

| tool             | runtime         | gap                                                       |
| ---------------- | --------------- | --------------------------------------------------------- |
| Ollama           | llama.cpp       | Not MLX. Leaves perf on the table for native shapes.      |
| LM Studio        | mlx-lm wrapper  | Closed source. Not embeddable. No API control.            |
| mlx_lm.server    | mlx-lm (Python) | Thin server. No enforced JSON schema. No concurrent reqs. |
| mistral.rs       | candle          | Not MLX.                                                  |

base-mlx is a Rust binary that drives MLX directly, exposes the OpenAI surface properly (streaming, tools, structured output), and aims to be the engine you can either run as a server or link into a Mac-native app.

## Status

Working today:

- OpenAI-compatible HTTP API at `127.0.0.1:11435/v1`
  - `chat/completions` (one-shot + SSE streaming)
  - `models`
  - `embeddings`
- Qwen3 family forward pass (dense), MLX 4-bit quant (`group_size=64`)
- Tokenizer + chat template + tool-call extraction
- Tool calling end-to-end (Qwen3 `<tool_call>...</tool_call>` markup → OpenAI `tool_calls` array)
- JSON / `json_schema` response format (soft-prompt nudge + post-validate; grammar-constrained sampling is on the roadmap)
- Multi-model state: load several models simultaneously; switch by request `model` field
- HuggingFace pull; reuses LM Studio's `~/.lmstudio/models/` if present (no duplicate downloads)
- Pre-allocated KV cache with in-place writes via a direct `mlx-sys` binding (`mlx_ext::slice_update`)
- Per-request instrumentation: `gen done elapsed_ms=… active_mb_*=… cache_mb_*=…`
- Speculative decoding: opt-in via `?spec=<draft_id>` (greedy only currently). Correct, not yet a win — see [base.md](./base.md) for the verify-cost analysis.

Not yet:

- Continuous batching (single request at a time per model)
- Grammar-constrained sampling (`json_schema` is currently best-effort + validate)
- Prefix / system-prompt cache (TTFT is honest re-prefill every time)
- MCP gateway
- Other model families beyond Qwen3 dense
- macOS menu bar app
- Distribution as `brew install`

See [base.md](./base.md) for a detailed status snapshot (what works, what doesn't, why) and [ROADMAP.md](./ROADMAP.md) for the milestone plan.

## Build

Requires Rust (stable) and a Mac with Apple Silicon. MLX is linked through `mlx-sys`, which builds the C library on first compile (~3 min cold).

```bash
git clone https://github.com/flakerimi/base-mlx
cd base-mlx
cargo build --release -p base-mlx-cli
```

Output binary: `target/release/base-mlx`.

## Run

```bash
RUST_LOG=info ./target/release/base-mlx serve
# base-mlx serving addr=127.0.0.1:11435
```

Port `11435` is chosen so the server can coexist with Ollama on `11434`.

### Pull a model

The first chat request loads on demand. If you want to fetch ahead of time:

```bash
./target/release/base-mlx pull mlx-community/Qwen3-4B-Instruct-2507-4bit
```

If LM Studio is already on the machine, base-mlx will find its existing copies under `~/.lmstudio/models/` and skip the download.

### Chat (one-shot)

```bash
curl -s http://localhost:11435/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "qwen3-4b-instruct",
    "temperature": 0,
    "max_tokens": 300,
    "messages": [{"role": "user", "content": "Explain MLX in two sentences."}]
  }'
```

### Chat (streaming)

```bash
curl -sN http://localhost:11435/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "qwen3-4b-instruct",
    "stream": true,
    "max_tokens": 200,
    "messages": [{"role": "user", "content": "Write a haiku about latency."}]
  }'
```

### Tool calling

```bash
curl -s http://localhost:11435/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "qwen3-4b-instruct",
    "temperature": 0,
    "messages": [{"role": "user", "content": "What is the weather in Tokyo?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "parameters": {
          "type": "object",
          "properties": { "city": {"type": "string"} },
          "required": ["city"]
        }
      }
    }]
  }'
```

The response comes back with `finish_reason: "tool_calls"` and a populated `tool_calls` array.

### Structured output

```bash
curl -s http://localhost:11435/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "qwen3-4b-instruct",
    "temperature": 0,
    "messages": [{"role": "user", "content": "List three primes."}],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "schema": {
          "type": "object",
          "properties": {
            "primes": {"type": "array", "items": {"type": "integer"}}
          },
          "required": ["primes"]
        }
      }
    }
  }'
```

Note: response_format is currently advisory (system-prompt nudge + JSON validation pass). True grammar-constrained sampling is on the roadmap.

### Speculative decoding (experimental)

Opt-in per request. Greedy only (temperature must be 0). The server loads target and draft simultaneously.

```bash
curl -s 'http://localhost:11435/v1/chat/completions?spec=mlx-community/Qwen3-0.6B-4bit' \
  -H 'content-type: application/json' \
  -d '{
    "model": "qwen3-4b-instruct",
    "temperature": 0,
    "max_tokens": 300,
    "messages": [{"role": "user", "content": "Write a 200-word story."}]
  }'
```

Per-iter timing lands in the server log (`spec-dec done draft_ms_mean=… verify_ms_mean=…`). See base.md for current numbers and what's blocking spec-dec from being a win.

### List models

```bash
curl -s http://localhost:11435/v1/models | jq
```

Returns every catalog model that resolves to a local copy plus any LM Studio model on disk, each annotated with `loaded: true|false` for what's currently in memory.

## Architecture

```
crates/
  base-mlx-core/      forward pass, KV cache, sampler, tokenizer, speculative engine
    model/qwen3.rs       Qwen3 architecture + KvCache
    model/kernels.rs     fused qkv / mlp blocks (mlx::compile inputs)
    engine.rs            LoadedModel; greedy / sampled decode loop
    speculative.rs       target+draft verify loop (greedy)
    mlx_ext.rs           direct mlx-sys bindings we needed but mlx-rs doesn't expose
    memory.rs            MLX free-list cap, active/cache byte queries
    pull.rs              HF download + local model discovery (incl. LM Studio cache)
    chat_template.rs     Qwen3 chat template + tool_call markup
  base-mlx-server/    axum HTTP server, OpenAI routes
  base-mlx-cli/       single binary entry point
```

## Performance notes

- **KV cache is in-place.** We bind `mlx_sys::mlx_slice_update` directly because mlx-rs 0.25 marks the equivalent as `pub(crate)`. MLX's runtime planner elides the copy when the input array's refcount is 1, which we guarantee by reassigning the cache slot before the next slice op. This is the same mechanism mlx-lm uses in Python.
- **Metal free-list capped at 16 GiB** at server startup. Without a cap, MLX's per-shape buffer cache grows unbounded and the process eventually fragments / swaps. With the cap, active memory is flat across requests (~2.16 GB for Qwen3-4B-4bit).
- **Warmup on load.** A throwaway 2-token prefill + 1-token decode runs when the model is first loaded so the qkv/mlp kernels compile before the first user request.
- **Per-request log line** lets you debug regressions without external profilers:
  ```
  gen done tag=stream elapsed_ms=9163 active_mb_before=2160 active_mb_after=2160 cache_mb_before=601 cache_mb_after=601
  ```

## Contributing

This is early. The structure is settling; bug reports and perf data are the most useful contributions right now. If you have a Mac that isn't M2-class, dropping a `gen done` log from a known prompt is valuable.

Conventions:

- Run `cargo fmt` and `cargo clippy --release` before sending a PR.
- Commit messages are short, imperative, and lead with a tag (`perf:`, `fix:`, `feat:`, `doc:`). See `git log` for the existing style.
- No co-authored-by lines.

## Acknowledgements

- [MLX](https://github.com/ml-explore/mlx) and the `mlx-community` quants. The engine here is a thin layer over the actual work Apple ML did.
- [mlx-rs](https://github.com/oxidemlx/mlx-rs) for the Rust binding we extend.
- [HuggingFace `tokenizers`](https://github.com/huggingface/tokenizers) for the BPE pipeline.
- [mlx-lm](https://github.com/ml-explore/mlx-lm) — the Python reference we benchmark against and learn from.

## License

MIT.
