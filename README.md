# base-mlx

Local LLM runtime for Apple Silicon. One binary, Rust, native MLX backend, OpenAI-compatible API.

## Why

Every Mac-native local LLM tool today picks a corner of the problem:

- **Ollama** is great, but it's llama.cpp under the hood — not MLX. On Apple Silicon you leave 20–40% performance on the table for many model shapes.
- **LM Studio** is closed source and not embeddable.
- **mlx_lm.server** is fast but its server layer is thin: no real tool calling, no enforced `json_schema`, no concurrent requests, Python deps.
- **mistral.rs** is fast but candle-based, not MLX.

`base-mlx` is the missing tile: a Rust binary that runs MLX-quantized models on Apple Silicon, with the *whole* OpenAI surface implemented properly, plus the things developers actually need (continuous batching, prefix caching, speculative decoding, MCP gateway, RAG out of the box).

## What it does

- OpenAI-compatible HTTP API (chat completions + streaming + tools + json_schema, embeddings, models)
- MCP-native: drop in MCP servers; they show up to the model as tools
- Concurrent multi-model: chat + code + embed + vision all loaded; LRU paging when RAM gets tight
- Continuous batching scheduler so multiple requests share one model's GPU time
- Prefix cache (same system prompt → free prefill)
- Speculative decoding (small draft model accelerates large target)
- Grammar-constrained sampling for `json_schema` — *enforced*, not requested
- Tool-call extraction per model family (Qwen3, Llama3, Mistral, Gemma3)
- Native macOS menu bar app (Tauri) with chat, model manager, MCP toggle, RAM gauge, Spotlight-style capture

## Status

🚧 Early scaffolding. The workspace builds; the engine forward-pass and HTTP routes are stubs. See [ROADMAP.md](./ROADMAP.md) for the build plan.

## Install

Not yet. v1 target: `brew install base-mlx`.

For now, from source:

```bash
git clone https://github.com/base-go/base-mlx
cd base-mlx
cargo run -p base-mlx-cli -- serve
```

## API

OpenAI-compatible at `http://localhost:11435/v1`. Port 11435 chosen so it can coexist with Ollama on 11434.

```bash
curl http://localhost:11435/v1/models
```

## License

MIT.
