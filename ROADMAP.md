# Roadmap

Order matters. Each milestone ends with something demoable.

## M1 — bridge + Qwen3 forward pass

- [ ] `mlx-rs` smoke test: load an MLX-quantized safetensors blob and run a single matmul on Metal.
- [ ] Tokenizer wired up via HF `tokenizers` crate.
- [ ] Qwen3 architecture: token embedding → N transformer blocks (RMSNorm, RoPE, GQA, SwiGLU) → final norm → LM head.
- [ ] KV cache (static-shape buffer, attention masking on growth).
- [ ] One CLI subcommand: `base-mlx generate "Hello"` — non-streaming, single request, single model.

Exit criteria: a Qwen3-4B reply printed to stdout at MLX speed (>30 tok/s on M3).

## M2 — OpenAI server

- [ ] `/v1/chat/completions` non-streaming, single request.
- [ ] Streaming via SSE in the OpenAI `chat.completion.chunk` shape.
- [ ] Chat-template rendering for Qwen3 (Jinja-lite is fine).
- [ ] `/v1/models` from the registry.

Exit criteria: a vanilla OpenAI client (e.g. the `openai` Python SDK with `base_url`) drives a back-and-forth conversation.

## M3 — tools + structured output

- [ ] Qwen3 tool-call template parser (extracts `<tool_call>{...}</tool_call>` into `tool_calls[]`).
- [ ] GBNF grammar engine (port of llama.cpp's, minimal subset) wired into the sampler as a token mask.
- [ ] `response_format: { type: "json_schema" }` compiles schema → GBNF → enforced at decode.

Exit criteria: agent loop with three tool calls in a row works; strict JSON schema is provably enforced (sampler refuses to emit a violating token).

## M4 — embeddings + RAG

- [ ] `/v1/embeddings` against nomic-embed-text.
- [ ] Built-in vector store (sqlite + cosine) — no extra deps.
- [ ] CLI: `base-mlx index ./folder` → indexes; agent gets a `search_files` tool automatically.

## M5 — scheduler + speculative decoding

- [ ] Continuous batching: one model handles N concurrent requests.
- [ ] Prefix cache keyed on `(model, prompt_prefix_hash)`.
- [ ] Speculative decoding with a 0.5B–1.5B draft model.
- [ ] Memory orchestrator: LRU model paging, RAM ceiling enforcement.

Exit criteria: 4 concurrent chat sessions on a 16GB M3 with a code model + chat model + embed model all loaded.

## M6 — vision

- [ ] Qwen3-VL architecture (image encoder + projector + LLM).
- [ ] `image_url` content blocks (base64 + URL fetched server-side).

## M7 — MCP gateway

- [ ] Local MCP server registration (stdio + HTTP).
- [ ] Auto-expose MCP tools to chat completions.
- [ ] Per-server allow/deny toggles.

## M8 — native macOS app

- [ ] Tauri 2 menu bar shell.
- [ ] Chat / Models / MCP / Settings tabs.
- [ ] Global Spotlight-style capture (`⌘⇧L`).
- [ ] Share extension target ("Summarize with local").
- [ ] Shortcuts actions.

## M9 — launch

- [ ] Homebrew tap.
- [ ] Signed + notarized binary.
- [ ] Auto-update.
- [ ] Landing page + 30s demo video.
- [ ] HN / r/LocalLLaMA / X post.

## Non-goals (for v1)

- Training / fine-tuning.
- Linux / Windows ports (post-launch; we have a candle fallback path).
- Custom model architectures via a plugin API.
- Multi-GPU / distributed inference.
