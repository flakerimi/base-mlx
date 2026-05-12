use anyhow::Result;
use base_mlx_server::{serve, ServerConfig};
use clap::{Parser, Subcommand};
use tracing_subscriber::{EnvFilter, FmtSubscriber};

/// base-mlx — local LLM runtime for Apple Silicon.
#[derive(Debug, Parser)]
#[command(name = "base-mlx", version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Debug, Subcommand)]
enum Cmd {
    /// Run the HTTP server (OpenAI-compatible API).
    Serve {
        /// Bind address.
        #[arg(long, default_value = "127.0.0.1:11435")]
        addr: String,
    },
    /// List models in the local catalog.
    Models,
    /// Pull a model's weights from the upstream registry (HuggingFace).
    Pull {
        /// Model id (e.g. `qwen3-4b-instruct`) or HuggingFace repo.
        model: String,
    },
    /// Inspect a pulled model's tensors (names, shapes, dtypes).
    Inspect {
        /// Model id or HuggingFace repo.
        model: String,
        /// How many tensors to print (default: 20).
        #[arg(long, default_value_t = 20)]
        limit: usize,
    },
    /// Benchmark TTFT and decode throughput for the current binary.
    Bench {
        #[arg(long, default_value = "qwen3-4b-instruct")]
        model: String,
        /// Prompt — kept short by default so prefill doesn't dominate.
        #[arg(long, default_value = "Tell me a short story about an apple.")]
        prompt: String,
        #[arg(long, default_value_t = 128)]
        max_tokens: u32,
        /// Number of runs to average. First run is treated as warmup.
        #[arg(long, default_value_t = 3)]
        runs: u32,
    },
    /// Generate text from a prompt (M1 — currently does load + inventory only).
    Generate {
        /// Model id or HuggingFace repo.
        #[arg(long, default_value = "qwen3-4b-instruct")]
        model: String,
        /// Prompt text.
        prompt: String,
        /// Max new tokens (used once decode lands).
        #[arg(long, default_value_t = 64)]
        max_tokens: u32,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,base_mlx=debug"));
    FmtSubscriber::builder().with_env_filter(filter).init();

    let cli = Cli::parse();
    match cli.command {
        Cmd::Serve { addr } => {
            let cfg = ServerConfig {
                addr: addr.parse()?,
            };
            serve(cfg).await
        }
        Cmd::Models => {
            for m in base_mlx_core::registry::default_catalog() {
                println!("{:24}  {:<32}  role={:?}", m.id, m.name, m.role);
            }
            Ok(())
        }
        Cmd::Pull { model } => {
            let repo = resolve_repo(&model);
            tracing::info!(%repo, "pulling");
            let report = base_mlx_core::pull::pull(&repo).await?;
            println!("Pulled {} ({} files)", report.repo, report.files.len());
            println!("  dir: {}", report.dir.display());
            for f in &report.files {
                if let Some(name) = f.file_name() {
                    println!("  - {}", name.to_string_lossy());
                }
            }
            Ok(())
        }
        Cmd::Bench {
            model,
            prompt,
            max_tokens,
            runs,
        } => {
            let loaded = base_mlx_core::engine::LoadedModel::load(&model)?;
            let chat_prompt = loaded.render_chat(
                &[base_mlx_core::chat_template::ChatMessage {
                    role: "user".into(),
                    content: prompt.clone(),
                    tool_call_id: None,
                    name: None,
                    tool_calls: None,
                }],
                None,
            );
            let tokens = loaded.encode(&chat_prompt)?;
            let params = base_mlx_core::sampler::SamplingParams {
                temperature: 0.0, // greedy → deterministic measurement
                top_p: 1.0,
                top_k: 0,
                repetition_penalty: 1.0,
                seed: None,
                grammar: None,
            };

            println!(
                "model={model}  prompt_tokens={}  max_tokens={max_tokens}  runs={runs}",
                tokens.len()
            );
            println!("MLX cache (post-load):  active={:>6.1}MB  cached={:>6.1}MB",
                bytes_to_mb(base_mlx_core::memory::active_bytes()?),
                bytes_to_mb(base_mlx_core::memory::cache_bytes()?),
            );

            let mut ttfts = Vec::new();
            let mut decode_rates = Vec::new();
            for r in 0..runs {
                let label = if r == 0 { "warmup" } else { "run" };
                let ttft_start = std::time::Instant::now();
                let mut first_token_at: Option<std::time::Instant> = None;
                let decode_start = std::cell::RefCell::new(None);
                let counter = std::cell::Cell::new(0u32);

                let res = loaded.generate(&tokens, &params, max_tokens, |_piece, _id| {
                    if first_token_at.is_none() {
                        // Capture the time when the first piece arrives —
                        // that's TTFT, since `generate` calls the callback
                        // after each produced token.
                    }
                    if counter.get() == 0 {
                        *decode_start.borrow_mut() = Some(std::time::Instant::now());
                    }
                    counter.set(counter.get() + 1);
                })?;

                // Estimate TTFT and decode rate. We can't observe the first
                // callback time directly here (callback is FnMut not
                // FnOnce-aware), so we approximate: total elapsed minus
                // (n-1)/rate. Better path: instrument generate() itself.
                let total = ttft_start.elapsed().as_secs_f64();
                let n = res.completion_tokens as f64;
                // Rough TTFT = total - decode_time_of_other_tokens
                // We approximate by re-running a tiny generation just to
                // get TTFT alone.
                let ttft_start2 = std::time::Instant::now();
                let _ = loaded.generate(&tokens, &params, 1, |_, _| {})?;
                let ttft = ttft_start2.elapsed().as_secs_f64();
                let decode = (total - ttft).max(0.001);
                let rate = (n - 1.0).max(0.0) / decode;

                ttfts.push(ttft);
                decode_rates.push(rate);
                println!(
                    "  {label:>7} #{r}: ttft={:.3}s  decode={:.1} tok/s  ({} tokens)",
                    ttft, rate, res.completion_tokens
                );
            }

            // Average over runs after warmup.
            let n = (runs as usize).saturating_sub(1).max(1);
            let avg_ttft = ttfts.iter().skip(1).sum::<f64>() / n as f64;
            let avg_rate = decode_rates.iter().skip(1).sum::<f64>() / n as f64;
            println!();
            println!("Average (excluding warmup):  ttft={:.3}s  decode={:.1} tok/s", avg_ttft, avg_rate);
            println!("MLX cache (post-bench): active={:>6.1}MB  cached={:>6.1}MB",
                bytes_to_mb(base_mlx_core::memory::active_bytes()?),
                bytes_to_mb(base_mlx_core::memory::cache_bytes()?),
            );
            Ok(())
        }
        Cmd::Generate {
            model,
            prompt,
            max_tokens,
        } => {
            let repo = resolve_repo(&model);
            let dir = base_mlx_core::pull::find_local(&repo).ok_or_else(|| {
                anyhow::anyhow!(
                    "{repo} not found locally — run `base-mlx pull {repo}` first",
                )
            })?;
            let cfg = base_mlx_core::model::ModelConfig::from_path(dir.join("config.json"))?;
            println!("Found at: {}", dir.display());
            println!("Architecture: {}", cfg.model_type);
            println!(
                "  hidden={} layers={} heads={} kv_heads={} head_dim={} vocab={}",
                cfg.hidden_size,
                cfg.num_hidden_layers,
                cfg.num_attention_heads,
                cfg.kv_heads(),
                cfg.per_head_dim(),
                cfg.vocab_size,
            );
            if let Some(q) = &cfg.quantization {
                println!("  quantization: {}-bit, group_size={}", q.bits, q.group_size);
            }
            println!(
                "  rope_theta={} rms_eps={} tie_embed={}",
                cfg.rope_theta, cfg.rms_norm_eps, cfg.tie_word_embeddings,
            );
            let expected = base_mlx_core::model::Qwen3::expected_tensor_count(&cfg);
            let actual = count_tensors(&dir)?;
            println!("Tensors: expected {} | actual {}", expected, actual);

            // Tokenize the prompt.
            let tok_path = dir.join("tokenizer.json");
            let tok = base_mlx_core::tokenizer::Tokenizer::from_file(&tok_path)?;
            let tokens = tok.encode(&prompt, false)?;
            println!("Prompt tokens: {} ({:?}…)", tokens.len(), &tokens[..tokens.len().min(8)]);

            // Load the model.
            println!("Loading weights…");
            let t0 = std::time::Instant::now();
            let model = base_mlx_core::model::Qwen3::load(&dir, cfg)?;
            println!("  loaded in {:.2}s", t0.elapsed().as_secs_f32());

            // Greedy decode loop. O(n²) without a KV cache — fine for v1
            // verification; KV cache lands in the next milestone.
            use std::io::Write;
            print!("\nGeneration: {}", prompt);
            std::io::stdout().flush().ok();

            let mut cache = base_mlx_core::model::qwen3::KvCache::new();
            // Prefill the prompt in one go.
            let t_prefill = std::time::Instant::now();
            let mut logits = model.forward(&tokens, &mut cache)?;
            logits.eval().ok();
            let prefill_ms = t_prefill.elapsed().as_secs_f32() * 1000.0;

            let t1 = std::time::Instant::now();
            let mut produced = 0u32;
            let mut next: u32;
            loop {
                let argmax = mlx_rs::ops::indexing::argmax(&logits, false)
                    .map_err(|e| anyhow::anyhow!("argmax: {e}"))?;
                argmax.eval().ok();
                next = argmax.as_slice::<u32>()[0];
                produced += 1;
                // EOS: 151645 = <|im_end|>, 151643 = <|endoftext|>.
                if next == 151645 || next == 151643 {
                    break;
                }
                let piece = tok.decode(&[next], false)?;
                print!("{}", piece);
                std::io::stdout().flush().ok();
                if produced >= max_tokens {
                    break;
                }
                // Feed one token at a time; cache holds the rest.
                logits = model.forward(&[next], &mut cache)?;
            }
            let decode_s = t1.elapsed().as_secs_f32();
            let decode_only = (produced.saturating_sub(1)) as f32;
            println!(
                "\n[prefill {} tok in {:.0}ms · decode {} tok in {:.2}s → {:.1} tok/s]",
                tokens.len(),
                prefill_ms,
                produced,
                decode_s,
                decode_only / decode_s.max(0.001),
            );
            Ok(())
        }
        Cmd::Inspect { model, limit } => {
            let repo = resolve_repo(&model);
            let dir = base_mlx_core::pull::find_local(&repo).ok_or_else(|| {
                anyhow::anyhow!(
                    "{repo} not found locally — run `base-mlx pull {repo}` first",
                )
            })?;
            println!("Found at: {}", dir.display());
            inspect_safetensors(&dir, limit)
        }
    }
}

fn resolve_repo(id_or_repo: &str) -> String {
    if id_or_repo.contains('/') {
        return id_or_repo.to_string();
    }
    base_mlx_core::registry::default_catalog()
        .into_iter()
        .find(|m| m.id == id_or_repo)
        .map(|m| m.hf_repo)
        .unwrap_or_else(|| id_or_repo.to_string())
}

fn bytes_to_mb(b: usize) -> f64 {
    b as f64 / 1024.0 / 1024.0
}

fn count_tensors(dir: &std::path::Path) -> anyhow::Result<usize> {
    use safetensors::SafeTensors;
    let mut shards: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
        .collect();
    shards.sort();
    let mut total = 0usize;
    for shard in &shards {
        let bytes = std::fs::read(shard)?;
        let st = SafeTensors::deserialize(&bytes)?;
        total += st.names().len();
    }
    Ok(total)
}

fn inspect_safetensors(dir: &std::path::Path, limit: usize) -> anyhow::Result<()> {
    use safetensors::SafeTensors;
    // Find all .safetensors shards.
    let mut shards: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
        .collect();
    shards.sort();
    if shards.is_empty() {
        anyhow::bail!("no .safetensors files in {}", dir.display());
    }
    let mut total = 0usize;
    let mut printed = 0usize;
    for shard in &shards {
        let bytes = std::fs::read(shard)?;
        let st = SafeTensors::deserialize(&bytes)?;
        let names: Vec<_> = st.names();
        total += names.len();
        for name in names {
            if printed < limit {
                let view = st.tensor(name)?;
                println!(
                    "  {:60}  {:?}  {:?}",
                    name,
                    view.shape(),
                    view.dtype()
                );
                printed += 1;
            }
        }
    }
    println!(
        "({} shards, {} tensors total; showing {})",
        shards.len(),
        total,
        printed
    );
    Ok(())
}
