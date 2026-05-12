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
            tracing::warn!("pull not implemented yet for {model}");
            Ok(())
        }
    }
}
