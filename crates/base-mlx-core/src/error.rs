use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    #[error("safetensors: {0}")]
    Safetensors(#[from] safetensors::SafeTensorError),

    #[error("serde: {0}")]
    Serde(#[from] serde_json::Error),

    #[error("tokenizer: {0}")]
    Tokenizer(String),

    #[error("model load: {0}")]
    ModelLoad(String),

    #[error("unsupported architecture: {0}")]
    UnsupportedArchitecture(String),

    #[error("invalid config: {0}")]
    InvalidConfig(String),

    #[error("inference: {0}")]
    Inference(String),

    #[error("other: {0}")]
    Other(#[from] anyhow::Error),
}
