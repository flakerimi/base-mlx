//! Chat-message → prompt rendering. Each architecture has its own
//! conventions; for v1 we hand-roll Qwen3's. Later we'll switch to a
//! Jinja-lite engine driven by `tokenizer_config.json#chat_template`.

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    /// Plain text content. (Vision parts will widen this later.)
    #[serde(default)]
    pub content: String,
}

/// Render a Qwen3 (Instruct) chat as a prompt string.
///
/// Format:
///   <|im_start|>system\n{system}<|im_end|>\n
///   <|im_start|>user\n{u1}<|im_end|>\n
///   <|im_start|>assistant\n{a1}<|im_end|>\n
///   <|im_start|>user\n{u2}<|im_end|>\n
///   <|im_start|>assistant\n
pub fn qwen3_chat(messages: &[ChatMessage]) -> String {
    let mut out = String::new();
    let mut saw_system = false;
    for m in messages {
        if m.role == "system" {
            out.push_str("<|im_start|>system\n");
            out.push_str(&m.content);
            out.push_str("<|im_end|>\n");
            saw_system = true;
        }
    }
    if !saw_system {
        // Qwen3 expects something; an empty system is fine but a useful
        // default lets it behave like a chat assistant when callers omit
        // one entirely.
        out.push_str("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n");
    }
    for m in messages {
        match m.role.as_str() {
            "user" => {
                out.push_str("<|im_start|>user\n");
                out.push_str(&m.content);
                out.push_str("<|im_end|>\n");
            }
            "assistant" => {
                out.push_str("<|im_start|>assistant\n");
                out.push_str(&m.content);
                out.push_str("<|im_end|>\n");
            }
            "system" => { /* already emitted at top */ }
            other => {
                // Unknown roles are tolerated — emit as a user turn with
                // a role tag so the model has *some* signal.
                out.push_str("<|im_start|>user\n[");
                out.push_str(other);
                out.push_str("] ");
                out.push_str(&m.content);
                out.push_str("<|im_end|>\n");
            }
        }
    }
    out.push_str("<|im_start|>assistant\n");
    out
}
