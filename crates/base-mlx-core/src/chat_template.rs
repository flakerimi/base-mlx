//! Chat-message → prompt rendering. Each architecture has its own
//! conventions; for v1 we hand-roll Qwen3's, matching its shipped
//! `chat_template.jinja` for the common cases (tools, tool results,
//! plain chat).

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    /// Plain text content. (Vision parts will widen this later.)
    #[serde(default)]
    pub content: String,
    /// For `role: "tool"` — id of the originating assistant tool call.
    #[serde(default)]
    pub tool_call_id: Option<String>,
    /// For `role: "tool"` — function name. Optional; Qwen3 only needs
    /// the result content.
    #[serde(default)]
    pub name: Option<String>,
    /// For `role: "assistant"` — replayed tool calls in a prior turn.
    #[serde(default)]
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String, // "function"
    pub function: ToolCallFunction,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolCallFunction {
    pub name: String,
    /// JSON string per OpenAI convention. We keep it as a string so the
    /// client can decide how to parse it.
    pub arguments: String,
}

/// Render a Qwen3 (Instruct) chat as a prompt string.
///
/// If `tools` is non-empty, they are emitted in the system block per
/// Qwen3's template (`# Tools` section with `<tools>...</tools>` JSON
/// array). The assistant is then expected to either reply normally or
/// emit one or more `<tool_call>{...}</tool_call>` blocks.
pub fn qwen3_chat(messages: &[ChatMessage], tools: Option<&[Value]>) -> String {
    let mut out = String::new();

    // System block — combines user-provided system content with the tool
    // protocol instruction when tools are present.
    let system_content = messages
        .iter()
        .find(|m| m.role == "system")
        .map(|m| m.content.as_str())
        .unwrap_or("You are a helpful assistant.");

    let has_tools = tools.map(|t| !t.is_empty()).unwrap_or(false);
    out.push_str("<|im_start|>system\n");
    out.push_str(system_content);
    if has_tools {
        out.push_str("\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>");
        for t in tools.unwrap() {
            out.push('\n');
            out.push_str(&serde_json::to_string(t).unwrap_or_else(|_| "{}".into()));
        }
        out.push_str("\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>");
    }
    out.push_str("<|im_end|>\n");

    // Subsequent turns, in order. Tool messages are bundled into a
    // single user turn wrapped in <tool_response> blocks per Qwen3's
    // expectations.
    let mut i = 0;
    while i < messages.len() {
        let m = &messages[i];
        match m.role.as_str() {
            "system" => {
                // Already emitted (only the first one is used).
            }
            "user" => {
                out.push_str("<|im_start|>user\n");
                out.push_str(&m.content);
                out.push_str("<|im_end|>\n");
            }
            "assistant" => {
                out.push_str("<|im_start|>assistant\n");
                if !m.content.is_empty() {
                    out.push_str(&m.content);
                }
                if let Some(calls) = &m.tool_calls {
                    for c in calls {
                        if !m.content.is_empty() || calls.len() > 1 {
                            out.push('\n');
                        }
                        out.push_str("<tool_call>\n{\"name\": \"");
                        out.push_str(&c.function.name);
                        out.push_str("\", \"arguments\": ");
                        out.push_str(&c.function.arguments);
                        out.push_str("}\n</tool_call>");
                    }
                }
                out.push_str("<|im_end|>\n");
            }
            "tool" => {
                // Collapse a run of tool messages into one user turn.
                out.push_str("<|im_start|>user");
                while i < messages.len() && messages[i].role == "tool" {
                    out.push_str("\n<tool_response>\n");
                    out.push_str(&messages[i].content);
                    out.push_str("\n</tool_response>");
                    i += 1;
                }
                out.push_str("<|im_end|>\n");
                continue; // i already advanced
            }
            other => {
                out.push_str("<|im_start|>user\n[");
                out.push_str(other);
                out.push_str("] ");
                out.push_str(&m.content);
                out.push_str("<|im_end|>\n");
            }
        }
        i += 1;
    }
    out.push_str("<|im_start|>assistant\n");
    out
}

/// Scan generated text for `<tool_call>...</tool_call>` blocks.
/// Returns `(plain_text_with_tool_calls_stripped, tool_calls)`.
///
/// Each tool call payload is expected to be a JSON object with `name`
/// and `arguments` fields (Qwen3's format).
pub fn extract_tool_calls(text: &str) -> (String, Vec<ToolCall>) {
    const OPEN: &str = "<tool_call>";
    const CLOSE: &str = "</tool_call>";
    let mut calls = Vec::new();
    let mut clean = String::new();
    let mut cursor = 0;
    while let Some(rel_open) = text[cursor..].find(OPEN) {
        let abs_open = cursor + rel_open;
        clean.push_str(text[cursor..abs_open].trim_end_matches('\n'));
        let payload_start = abs_open + OPEN.len();
        // Try the well-formed case first: `<tool_call>...</tool_call>`.
        let (payload, advance_to) = if let Some(rel_close) = text[payload_start..].find(CLOSE) {
            let abs_close = payload_start + rel_close;
            (
                text[payload_start..abs_close].trim(),
                abs_close + CLOSE.len(),
            )
        } else {
            // Unterminated — Qwen3 sometimes emits EOS immediately after
            // a closing `}` and skips `</tool_call>`. Take everything to
            // the end and try to parse it.
            (text[payload_start..].trim(), text.len())
        };
        if let Some(call) = parse_tool_call(payload) {
            calls.push(call);
            cursor = advance_to;
        } else {
            // Couldn't parse — leave the markup in content so it's visible.
            clean.push_str(&text[abs_open..advance_to]);
            cursor = advance_to;
        }
    }
    clean.push_str(&text[cursor..]);
    (clean.trim().to_string(), calls)
}

fn parse_tool_call(raw: &str) -> Option<ToolCall> {
    #[derive(Deserialize)]
    struct Raw {
        name: String,
        #[serde(default)]
        arguments: Value,
    }
    let parsed: Raw = serde_json::from_str(raw).ok()?;
    let args = match &parsed.arguments {
        Value::String(s) => s.clone(),
        v => serde_json::to_string(v).unwrap_or_else(|_| "{}".into()),
    };
    Some(ToolCall {
        id: format!("call_{}", &uuid_short()),
        kind: "function".into(),
        function: ToolCallFunction {
            name: parsed.name,
            arguments: args,
        },
    })
}

fn uuid_short() -> String {
    let mut s = uuid::Uuid::new_v4().simple().to_string();
    s.truncate(16);
    s
}
