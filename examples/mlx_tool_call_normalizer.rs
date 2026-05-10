//! Live smoke test for MLX-hosted LFM tool-call marker normalization.
//!
//! Start the local MLX server with:
//!
//! ```sh
//! uvx --from mlx-lm mlx_lm.server \
//!   --model LiquidAI/LFM2.5-1.2B-Thinking-MLX-8bit \
//!   --host 127.0.0.1 --port 8080 \
//!   --temp 0.1 --top-p 0.1 --top-k 50 --max-tokens 2048
//! ```
//!
//! Then run:
//!
//! ```sh
//! cargo run --example mlx_tool_call_normalizer
//! ```

use anyhow::{Context, Result};
use reqwest::Client;
use rig_compose::{
    LfmNormalizer, LocalTool, StructuredToolCallNormalizer, ToolCallNormalizer, ToolRegistry,
    ToolSchema, dispatch_tool_invocations,
};
use serde_json::json;
use std::sync::Arc;

const DEFAULT_BASE_URL: &str = "http://127.0.0.1:8080/v1";
const DEFAULT_MODEL: &str = "LiquidAI/LFM2.5-1.2B-Thinking-MLX-8bit";

#[tokio::main]
async fn main() -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .try_init();

    let base_url = std::env::var("MLX_OPENAI_BASE_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.into());
    let model = std::env::var("MLX_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.into());
    let max_tokens = std::env::var("MLX_MAX_TOKENS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(2048);
    let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));

    let system_prompt = r#"You are a helpful assistant. You have access to the following tool:
- get_weather(city: str) -> str: Get the current weather for a city.

If you need to use a tool, output exactly:
<|tool_call_start|>[tool_name(kwarg=value)]<|tool_call_end|>
Do not answer in prose before the tool call."#;

    let request_body = json!({
        "model": model,
        "messages": [
            { "role": "system", "content": system_prompt },
            { "role": "user", "content": "What is the weather like in Berlin today?" }
        ],
        "temperature": 0.1,
        "max_tokens": max_tokens,
    });

    println!("MLX OpenAI-compatible endpoint = {url}");
    println!("model = {model}");

    let response = Client::new()
        .post(&url)
        .json(&request_body)
        .send()
        .await
        .with_context(|| format!("failed to call MLX server at {url}"))?;

    let status = response.status();
    let body = response
        .text()
        .await
        .context("failed to read MLX response")?;
    if !status.is_success() {
        anyhow::bail!("MLX server returned {status}: {body}");
    }

    let json_response: serde_json::Value =
        serde_json::from_str(&body).context("failed to parse MLX response JSON")?;
    let content = json_response
        .get("choices")
        .and_then(|choices| choices.as_array())
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("message"))
        .and_then(|message| message.get("content"))
        .and_then(|content| content.as_str())
        .context("MLX response did not contain choices[0].message.content")?;

    println!("\n=== Raw Model Output ===");
    println!("{}", content.trim());

    let normalizer = LfmNormalizer;
    let mut invocations = StructuredToolCallNormalizer::normalize(&json_response)
        .context("failed to normalize structured provider tool calls")?;
    if invocations.is_empty() && normalizer.is_applicable(content) {
        invocations = normalizer
            .normalize(content)
            .context("failed to normalize LFM tool-call markers")?;
    }
    if invocations.is_empty() {
        anyhow::bail!(
            "model output did not contain structured tool calls or LFM tool-call markers"
        );
    }

    println!("\n=== Structured Invocations ===");
    for invocation in &invocations {
        println!("tool = {}", invocation.name);
        println!("args = {}", invocation.args);
    }

    let tools = ToolRegistry::new();
    tools.register(Arc::new(LocalTool::new(
        ToolSchema {
            name: "get_weather".into(),
            description: "Get the current weather for a city.".into(),
            args_schema: json!({"type": "object"}),
            result_schema: json!({"type": "object"}),
        },
        |args| async move {
            let city = args
                .get("city")
                .and_then(|value| value.as_str())
                .unwrap_or("unknown");
            Ok(json!({
                "city": city,
                "forecast": "clear and cool",
                "source": "local demo tool"
            }))
        },
    )));

    let dispatch_results = dispatch_tool_invocations(&tools, &invocations)
        .await
        .context("failed to dispatch normalized tool invocations")?;

    println!("\n=== Tool Results ===");
    for result in dispatch_results {
        println!("tool = {}", result.invocation.name);
        println!("output = {}", result.output);
    }

    Ok(())
}
