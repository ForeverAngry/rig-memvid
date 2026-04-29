//! Persistent chatbot memory backed by `rig-memvid`, driven by a local
//! Ollama model. No cloud API keys required.
//!
//! Prereqs:
//!   1. Install Ollama (https://ollama.com) and start the daemon
//!      (`ollama serve` — usually auto-started on macOS).
//!   2. Pull a model, e.g. `ollama pull qwen2.5-coder:3b-instruct`.
//!
//! Run with:
//!
//! ```bash
//! # Optional overrides:
//! #   OLLAMA_MODEL=llama3.2 OLLAMA_API_BASE_URL=http://localhost:11434 \
//! cargo run --example chatbot_with_memory_ollama
//! ```
//!
//! The example creates (or reopens) `./chatbot_memory_ollama.mv2`, attaches it
//! as both a recall store (via `dynamic_context`) and a write target (via
//! `MemvidPersistHook`), then runs two turns so you can observe recall.
//! Re-running the binary retains whatever previous runs wrote.

use std::path::PathBuf;

use anyhow::Result;
use rig::client::{CompletionClient, Nothing};
use rig::completion::Prompt;
use rig::providers::ollama;
use rig::vector_store::VectorStoreIndex;
use rig::vector_store::request::VectorSearchRequest;
use rig_memvid::{MemoryConfig, MemvidPersistHook, MemvidStore, WritePolicy};

/// Best-effort listing of models served by the Ollama daemon at `base_url`.
/// Returns `None` if the daemon is unreachable.
async fn served_models(base_url: &str) -> Option<Vec<String>> {
    #[derive(serde::Deserialize)]
    struct Tag {
        name: String,
    }
    #[derive(serde::Deserialize)]
    struct Tags {
        models: Vec<Tag>,
    }
    let url = format!("{}/api/tags", base_url.trim_end_matches('/'));
    let resp = reqwest::Client::new().get(&url).send().await.ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let tags: Tags = resp.json().await.ok()?;
    Some(tags.models.into_iter().map(|t| t.name).collect())
}

/// Resolve the model to use against the daemon at `base_url`:
///   1. honour `requested` if the daemon serves it
///   2. otherwise fall back to the first served model
///   3. otherwise return `requested` and let the provider error out
async fn resolve_model(base_url: &str, requested: &str) -> String {
    let Some(installed) = served_models(base_url).await else {
        eprintln!(
            "[warn] could not reach Ollama at {base_url}. \
             Is `ollama serve` running?"
        );
        return requested.to_string();
    };
    if installed.iter().any(|m| m == requested) {
        return requested.to_string();
    }
    if let Some(first) = installed.first() {
        eprintln!(
            "[warn] requested model `{requested}` is not served by {base_url}; \
             falling back to `{first}`"
        );
        eprintln!(
            "       installed: {}\n       run `ollama pull {requested}` to use the requested one.",
            installed.join(", ")
        );
        return first.clone();
    }
    eprintln!(
        "[warn] no models are served by {base_url}. \
         Run e.g. `ollama pull {requested}` first."
    );
    requested.to_string()
}

#[tokio::main]
async fn main() -> Result<()> {
    let path = PathBuf::from("chatbot_memory_ollama.mv2");
    let store = MemvidStore::builder()
        .path(&path)
        .enable_lex()
        .open_or_create()?;

    let base_url = std::env::var("OLLAMA_API_BASE_URL")
        .unwrap_or_else(|_| "http://localhost:11434".to_string());
    let requested = std::env::var("OLLAMA_MODEL")
        .unwrap_or_else(|_| "qwen2.5-coder:3b-instruct".to_string());
    let model_name = resolve_model(&base_url, &requested).await;

    println!("Using Ollama at {base_url} with model `{model_name}`");

    let client = ollama::Client::builder()
        .api_key(Nothing)
        .base_url(&base_url)
        .build()?;
    let model = client.completion_model(&model_name);

    let hook = MemvidPersistHook::new(
        store.clone(),
        MemoryConfig {
            policy: WritePolicy::Raw,
            commit_each_turn: true,
            default_tags: vec!["chatbot".into(), "ollama".into()],
            scope: Some("chatbot_memory_ollama".into()),
        },
    );

    let agent = rig::agent::AgentBuilder::new(model)
        .preamble(
            "You are a helpful assistant with long-term memory. \
             Use any provided context from previous conversations \
             to answer accurately. Keep replies concise.",
        )
        .dynamic_context(4, store.clone())
        .build();

    // Turn 1: write a fact into memory.
    let prompt1 = "My favourite colour is teal. Please remember it.";
    println!("\n>>> {prompt1}");
    let response1 = agent.prompt(prompt1).with_hook(hook.clone()).await?;
    println!("<<< {response1}");

    // Turn 2: ask the agent to recall it. The retrieval pulls turn 1
    // (now committed to the memvid file) into the prompt context.
    //
    // Memvid's BM25 lex tokenizer is keyword-AND oriented; full-sentence
    // questions like "What is my favourite colour?" can drop to zero hits
    // because of stopword filtering and out-of-vocabulary terms. Phrase
    // the query around content keywords actually present in the stored
    // frames so the lex index can match.
    let prompt2 = "favourite colour";

    // Peek at what the lexical store would surface for this query so we can
    // see the retrieval path independently of the model.
    let req = VectorSearchRequest::builder()
        .query(prompt2)
        .samples(4)
        .build();
    let hits = store.top_n::<serde_json::Value>(req).await?;
    println!("\n[retrieval preview for turn 2 — query={prompt2:?}]");
    for (score, id, _doc) in &hits {
        println!("  score={score:.3} id={id}");
    }

    println!("\n>>> {prompt2}");
    let response2 = agent.prompt(prompt2).with_hook(hook).await?;
    println!("<<< {response2}");

    Ok(())
}
