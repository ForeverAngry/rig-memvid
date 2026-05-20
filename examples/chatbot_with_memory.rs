//! Persistent chatbot memory backed by `rig-memvid`.
//!
//! Run with:
//!
//! ```bash
//! OPENAI_API_KEY=sk-... cargo run --example chatbot_with_memory
//! ```
//!
//! The example creates (or reopens) `./chatbot_memory.mv2`, attaches it as
//! both a recall store (via `dynamic_context`) and a write target (via
//! `MemvidPersistHook`), then runs a single turn. Re-running the binary will
//! retain whatever the previous run wrote into the archive.

use std::path::PathBuf;

use anyhow::Result;
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;
use rig::providers::openai::{self, GPT_4O_MINI};
use rig_memvid::{MemoryConfig, MemvidPersistHook, MemvidStore, WritePolicy};

#[tokio::main]
async fn main() -> Result<()> {
    // Surface library `tracing` warnings (e.g. failed persist writes from
    // the hook) on stderr. Set `RUST_LOG=rig_memvid=debug` to see more.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("rig_memvid=warn")),
        )
        .try_init();

    let path = PathBuf::from("chatbot_memory.mv2");
    let store = MemvidStore::builder()
        .path(&path)
        .enable_lex()
        .open_or_create()?;

    let client = openai::Client::from_env()?;
    let model = client.completion_model(GPT_4O_MINI);

    let hook = MemvidPersistHook::new(
        store.clone(),
        MemoryConfig::builder()
            .policy(WritePolicy::Raw)
            .commit_each_turn(true)
            .default_tags(vec!["chatbot".into()])
            .scope(Some("chatbot_memory".into()))
            .build(),
    );

    let agent = rig::agent::AgentBuilder::new(model)
        .preamble(
            "You are a helpful assistant with long-term memory. \
             Use any provided context from previous conversations \
             to answer accurately.",
        )
        .dynamic_context(4, store)
        .build();

    let response = agent
        .prompt("My favourite colour is teal. Please remember it.")
        .with_hook(hook)
        .await?;

    println!("{response}");

    Ok(())
}
