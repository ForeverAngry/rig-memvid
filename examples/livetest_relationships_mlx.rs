//! Scripted multi-turn live test against an MLX-hosted LFM model.
//!
//! This is the same relationship-memory exercise as
//! `livetest_relationships`, but it talks to an OpenAI-compatible MLX
//! server instead of Ollama. Start the server with:
//!
//! ```sh
//! uvx --from mlx-lm mlx_lm.server \
//!   --model LiquidAI/LFM2.5-1.2B-Thinking-MLX-8bit \
//!   --host 127.0.0.1 --port 8080 \
//!   --temp 0.1 --top-p 0.1 --top-k 50 --max-tokens 512
//! ```
//!
//! Then run:
//!
//! ```sh
//! MEMVID_PRINCIPAL=Alice \
//!   cargo run --example livetest_relationships_mlx --features vec
//! ```

#[cfg(not(feature = "vec"))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("livetest_relationships_mlx requires --features vec")
}

#[cfg(feature = "vec")]
mod app {
    use std::path::PathBuf;

    use anyhow::Result;
    use rig::client::CompletionClient;
    use rig::completion::Prompt;
    use rig::providers::openai;
    use rig::vector_store::VectorStoreIndex;
    use rig::vector_store::request::VectorSearchRequest;
    use rig_memvid::{
        CardSelection, MemoryCardContext, MemoryConfig, MemvidPersistHook, MemvidStore, WritePolicy,
    };

    const DEFAULT_MODEL: &str = "LiquidAI/LFM2.5-1.2B-Thinking-MLX-8bit";

    const TRANSCRIPT: &[&str] = &[
        "My name is Alice and I work at Acme as a staff engineer.",
        "Bob is my manager at Acme. He reports to Carol, the VP.",
        "I really like espresso and I dislike instant coffee.",
        "I live in Berlin but I grew up in Lisbon.",
        "I'm allergic to peanuts.",
    ];

    const PROBES: &[&str] = &[
        "Where does Alice live?",
        "Who is Bob's manager?",
        "What food should I avoid serving Alice?",
        "Tell me Alice's coffee preferences.",
    ];

    #[tokio::main]
    pub async fn main() -> Result<()> {
        let _ = tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("rig_memvid=warn")),
            )
            .try_init();

        let path = PathBuf::from(
            std::env::var("MEMVID_PATH").unwrap_or_else(|_| "livetest_mlx.mv2".to_string()),
        );
        let store = MemvidStore::builder()
            .path(&path)
            .enable_lex()
            .with_default_embedder()?
            .open_or_create()?;
        store.commit()?;

        let base_url = std::env::var("MLX_OPENAI_BASE_URL")
            .unwrap_or_else(|_| "http://127.0.0.1:8080/v1".to_string());
        let model_name = std::env::var("MLX_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
        let api_key = std::env::var("MLX_API_KEY").unwrap_or_else(|_| "mlx-local".to_string());
        let max_tokens = std::env::var("MLX_MAX_TOKENS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(4096);
        println!("MLX OpenAI-compatible @ {base_url}, model = {model_name}");

        let client = openai::CompletionsClient::builder()
            .api_key(api_key)
            .base_url(&base_url)
            .build()?;
        let model = client.completion_model(&model_name);
        let principal = std::env::var("MEMVID_PRINCIPAL").unwrap_or_else(|_| "Alice".to_string());
        println!("Structured-memory principal = {principal}");

        let hook = MemvidPersistHook::new(
            store.clone(),
            MemoryConfig::builder()
                .policy(WritePolicy::Raw)
                .commit_each_turn(true)
                .default_tags(vec!["livetest".into(), "mlx".into()])
                .scope(Some("livetest_mlx".into()))
                .principal(Some(principal.clone()))
                .persist_assistant(false)
                .build(),
        );

        let cards_ctx = MemoryCardContext::new(
            store.clone(),
            CardSelection::ForPrincipal(principal.clone()),
        );

        let agent = rig::agent::AgentBuilder::new(model)
            .preamble(
                "You are a helpful assistant with long-term memory. \
             Use any provided context from previous conversations \
             to answer accurately. Keep replies to one or two sentences.",
            )
            .max_tokens(max_tokens)
            .dynamic_context(4, store.clone())
            .dynamic_context(8, cards_ctx)
            .build();

        println!("\n=== Phase 1: ingest ===");
        for turn in TRANSCRIPT {
            println!("\n>>> user: {turn}");
            let reply = agent.prompt(*turn).with_hook(hook.clone()).await?;
            println!("<<< asst: {}", reply.trim());
        }

        println!("\n=== Phase 2: structure captured by memvid ===");
        println!(
            "frames={}  memory_cards={}  mesh_nodes={}  mesh_edges={}",
            store.frame_count()?,
            store.memory_card_count()?,
            store.mesh_node_count()?,
            store.mesh_edge_count()?,
        );

        let cards = store.all_memory_cards()?;
        println!("\n--- all memory cards ({}) ---", cards.len());
        for c in &cards {
            let pol = c
                .polarity
                .map(|p| format!(" [{}]", p.as_str()))
                .unwrap_or_default();
            println!(
                "  * {} {}/{} = {:?}{} (frame {}, conf {:.2})",
                c.kind.as_str(),
                c.entity,
                c.slot,
                c.value,
                pol,
                c.source_frame_id,
                c.confidence.unwrap_or(1.0),
            );
        }

        println!("\n--- entity mentions ---");
        for name in ["Alice", "Bob", "Carol", "Acme", "Berlin", "Lisbon"] {
            match store.find_entity(name)? {
                Some(node) => println!(
                    "  {} -> kind={:?} canonical={:?}",
                    name, node.kind, node.canonical_name
                ),
                None => println!("  {name} -> (not in mesh)"),
            }
        }

        println!("\n--- /entity {principal} ---");
        for c in store.entity_memories(&principal)? {
            println!("  * {}/{} = {:?}", c.entity, c.slot, c.value);
        }

        println!("\n=== Phase 3: probe questions (memory-only, fresh agent calls) ===");
        for q in PROBES {
            println!("\n>>> user: {q}");
            let req = VectorSearchRequest::builder().query(*q).samples(4).build();
            let card_hits = MemoryCardContext::new(
                store.clone(),
                CardSelection::ForPrincipal(principal.clone()),
            )
            .top_n::<serde_json::Value>(req)
            .await?;
            if !card_hits.is_empty() {
                println!("    [card-context hits]");
                for (score, id, doc) in &card_hits {
                    let text = doc
                        .get("text")
                        .and_then(|v| v.as_str())
                        .unwrap_or("<no text>");
                    println!("      score={score:.2} id={id} :: {text}");
                }
            }

            let reply = agent.prompt(*q).await?;
            println!("<<< asst: {}", reply.trim());
        }

        Ok(())
    }
}

#[cfg(feature = "vec")]
fn main() -> anyhow::Result<()> {
    app::main()
}
