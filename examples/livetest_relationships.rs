//! Scripted multi-turn live test of relationship capture.
//!
//! Walks the agent through a fixed transcript that mentions several
//! entities and slots, then dumps:
//!   * frame + card counts
//!   * the Logic-Mesh (entity nodes + edges)
//!   * `MemoryCardContext` hits for follow-up queries that should pull
//!     the right cards through principal-bound selection.
//!
//! Run:
//!   OLLAMA_MODEL=qwen3.5:9b \
//!     cargo run --example livetest_relationships --features vec

#[cfg(not(feature = "vec"))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("livetest_relationships requires --features vec")
}

#[cfg(feature = "vec")]
mod app {

    use std::path::PathBuf;

    use anyhow::Result;
    use rig::client::{CompletionClient, Nothing};
    use rig::completion::Prompt;
    use rig::providers::ollama;
    use rig::vector_store::VectorStoreIndex;
    use rig::vector_store::request::VectorSearchRequest;
    use rig_memvid::{
        CardSelection, MemoryCardContext, MemoryConfig, MemvidPersistHook, MemvidStore, WritePolicy,
    };

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
            std::env::var("MEMVID_PATH").unwrap_or_else(|_| "livetest.mv2".to_string()),
        );
        let store = MemvidStore::builder()
            .path(&path)
            .enable_lex()
            .with_default_embedder()?
            .open_or_create()?;
        store.commit()?;

        let base_url = std::env::var("OLLAMA_API_BASE_URL")
            .unwrap_or_else(|_| "http://localhost:11434".to_string());
        let model_name = std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "qwen3.5:9b".to_string());
        println!("Ollama @ {base_url}, model = {model_name}");

        let client = ollama::Client::builder()
            .api_key(Nothing)
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
                .default_tags(vec!["livetest".into()])
                .scope(Some("livetest".into()))
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
            .dynamic_context(4, store.clone())
            .dynamic_context(8, cards_ctx)
            .build();

        // -- Phase 1: feed the transcript -------------------------------------
        println!("\n=== Phase 1: ingest ===");
        for turn in TRANSCRIPT {
            println!("\n>>> user: {turn}");
            let reply = agent.prompt(*turn).with_hook(hook.clone()).await?;
            println!("<<< asst: {}", reply.trim());
        }

        // -- Phase 2: inspect captured structure ------------------------------
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
                "  • {} {}/{} = {:?}{} (frame {}, conf {:.2})",
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
                    "  {} → kind={:?} canonical={:?}",
                    name, node.kind, node.canonical_name
                ),
                None => println!("  {name} → (not in mesh)"),
            }
        }

        println!("\n--- /entity {principal} ---");
        for c in store.entity_memories(&principal)? {
            println!("  • {}/{} = {:?}", c.entity, c.slot, c.value);
        }

        // -- Phase 3: probe queries through the agent -------------------------
        println!("\n=== Phase 3: probe questions (memory-only, fresh agent calls) ===");
        for q in PROBES {
            println!("\n>>> user: {q}");

            // Show what the cards-context view would surface for the probe.
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
