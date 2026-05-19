//! Persistent chatbot memory backed by `rig-memvid`, driven by a local
//! Ollama model. No cloud API keys required.
//!
//! Prereqs:
//!   1. Install Ollama (https://ollama.com) and start the daemon
//!      (`ollama serve` — usually auto-started on macOS).
//!   2. Pull a model, e.g. `ollama pull qwen3.5:9b`.
//!
//! Run with:
//!
//! ```bash
//! # Optional overrides:
//! #   OLLAMA_MODEL=qwen3.5:9b OLLAMA_API_BASE_URL=http://localhost:11434 \
//! cargo run --example chatbot_with_memory_ollama
//! ```
//!
//! The example creates (or reopens) `./chatbot_memory_ollama.mv2`, attaches it
//! as both a recall store (via `dynamic_context`) and a write target (via
//! `MemvidPersistHook`), then drops you into an interactive REPL. Anything
//! you type is sent to the agent; user turns are appended to memory and surface
//! on later turns via `dynamic_context`. Re-running the binary retains
//! whatever previous runs wrote. By default, first-person user turns are bound
//! to the stable principal `User`; override with `MEMVID_PRINCIPAL=Alice`.
//!
//! REPL commands: `/recall <query>` to peek at the lexical retrieval hits
//! for a query without prompting the model, `/quit` (or Ctrl-D) to exit.

use std::io::{BufRead, Write};
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

/// Pretty-print a memvid [`MemoryCard`] for the REPL.
fn print_card(card: &rig_memvid::MemoryCard) {
    let polarity = card
        .polarity
        .map(|p| format!(" [{}]", p.as_str()))
        .unwrap_or_default();
    println!(
        "  • {kind} {entity}/{slot} = {value:?}{polarity} (frame {frame}, conf {conf:.2})",
        kind = card.kind.as_str(),
        entity = card.entity,
        slot = card.slot,
        value = card.value,
        frame = card.source_frame_id,
        conf = card.confidence.unwrap_or(1.0),
    );
}

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

    // Use a separate file per retrieval backend so a lex-only archive is
    // not reopened in vec mode (and vice versa). Override with
    // `MEMVID_PATH=/path/to/file.mv2` to share or relocate.
    let default_filename = if cfg!(feature = "vec") {
        "chatbot_memory_ollama_vec.mv2"
    } else {
        "chatbot_memory_ollama.mv2"
    };
    let path = PathBuf::from(
        std::env::var("MEMVID_PATH").unwrap_or_else(|_| default_filename.to_string()),
    );

    // When built with `--features vec`, attach the bundled BGE-small
    // embedder so retrieval is dense-vector (sentence-similarity) rather
    // than BM25 keyword-AND. Lex stays on as well for hybrid retrieval.
    #[cfg(feature = "vec")]
    let store = MemvidStore::builder()
        .path(&path)
        .enable_lex()
        .with_default_embedder()?
        .open_or_create()?;
    #[cfg(not(feature = "vec"))]
    let store = MemvidStore::builder()
        .path(&path)
        .enable_lex()
        .open_or_create()?;

    let retrieval_mode = if cfg!(feature = "vec") {
        "lex+vec (BM25 + BGE-small)"
    } else {
        "lex only (BM25)"
    };
    println!("Retrieval: {retrieval_mode}");

    // Force a commit so freshly enabled lex/vec manifests are persisted to
    // disk before any search runs. Memvid lazily writes index manifests on
    // commit, and `vec_search` errors with `VecNotEnabled` if the manifest
    // is not present even when `enable_vec()` was called in-memory.
    store.commit()?;

    let base_url = std::env::var("OLLAMA_API_BASE_URL")
        .unwrap_or_else(|_| "http://localhost:11434".to_string());
    let requested = std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "qwen3.5:9b".to_string());
    let model_name = resolve_model(&base_url, &requested).await;

    println!("Using Ollama at {base_url} with model `{model_name}`");

    let client = ollama::Client::builder()
        .api_key(Nothing)
        .base_url(&base_url)
        .build()?;
    let model = client.completion_model(&model_name);
    let principal = std::env::var("MEMVID_PRINCIPAL")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| "User".to_string());
    println!("Binding user turns to principal `{principal}` for structured memory extraction");
    let persist_assistant = std::env::var("MEMVID_PERSIST_ASSISTANT")
        .map(|value| !matches!(value.as_str(), "0" | "false" | "FALSE" | "False"))
        .unwrap_or(false);

    let hook = MemvidPersistHook::new(
        store.clone(),
        MemoryConfig {
            policy: WritePolicy::Raw,
            commit_each_turn: true,
            default_tags: vec!["chatbot".into(), "ollama".into()],
            scope: Some("chatbot_memory_ollama".into()),
            principal: Some(principal.clone()),
            persist_assistant,
            ..MemoryConfig::default()
        },
    );

    let card_selection = rig_memvid::CardSelection::ForPrincipal(principal.clone());
    let cards_ctx = rig_memvid::MemoryCardContext::new(store.clone(), card_selection);

    let agent = rig::agent::AgentBuilder::new(model)
        .preamble(
            "You are a helpful assistant with long-term memory. \
             Use any provided context from previous conversations \
             to answer accurately. Keep replies concise.",
        )
        .dynamic_context(4, store.clone())
        .dynamic_context(8, cards_ctx)
        .build();

    println!(
        "\nInteractive chat. Type a message and press Enter.\n\
                 Commands:\n  \
                     /recall <query>         preview retrieval hits\n  \
                     /stats                  frame + memory-card counts\n  \
                     /entity <name>          list memory cards for an entity\n  \
                     /prefs <name>           list preference cards for an entity\n  \
                     /slot <entity> <slot>   show the current value for a slot\n  \
                     /quit (Ctrl-D)          exit\n\
                 Note: memvid's lex index is BM25 keyword-AND with stopword\n\
                 filtering. Phrase recall queries with content keywords\n\
                 (e.g. `pizza`, not `what food do you like?`).\n\
                 Defaults: MEMVID_PRINCIPAL=User and MEMVID_PERSIST_ASSISTANT=false,\n\
                 so first-person turns become stable user-profile memory without\n\
                 assistant paraphrases polluting recall. Override either env var\n\
                 when you need a different archive policy.\n\
                 Memory file: {}",
        path.display()
    );

    loop {
        print!("\n>>> ");
        std::io::stdout().flush().ok();

        let mut line = String::new();
        // Blocking stdin read on the tokio runtime is fine for an
        // interactive example; we have nothing else to do until the
        // user types something.
        let n = std::io::stdin().lock().read_line(&mut line)?;
        if n == 0 {
            // EOF (Ctrl-D)
            println!();
            break;
        }
        let input = line.trim();
        if input.is_empty() {
            continue;
        }
        if matches!(input, "/quit" | "/exit" | ":q") {
            break;
        }
        if let Some(query) = input.strip_prefix("/recall ") {
            if store.frame_count()? == 0 {
                println!("[retrieval preview — query={query:?}] (archive is empty)");
                continue;
            }
            let req = VectorSearchRequest::builder()
                .query(query)
                .samples(4)
                .build();
            let hits = store.top_n::<serde_json::Value>(req).await?;
            println!("[retrieval preview — query={query:?}]");
            for (score, id, _doc) in &hits {
                println!("  score={score:.3} id={id}");
            }
            continue;
        }
        if input == "/stats" {
            println!(
                "[stats] frames={} memory_cards={}",
                store.frame_count()?,
                store.memory_card_count()?
            );
            continue;
        }
        if let Some(name) = input.strip_prefix("/entity ") {
            let cards = store.entity_memories(name.trim())?;
            if cards.is_empty() {
                println!("[entity {name:?}] no memory cards");
            } else {
                println!("[entity {name:?}] {} card(s):", cards.len());
                for c in &cards {
                    print_card(c);
                }
            }
            continue;
        }
        if let Some(name) = input.strip_prefix("/prefs ") {
            let cards = store.entity_preferences(name.trim())?;
            if cards.is_empty() {
                println!("[prefs {name:?}] no preference cards");
            } else {
                println!("[prefs {name:?}] {} card(s):", cards.len());
                for c in &cards {
                    print_card(c);
                }
            }
            continue;
        }
        if let Some(rest) = input.strip_prefix("/slot ") {
            let mut parts = rest.splitn(2, char::is_whitespace);
            let (Some(entity), Some(slot)) = (parts.next(), parts.next()) else {
                println!("[slot] usage: /slot <entity> <slot>");
                continue;
            };
            let entity = entity.trim();
            let slot = slot.trim();
            match store.current_memory(entity, slot)? {
                Some(card) => {
                    println!("[slot {entity}/{slot}] = {:?}", card.value);
                    print_card(&card);
                }
                None => println!("[slot {entity}/{slot}] no value recorded"),
            }
            continue;
        }

        // Inline retrieval preview so it's obvious whether the index is
        // matching this turn's query. Under the default `lex` build,
        // memvid uses BM25 keyword-AND with stopword filtering, so
        // full-sentence questions ("what food do you like?") often drop
        // to zero hits even when relevant content has been written —
        // build with `--features vec` for sentence-similarity retrieval.
        //
        // Skip the preview when the archive is empty: the vec search
        // path errors on an empty index rather than returning no hits.
        if store.frame_count()? == 0 {
            println!("[retrieval] (archive is empty — skipping preview)");
        } else {
            let preview_req = VectorSearchRequest::builder()
                .query(input)
                .samples(4)
                .build();
            let preview = store.top_n::<serde_json::Value>(preview_req).await?;
            if preview.is_empty() {
                println!("[retrieval] no hits for this query");
            } else {
                println!("[retrieval] {} hit(s):", preview.len());
                for (score, id, _doc) in &preview {
                    println!("  score={score:.3} id={id}");
                }
            }
        }

        let response = agent.prompt(input).with_hook(hook.clone()).await?;
        println!("<<< {response}");
    }

    Ok(())
}
