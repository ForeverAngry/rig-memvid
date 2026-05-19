//! End-to-end integration test for the `compaction` feature.
//!
//! Wires the upstream [`rig_memory::CompactingMemory`] around
//! [`rig_memvid::MemvidStoringCompactor`] and exercises the *real*
//! integration seam: `append` then `load` until the policy demotes
//! enough messages to fire compaction, after which the summary must be
//! retrievable from the underlying `.mv2` store via lex search.
//!
//! This complements [`storing_compactor.rs`] (which calls `Compactor`
//! directly) by proving the upstream wiring works as documented.
#![allow(
    clippy::panic_in_result_fn,
    clippy::panic,
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::indexing_slicing
)]

mod common;

#[cfg(all(feature = "lex", feature = "compaction"))]
use anyhow::Result;
#[cfg(all(feature = "lex", feature = "compaction"))]
use common::lex_store;
#[cfg(all(feature = "lex", feature = "compaction"))]
use rig::OneOrMany;
#[cfg(all(feature = "lex", feature = "compaction"))]
use rig::completion::Message;
#[cfg(all(feature = "lex", feature = "compaction"))]
use rig::completion::message::{AssistantContent, Text, UserContent};
#[cfg(all(feature = "lex", feature = "compaction"))]
use rig::memory::ConversationMemory;
#[cfg(all(feature = "lex", feature = "compaction"))]
use rig::vector_store::{
    VectorSearchRequest, VectorStoreIndex, request::VectorSearchRequestBuilder,
};
#[cfg(all(feature = "lex", feature = "compaction"))]
use rig_memory::{
    CompactingMemory, InMemoryConversationMemory, SlidingWindowMemory, TemplateCompactor,
};
#[cfg(all(feature = "lex", feature = "compaction"))]
use rig_memvid::{MemvidFilter, MemvidStoringCompactor};
#[cfg(all(feature = "lex", feature = "compaction"))]
use tempfile::tempdir;

#[cfg(all(feature = "lex", feature = "compaction"))]
fn user(text: &str) -> Message {
    Message::User {
        content: OneOrMany::one(UserContent::Text(Text {
            text: text.to_string(),
        })),
    }
}

#[cfg(all(feature = "lex", feature = "compaction"))]
fn assistant(text: &str) -> Message {
    Message::Assistant {
        id: None,
        content: OneOrMany::one(AssistantContent::Text(Text {
            text: text.to_string(),
        })),
    }
}

/// Drive `append` → `load` traffic against `CompactingMemory` and verify
/// that once the sliding window evicts older turns, the summary frame
/// produced by the compactor is searchable in the backing `.mv2`.
#[cfg(all(feature = "lex", feature = "compaction"))]
#[tokio::test]
async fn compacting_memory_persists_summaries_through_append_load() -> Result<()> {
    let dir = tempdir()?;
    let store = lex_store(&dir.path().join("e2e-compact.mv2"))?;

    // Tight window: 4 messages → 2 turns. Anything older gets demoted
    // and routed through the compactor on the next `load`.
    let memory = CompactingMemory::new(
        InMemoryConversationMemory::new(),
        SlidingWindowMemory::last_messages(4),
        MemvidStoringCompactor::with_defaults(
            store.clone(),
            TemplateCompactor::with_header("[earlier]"),
        ),
    );

    let conv = "conv-e2e";

    // Turn 1: contains the canary topic we'll later query for.
    memory
        .append(
            conv,
            vec![
                user("we agreed the kickoff date is june third"),
                assistant("kickoff june 3 — confirmed"),
            ],
        )
        .await
        .map_err(|e| anyhow::anyhow!("append turn 1: {e}"))?;

    // Turns 2–4: fill the window so turn 1 falls out and gets demoted.
    for i in 2..=4 {
        memory
            .append(
                conv,
                vec![
                    user(&format!("filler message {i}")),
                    assistant(&format!("filler reply {i}")),
                ],
            )
            .await
            .map_err(|e| anyhow::anyhow!("append turn {i}: {e}"))?;
    }

    // Loading drives the policy + compactor. After this call the
    // summary frame should be in the store.
    let kept = memory
        .load(conv)
        .await
        .map_err(|e| anyhow::anyhow!("load: {e}"))?;
    assert!(
        kept.len() <= 5,
        "kept history should be window + at most one summary, got {}",
        kept.len()
    );

    // The canary topic is no longer in the working window — verify it
    // is recoverable from the `.mv2` via the store interface.
    let req: VectorSearchRequest<MemvidFilter> =
        VectorSearchRequestBuilder::<MemvidFilter>::default()
            .query("kickoff june")
            .samples(5)
            .build();
    let hits = store.top_n_ids(req).await?;
    assert!(
        !hits.is_empty(),
        "compaction summary for demoted turn not retrievable from .mv2"
    );

    Ok(())
}

/// Drive enough traffic to force two compaction passes and confirm the
/// `carry_over` artifact threads through, so the *first* topic is still
/// recoverable after a *second* round of evictions.
#[cfg(all(feature = "lex", feature = "compaction"))]
#[tokio::test]
async fn compacting_memory_threads_carry_over_across_rounds() -> Result<()> {
    let dir = tempdir()?;
    let store = lex_store(&dir.path().join("e2e-carry.mv2"))?;

    let memory = CompactingMemory::new(
        InMemoryConversationMemory::new(),
        SlidingWindowMemory::last_messages(4),
        MemvidStoringCompactor::with_defaults(store.clone(), TemplateCompactor::new()),
    );

    let conv = "conv-carry";

    // Two unique markers, one per round of compaction.
    memory
        .append(
            conv,
            vec![
                user("alpha-marker-zzz: contract id is 7741"),
                assistant("ack alpha-marker-zzz contract 7741"),
            ],
        )
        .await
        .map_err(|e| anyhow::anyhow!("append alpha: {e}"))?;

    // Push alpha out of the window → round 1 compaction on next load.
    for i in 0..2 {
        memory
            .append(
                conv,
                vec![
                    user(&format!("pad-a {i}")),
                    assistant(&format!("ack-a {i}")),
                ],
            )
            .await
            .map_err(|e| anyhow::anyhow!("pad-a {i}: {e}"))?;
    }
    let _ = memory.load(conv).await?;

    memory
        .append(
            conv,
            vec![
                user("beta-marker-yyy: invoice id is 9920"),
                assistant("ack beta-marker-yyy invoice 9920"),
            ],
        )
        .await
        .map_err(|e| anyhow::anyhow!("append beta: {e}"))?;

    // Push beta out → round 2 compaction, carry_over should fold alpha
    // forward into the new summary.
    for i in 0..3 {
        memory
            .append(
                conv,
                vec![
                    user(&format!("pad-b {i}")),
                    assistant(&format!("ack-b {i}")),
                ],
            )
            .await
            .map_err(|e| anyhow::anyhow!("pad-b {i}: {e}"))?;
    }
    let _ = memory.load(conv).await?;

    // Both markers must remain recoverable from `.mv2`.
    for marker in ["alpha-marker-zzz", "beta-marker-yyy"] {
        let req: VectorSearchRequest<MemvidFilter> =
            VectorSearchRequestBuilder::<MemvidFilter>::default()
                .query(marker)
                .samples(5)
                .build();
        let hits = store.top_n_ids(req).await?;
        assert!(
            !hits.is_empty(),
            "{marker} not retrievable after compaction rounds"
        );
    }

    Ok(())
}
