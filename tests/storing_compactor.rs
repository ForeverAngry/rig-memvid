//! Integration tests for [`MemvidStoringCompactor`] — verifies that the
//! decorator forwards to the inner compactor and persists the produced
//! summary into the underlying `.mv2` archive.
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
use rig::memory::Compactor;
#[cfg(all(feature = "lex", feature = "compaction"))]
use rig::vector_store::{
    VectorSearchRequest, VectorStoreIndex, request::VectorSearchRequestBuilder,
};
#[cfg(all(feature = "lex", feature = "compaction"))]
use rig_memory::TemplateCompactor;
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

#[cfg(all(feature = "lex", feature = "compaction"))]
#[tokio::test]
async fn storing_compactor_forwards_artifact_and_persists() -> Result<()> {
    let dir = tempdir()?;
    let store = lex_store(&dir.path().join("compaction.mv2"))?;
    let inner = TemplateCompactor::with_header("[earlier]");
    let compactor = MemvidStoringCompactor::with_defaults(store.clone(), inner);

    let evicted = vec![
        user("client confirmed kickoff date is june third"),
        assistant("noted: kickoff june 3"),
    ];

    let artifact = compactor
        .compact("conv-1", &evicted, None)
        .await
        .map_err(|e| anyhow::anyhow!("compact failed: {e}"))?;

    let summary = artifact.as_str();
    assert!(summary.contains("[earlier]"), "header missing: {summary}");
    assert!(
        summary.contains("kickoff"),
        "evicted body missing: {summary}"
    );

    // The same summary should now be searchable through the store.
    let req: VectorSearchRequest<MemvidFilter> =
        VectorSearchRequestBuilder::<MemvidFilter>::default()
            .query("kickoff")
            .samples(5)
            .build();
    let hits = store.top_n_ids(req).await?;
    assert!(!hits.is_empty(), "compaction summary not persisted");
    Ok(())
}

#[cfg(all(feature = "lex", feature = "compaction"))]
#[tokio::test]
async fn storing_compactor_threads_carry_over() -> Result<()> {
    let dir = tempdir()?;
    let store = lex_store(&dir.path().join("rolling.mv2"))?;
    let inner = TemplateCompactor::new();
    let compactor = MemvidStoringCompactor::with_defaults(store.clone(), inner);

    let first = compactor
        .compact("conv-2", &[user("alpha context one")], None)
        .await
        .map_err(|e| anyhow::anyhow!("compact failed: {e}"))?;

    let second = compactor
        .compact("conv-2", &[user("beta context two")], Some(&first))
        .await
        .map_err(|e| anyhow::anyhow!("compact failed: {e}"))?;

    let body = second.as_str();
    assert!(body.contains("alpha"), "carry_over not folded in: {body}");
    assert!(body.contains("beta"), "new evicted not included: {body}");
    Ok(())
}

#[cfg(all(feature = "lex", feature = "compaction"))]
#[tokio::test]
async fn storing_compactor_is_idempotent_within_process() -> Result<()> {
    let dir = tempdir()?;
    let store = lex_store(&dir.path().join("compact-dedup.mv2"))?;
    let inner = TemplateCompactor::new();
    let compactor = MemvidStoringCompactor::with_defaults(store.clone(), inner);

    let evicted = vec![
        user("dedup probe alpha"),
        assistant("dedup probe response alpha"),
    ];

    let frames_before = store.frame_count()?;
    let first = compactor
        .compact("conv-dedup", &evicted, None)
        .await
        .map_err(|e| anyhow::anyhow!("first compact failed: {e}"))?;
    let frames_after_first = store.frame_count()?;
    assert!(
        frames_after_first > frames_before,
        "first compact should append a summary frame"
    );

    // Same inputs → TemplateCompactor is deterministic → same rendered
    // text → same dedup key → no new frame.
    let _second = compactor
        .compact("conv-dedup", &evicted, None)
        .await
        .map_err(|e| anyhow::anyhow!("second compact failed: {e}"))?;
    let frames_after_second = store.frame_count()?;
    assert_eq!(
        frames_after_first, frames_after_second,
        "idempotency violated: second compact appended frames"
    );

    // Inner artifact is still returned unchanged.
    assert!(!first.as_str().is_empty());
    Ok(())
}

#[cfg(all(feature = "lex", feature = "compaction"))]
#[tokio::test]
async fn storing_compactor_pins_kind_and_dedup_key_metadata() -> Result<()> {
    use memvid_core::SearchRequest;

    let dir = tempdir()?;
    let store = lex_store(&dir.path().join("compact-meta.mv2"))?;
    let inner = TemplateCompactor::with_header("[earlier]");
    let compactor = MemvidStoringCompactor::with_defaults(store.clone(), inner);

    let evicted = vec![user("metadata-probe unique-marker-xyz")];
    let _ = compactor
        .compact("conv-meta", &evicted, None)
        .await
        .map_err(|e| anyhow::anyhow!("compact failed: {e}"))?;

    let req = SearchRequest {
        query: "unique-marker-xyz".to_string(),
        top_k: 5,
        snippet_chars: 200,
        uri: None,
        scope: None,
        cursor: None,
        #[cfg(feature = "temporal")]
        temporal: None,
        as_of_frame: None,
        as_of_ts: None,
        no_sketch: false,
        acl_context: None,
        acl_enforcement_mode: Default::default(),
    };
    let resp = store.search(req)?;
    assert!(!resp.hits.is_empty());

    let mut found_kind = false;
    let mut found_role = false;
    let mut found_dedup = false;
    for hit in &resp.hits {
        let Some(meta) = hit.metadata.as_ref() else {
            continue;
        };
        use rig_memvid::metadata::{FrameKind, MemvidFrameMetadata};
        if let Ok(metadata) = MemvidFrameMetadata::try_from_map(&meta.extra_metadata) {
            assert_eq!(metadata.schema_version, 1);
            if metadata.kind == FrameKind::CompactionSummary {
                found_kind = true;
            }
            if metadata.chat_role == "system" {
                found_role = true;
            }
            assert_eq!(
                metadata.dedup_key.len(),
                64,
                "dedup_key should be 64 hex chars"
            );
            assert!(
                metadata.dedup_key.chars().all(|c| c.is_ascii_hexdigit()),
                "dedup_key not hex"
            );
            found_dedup = true;
        }
    }
    assert!(found_kind, "no hit had kind=compaction_summary");
    assert!(found_role, "no hit had chat_role=system");
    assert!(found_dedup, "no hit had a dedup_key");
    Ok(())
}
