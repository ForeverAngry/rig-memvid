//! Integration tests for [`MemvidDemotionHook`] — verifies that messages
//! evicted from an active conversation window land as queryable frames in
//! the underlying `.mv2` archive with the expected metadata.
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
use rig::memory::DemotionHook;
#[cfg(all(feature = "lex", feature = "compaction"))]
use rig::vector_store::{
    VectorSearchRequest, VectorStoreIndex,
    request::{SearchFilter, VectorSearchRequestBuilder},
};
#[cfg(all(feature = "lex", feature = "compaction"))]
use rig_memvid::{MemoryConfig, MemvidDemotionHook, MemvidFilter};
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
async fn demotion_hook_persists_messages_into_archive() -> Result<()> {
    let dir = tempdir()?;
    let store = lex_store(&dir.path().join("demotion.mv2"))?;
    let hook = MemvidDemotionHook::with_defaults(store.clone());

    hook.on_demote(
        "conv-1",
        vec![
            user("the codename for project foxglove is starlight"),
            assistant("acknowledged: starlight"),
        ],
    )
    .await
    .map_err(|e| anyhow::anyhow!("on_demote failed: {e}"))?;

    let req: VectorSearchRequest<MemvidFilter> =
        VectorSearchRequestBuilder::<MemvidFilter>::default()
            .query("starlight")
            .samples(5)
            .build();
    let hits = store.top_n_ids(req).await?;
    assert!(!hits.is_empty(), "expected demoted text to be searchable");
    Ok(())
}

#[cfg(all(feature = "lex", feature = "compaction"))]
#[tokio::test]
async fn demotion_hook_disabled_policy_is_noop() -> Result<()> {
    let dir = tempdir()?;
    let store = lex_store(&dir.path().join("disabled.mv2"))?;
    let config = MemoryConfig {
        policy: rig_memvid::WritePolicy::Disabled,
        ..MemoryConfig::default()
    };
    let hook = MemvidDemotionHook::new(store.clone(), config);

    hook.on_demote("conv-1", vec![user("nothing should be persisted")])
        .await
        .map_err(|e| anyhow::anyhow!("on_demote failed: {e}"))?;

    let req: VectorSearchRequest<MemvidFilter> =
        VectorSearchRequestBuilder::<MemvidFilter>::default()
            .query("nothing")
            .samples(5)
            .build();
    let hits = store.top_n_ids(req).await?;
    assert!(hits.is_empty(), "disabled policy must not write frames");
    Ok(())
}

#[cfg(all(feature = "lex", feature = "compaction"))]
#[tokio::test]
async fn demotion_hook_empty_batch_succeeds_without_writes() -> Result<()> {
    let dir = tempdir()?;
    let store = lex_store(&dir.path().join("empty.mv2"))?;
    let hook = MemvidDemotionHook::with_defaults(store.clone());

    hook.on_demote("conv-1", vec![])
        .await
        .map_err(|e| anyhow::anyhow!("on_demote failed: {e}"))?;

    let req: VectorSearchRequest<MemvidFilter> =
        VectorSearchRequestBuilder::<MemvidFilter>::default()
            .query("anything")
            .samples(5)
            .build();
    let hits = store.top_n_ids(req).await?;
    assert!(hits.is_empty());
    Ok(())
}

#[cfg(all(feature = "lex", feature = "compaction"))]
#[tokio::test]
async fn demotion_hook_honours_scope_filter() -> Result<()> {
    let dir = tempdir()?;
    let store = lex_store(&dir.path().join("scoped.mv2"))?;
    let config = MemoryConfig {
        scope: Some("conv://abc".to_string()),
        ..MemoryConfig::default()
    };
    let hook = MemvidDemotionHook::new(store.clone(), config);

    hook.on_demote(
        "abc",
        vec![user("the safehouse address is 4242 oak avenue")],
    )
    .await
    .map_err(|e| anyhow::anyhow!("on_demote failed: {e}"))?;

    let scoped: VectorSearchRequest<MemvidFilter> =
        VectorSearchRequestBuilder::<MemvidFilter>::default()
            .query("safehouse")
            .samples(5)
            .filter(MemvidFilter::eq("scope", serde_json::json!("conv://abc")))
            .build();
    let hits = store.top_n_ids(scoped).await?;
    assert!(
        !hits.is_empty(),
        "scoped query should match scope-tagged frame"
    );
    Ok(())
}

#[cfg(all(feature = "lex", feature = "compaction"))]
#[tokio::test]
async fn demotion_hook_is_idempotent_within_process() -> Result<()> {
    let dir = tempdir()?;
    let store = lex_store(&dir.path().join("idempotent.mv2"))?;
    let hook = MemvidDemotionHook::with_defaults(store.clone());

    let batch = vec![
        user("dedup probe message one"),
        assistant("dedup probe response one"),
    ];

    let frames_before = store.frame_count()?;
    hook.on_demote("conv-1", batch.clone())
        .await
        .map_err(|e| anyhow::anyhow!("first on_demote failed: {e}"))?;
    let frames_after_first = store.frame_count()?;

    // Second call with the same input must not add any frames.
    hook.on_demote("conv-1", batch.clone())
        .await
        .map_err(|e| anyhow::anyhow!("second on_demote failed: {e}"))?;
    let frames_after_second = store.frame_count()?;

    assert!(
        frames_after_first > frames_before,
        "first call should append frames: before={frames_before} after={frames_after_first}"
    );
    assert_eq!(
        frames_after_first, frames_after_second,
        "idempotency violated: second call appended frames ({frames_after_first} -> {frames_after_second})"
    );
    Ok(())
}

#[cfg(all(feature = "lex", feature = "compaction"))]
#[tokio::test]
async fn demotion_hook_dedup_survives_via_snapshot() -> Result<()> {
    let dir = tempdir()?;
    let store = lex_store(&dir.path().join("snapshot.mv2"))?;
    let first = MemvidDemotionHook::with_defaults(store.clone());

    first
        .on_demote("conv-1", vec![user("snapshot probe")])
        .await
        .map_err(|e| anyhow::anyhow!("on_demote failed: {e}"))?;
    let frames_after_first = store.frame_count()?;
    let snap = first
        .dedup_snapshot()
        .map_err(|e| anyhow::anyhow!("snapshot failed: {e}"))?;
    assert!(!snap.is_empty(), "expected at least one dedup key");

    // Simulate process restart: fresh hook on the same store, replay snapshot.
    let second = MemvidDemotionHook::with_defaults(store.clone());
    second
        .load_dedup_snapshot(&snap)
        .map_err(|e| anyhow::anyhow!("load snapshot failed: {e}"))?;

    second
        .on_demote("conv-1", vec![user("snapshot probe")])
        .await
        .map_err(|e| anyhow::anyhow!("second on_demote failed: {e}"))?;
    let frames_after_second = store.frame_count()?;

    assert_eq!(
        frames_after_first, frames_after_second,
        "snapshot replay did not prevent duplicate write"
    );
    Ok(())
}

#[cfg(all(feature = "lex", feature = "compaction"))]
#[tokio::test]
async fn demotion_hook_pins_kind_and_dedup_key_metadata() -> Result<()> {
    use memvid_core::SearchRequest;

    let dir = tempdir()?;
    let store = lex_store(&dir.path().join("metadata.mv2"))?;
    let hook = MemvidDemotionHook::with_defaults(store.clone());

    hook.on_demote(
        "conv-meta",
        vec![user("the metadata pin probe carries a unique phrase")],
    )
    .await
    .map_err(|e| anyhow::anyhow!("on_demote failed: {e}"))?;

    // Search raw to inspect extra_metadata directly (MemvidFilter only
    // routes onto memvid's first-class SearchRequest fields, not custom
    // metadata keys).
    let req = SearchRequest {
        query: "metadata pin probe".to_string(),
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
    assert!(!resp.hits.is_empty(), "expected at least one hit");

    let mut found_kind = false;
    let mut found_conv = false;
    let mut found_role = false;
    let mut found_dedup = false;
    for hit in &resp.hits {
        let Some(meta) = hit.metadata.as_ref() else {
            continue;
        };
        use rig_memvid::metadata::{FrameKind, MemvidFrameMetadata};
        if let Ok(metadata) = MemvidFrameMetadata::try_from_map(&meta.extra_metadata) {
            assert_eq!(metadata.schema_version, 1);
            if metadata.kind == FrameKind::DemotedMessage {
                found_kind = true;
            }
            if metadata.conversation_id == "conv-meta" {
                found_conv = true;
            }
            if metadata.chat_role == "user" {
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
    assert!(found_kind, "no hit had kind=demoted_message");
    assert!(found_conv, "no hit had conversation_id=conv-meta");
    assert!(found_role, "no hit had chat_role=user");
    assert!(found_dedup, "no hit had a dedup_key");

    // Snapshot surface must also report the same keys.
    let snap = hook
        .dedup_snapshot()
        .map_err(|e| anyhow::anyhow!("snapshot failed: {e}"))?;
    assert!(!snap.is_empty());
    for hex in &snap {
        assert_eq!(hex.len(), 64);
        assert!(hex.chars().all(|c| c.is_ascii_hexdigit()));
    }
    Ok(())
}

#[cfg(all(feature = "lex", feature = "compaction"))]
#[tokio::test]
async fn demotion_hook_propagates_backend_errors_and_abandons_dedup() -> anyhow::Result<()> {
    use rig_memvid::MemvidDemotionHook;
    use rig_memvid::MemvidStore;
    use std::fs;
    use std::os::unix::fs::PermissionsExt;

    let dir = tempfile::tempdir()?;
    let path = dir.path().join("arch.mv2");

    let store = MemvidStore::builder().path(path.clone()).open_or_create()?;
    let hook = MemvidDemotionHook::with_defaults(store.clone());

    let evictions = vec![Message::User {
        content: OneOrMany::one(UserContent::Text(Text {
            text: "This is doomed to fail".into(),
        })),
    }];

    // Sabotage the backend by making the memvid directory read-only
    let mut perms = fs::metadata(&path)?.permissions();
    perms.set_mode(0o555); // Read-execute only, no write
    fs::set_permissions(&path, perms.clone())?;

    // The write should fail!
    let res = hook.on_demote("conv-1", evictions.clone()).await;
    assert!(
        res.is_err(),
        "Expected demotion to fail due to read-only backend"
    );

    // Now repair the backend
    perms.set_mode(0o755);
    fs::set_permissions(&path, perms)?;

    // Try again with exact same content!
    // If the hook prematurely wrote the dedup key during the failure,
    // it would consider this a silent success (skipping write).
    let res = hook.on_demote("conv-1", evictions.clone()).await;
    assert!(
        res.is_ok(),
        "Demotion should succeed after repairing backend"
    );

    // Wait, let's make sure it actually wrote.
    let snap = hook.dedup_snapshot().expect("snapshot");
    assert_eq!(
        snap.len(),
        1,
        "Should have successfully written and dedup'd one frame"
    );

    Ok(())
}
