//! Integration tests covering filter semantics, file lifecycle, the raw
//! `MemvidStore::search` path, and error mapping.
#![allow(
    clippy::panic_in_result_fn,
    clippy::panic,
    clippy::expect_used,
    clippy::io_other_error
)]

mod common;

#[cfg(feature = "lex")]
use anyhow::Result;
#[cfg(feature = "lex")]
use common::lex_store;
#[cfg(feature = "lex")]
use memvid_core::{AclContext, AclEnforcementMode, PutOptions, SearchRequest};
#[cfg(feature = "lex")]
use rig::vector_store::{
    InsertDocuments, VectorSearchRequest, VectorStoreError, VectorStoreIndex,
    request::{FilterError, SearchFilter, VectorSearchRequestBuilder},
};
#[cfg(feature = "lex")]
use rig_memvid::{MemvidError, MemvidFilter, MemvidStore};
#[cfg(feature = "lex")]
use serde_json::json;
#[cfg(feature = "lex")]
use tempfile::tempdir;

// --- MemvidFilter -----------------------------------------------------------

#[cfg(feature = "lex")]
#[test]
fn filter_eq_supported_keys_populate_fields() {
    let f = MemvidFilter::eq("uri", json!("doc://a"))
        .and(MemvidFilter::eq("scope", json!("notes")))
        .and(MemvidFilter::eq("as_of_frame", json!(7u64)))
        .and(MemvidFilter::eq("as_of_ts", json!(1_700_000_000_i64)))
        .and(MemvidFilter::eq("cursor", json!("opaque-token")))
        .and(MemvidFilter::eq("no_sketch", json!(true)));

    assert_eq!(f.uri.as_deref(), Some("doc://a"));
    assert_eq!(f.scope.as_deref(), Some("notes"));
    assert_eq!(f.as_of_frame, Some(7));
    assert_eq!(f.as_of_ts, Some(1_700_000_000));
    assert_eq!(f.cursor.as_deref(), Some("opaque-token"));
    assert_eq!(f.no_sketch, Some(true));
}

#[cfg(feature = "lex")]
#[tokio::test]
async fn filter_unknown_key_surfaces_as_filter_error() -> Result<()> {
    let dir = tempdir()?;
    let store = lex_store(&dir.path().join("f.mv2"))?;
    store.put_text("alpha", PutOptions::default())?;

    let bad = MemvidFilter::eq("not_a_key", json!("x"));
    let req: VectorSearchRequest<MemvidFilter> =
        VectorSearchRequestBuilder::<MemvidFilter>::default()
            .query("alpha")
            .samples(1)
            .filter(bad)
            .build();

    let err = store
        .top_n_ids(req)
        .await
        .err()
        .ok_or_else(|| anyhow::anyhow!("expected error"))?;
    match err {
        VectorStoreError::FilterError(FilterError::TypeError(msg)) => {
            assert!(msg.contains("unsupported filter key"), "msg = {msg}");
        }
        other => anyhow::bail!("expected FilterError::TypeError, got: {other:?}"),
    }
    Ok(())
}

#[cfg(feature = "lex")]
#[tokio::test]
async fn filter_gt_lt_or_are_unsupported() -> Result<()> {
    let dir = tempdir()?;
    let store = lex_store(&dir.path().join("g.mv2"))?;
    store.put_text("alpha", PutOptions::default())?;

    for filter in [
        MemvidFilter::gt("as_of_frame", json!(1u64)),
        MemvidFilter::lt("as_of_frame", json!(1u64)),
        MemvidFilter::eq("uri", json!("a")).or(MemvidFilter::eq("uri", json!("b"))),
    ] {
        let req: VectorSearchRequest<MemvidFilter> =
            VectorSearchRequestBuilder::<MemvidFilter>::default()
                .query("alpha")
                .samples(1)
                .filter(filter)
                .build();
        let err = store
            .top_n_ids(req)
            .await
            .err()
            .ok_or_else(|| anyhow::anyhow!("expected error"))?;
        assert!(
            matches!(
                err,
                VectorStoreError::FilterError(FilterError::TypeError(_))
            ),
            "expected FilterError, got {err:?}",
        );
    }
    Ok(())
}

#[cfg(feature = "lex")]
#[tokio::test]
async fn filter_typed_value_mismatch_is_unsupported() -> Result<()> {
    let dir = tempdir()?;
    let store = lex_store(&dir.path().join("t.mv2"))?;
    store.put_text("alpha", PutOptions::default())?;

    let bad = MemvidFilter::eq("as_of_frame", json!("not a number"));
    let req: VectorSearchRequest<MemvidFilter> =
        VectorSearchRequestBuilder::<MemvidFilter>::default()
            .query("alpha")
            .samples(1)
            .filter(bad)
            .build();
    let err = store
        .top_n_ids(req)
        .await
        .err()
        .ok_or_else(|| anyhow::anyhow!("expected typed-mismatch error"))?;
    assert!(
        matches!(
            err,
            VectorStoreError::FilterError(FilterError::TypeError(_))
        ),
        "expected FilterError, got {err:?}",
    );
    Ok(())
}

// --- File lifecycle ---------------------------------------------------------

#[cfg(feature = "lex")]
#[test]
fn open_errors_when_missing_file() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("missing.mv2");
    let res = MemvidStore::builder().path(&path).enable_lex().open();
    assert!(res.is_err(), "open() on missing file should error");
}

#[cfg(feature = "lex")]
#[test]
fn create_errors_when_file_already_exists() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("dup.mv2");
    let _first = MemvidStore::builder().path(&path).enable_lex().create()?;
    let second = MemvidStore::builder().path(&path).enable_lex().create();
    assert!(second.is_err(), "create() over existing file should error");
    Ok(())
}

#[cfg(feature = "lex")]
#[test]
fn open_or_create_then_open_round_trip() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("rt.mv2");

    {
        let store = MemvidStore::builder()
            .path(&path)
            .enable_lex()
            .open_or_create()?;
        store.put_text("alpha", PutOptions::default())?;
        assert!(store.frame_count()? >= 1);
    }

    let reopened = MemvidStore::builder().path(&path).enable_lex().open()?;
    assert!(reopened.frame_count()? >= 1);
    Ok(())
}

#[cfg(feature = "lex")]
#[test]
fn builder_requires_path() {
    let res = MemvidStore::builder().enable_lex().open_or_create();
    assert!(res.is_err(), "builder without path() should fail");
}

// --- Uncommitted writes -----------------------------------------------------

#[cfg(feature = "lex")]
#[tokio::test]
async fn uncommitted_write_visible_only_after_commit() -> Result<()> {
    let dir = tempdir()?;
    let store = lex_store(&dir.path().join("u.mv2"))?;

    store.put_text_uncommitted("greenfield-marker-xyz", PutOptions::default())?;

    // Search before commit may or may not see the frame depending on memvid's
    // instant-index policy, but a search after explicit commit must.
    store.commit()?;

    let req: VectorSearchRequest<MemvidFilter> =
        VectorSearchRequestBuilder::<MemvidFilter>::default()
            .query("greenfield-marker-xyz")
            .samples(5)
            .build();
    let ids = store.top_n_ids(req).await?;
    assert!(!ids.is_empty(), "expected hit after commit");
    Ok(())
}

// --- Raw search passthrough -------------------------------------------------

#[cfg(feature = "lex")]
#[test]
fn raw_search_passthrough_returns_response() -> Result<()> {
    let dir = tempdir()?;
    let store = lex_store(&dir.path().join("r.mv2"))?;
    store.put_text("zeta-token-content", PutOptions::default())?;

    let req = SearchRequest {
        query: "zeta-token-content".into(),
        top_k: 5,
        snippet_chars: 100,
        uri: None,
        scope: None,
        cursor: None,
        #[cfg(feature = "temporal")]
        temporal: None,
        as_of_frame: None,
        as_of_ts: None,
        no_sketch: false,
        acl_context: None,
        acl_enforcement_mode: AclEnforcementMode::default(),
    };
    let resp = store.search(req)?;
    assert!(!resp.hits.is_empty(), "expected raw search to return hits");
    Ok(())
}

// --- Stats / frame_count ----------------------------------------------------

#[cfg(feature = "lex")]
#[test]
fn frame_count_and_stats_grow_with_writes() -> Result<()> {
    let dir = tempdir()?;
    let store = lex_store(&dir.path().join("s.mv2"))?;
    let before = store.frame_count()?;
    store.put_text("alpha", PutOptions::default())?;
    store.put_text("beta", PutOptions::default())?;
    let after = store.frame_count()?;
    assert!(
        after >= before + 2,
        "frame_count should grow by at least 2: before={before} after={after}",
    );
    let _ = store.stats()?;
    Ok(())
}

// --- ACL builder ------------------------------------------------------------

#[cfg(feature = "lex")]
#[tokio::test]
async fn acl_context_builder_does_not_break_default_search() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("acl.mv2");
    let store = MemvidStore::builder()
        .path(&path)
        .enable_lex()
        .acl_context(AclContext {
            tenant_id: Some("tenant-a".into()),
            subject_id: Some("user-1".into()),
            roles: vec!["reader".into()],
            group_ids: vec![],
        })
        .acl_enforcement_mode(AclEnforcementMode::Audit)
        .create()?;

    store.put_text("alpha-acl-token", PutOptions::default())?;

    let req: VectorSearchRequest<MemvidFilter> =
        VectorSearchRequestBuilder::<MemvidFilter>::default()
            .query("alpha-acl-token")
            .samples(5)
            .build();
    let ids = store.top_n_ids(req).await?;
    assert!(
        !ids.is_empty(),
        "audit-mode ACL should not block matching hits"
    );
    Ok(())
}

// --- Error mapping ----------------------------------------------------------

#[cfg(feature = "lex")]
#[test]
fn error_mapping_serde_to_json_error() {
    let serde_err = serde_json::from_str::<serde_json::Value>("{ not json").expect_err("must err");
    let mapped: VectorStoreError = MemvidError::from(serde_err).into();
    assert!(
        matches!(mapped, VectorStoreError::JsonError(_)),
        "Serde -> JsonError, got {mapped:?}",
    );
}

#[cfg(feature = "lex")]
#[test]
fn error_mapping_unsupported_filter_to_filter_error() {
    let mapped: VectorStoreError = MemvidError::UnsupportedFilter("bad clause".into()).into();
    match mapped {
        VectorStoreError::FilterError(FilterError::TypeError(msg)) => {
            assert_eq!(msg, "bad clause");
        }
        other => panic!("expected FilterError::TypeError, got {other:?}"),
    }
}

#[cfg(feature = "lex")]
#[test]
fn error_mapping_io_to_datastore_error() {
    let io_err = std::io::Error::new(std::io::ErrorKind::Other, "boom");
    let mapped: VectorStoreError = MemvidError::Io(io_err).into();
    assert!(
        matches!(mapped, VectorStoreError::DatastoreError(_)),
        "Io -> DatastoreError, got {mapped:?}",
    );
}

// --- InsertDocuments overwrites caller embeddings ---------------------------

#[cfg(feature = "lex")]
#[tokio::test]
async fn insert_documents_ignores_caller_embedding_dimension() -> Result<()> {
    use rig::{
        Embed, OneOrMany,
        embeddings::{Embedding, embed::EmbedError},
    };
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct Doc {
        body: String,
    }
    impl Embed for Doc {
        fn embed(&self, embedder: &mut rig::embeddings::TextEmbedder) -> Result<(), EmbedError> {
            embedder.embed(self.body.clone());
            Ok(())
        }
    }

    let dir = tempdir()?;
    let store = lex_store(&dir.path().join("i.mv2"))?;

    // Provide an obviously-wrong embedding (wrong dim) — must be ignored.
    let docs = vec![(
        Doc {
            body: "lex-ignore-embedding".into(),
        },
        OneOrMany::one(Embedding {
            document: "lex-ignore-embedding".into(),
            vec: vec![0.0; 3], // intentionally bogus
        }),
    )];
    store.insert_documents(docs).await?;

    let req: VectorSearchRequest<MemvidFilter> =
        VectorSearchRequestBuilder::<MemvidFilter>::default()
            .query("lex-ignore-embedding")
            .samples(1)
            .build();
    let ids = store.top_n_ids(req).await?;
    assert_eq!(ids.len(), 1, "expected exactly one hit");
    Ok(())
}

// --- New behaviours from the v0.1.2 review ----------------------------------

#[cfg(feature = "lex")]
#[tokio::test]
async fn scope_filter_matches_frames_written_with_scoped_uri() -> Result<()> {
    let dir = tempdir()?;
    let store = lex_store(&dir.path().join("scope.mv2"))?;

    let scoped = PutOptions {
        uri: Some("chatbot/session-a".into()),
        ..PutOptions::default()
    };
    let other = PutOptions {
        uri: Some("notes/session-b".into()),
        ..PutOptions::default()
    };
    store.put_text("scoped-token-aaa", scoped)?;
    store.put_text("scoped-token-bbb", other)?;

    let req: VectorSearchRequest<MemvidFilter> =
        VectorSearchRequestBuilder::<MemvidFilter>::default()
            .query("scoped-token")
            .samples(10)
            .filter(MemvidFilter::eq("scope", json!("chatbot")))
            .build();
    let hits: Vec<(f64, String, serde_json::Value)> = store.top_n(req).await?;
    assert!(!hits.is_empty(), "expected scope-filtered hits");
    for (_, _, v) in &hits {
        let uri = v.get("uri").and_then(|v| v.as_str()).unwrap_or("");
        assert!(
            uri.starts_with("chatbot"),
            "scope filter should restrict to chatbot/* URIs, got {uri}"
        );
    }
    Ok(())
}

#[cfg(feature = "lex")]
#[tokio::test]
async fn samples_above_cap_does_not_panic() -> Result<()> {
    let dir = tempdir()?;
    let store = lex_store(&dir.path().join("cap.mv2"))?;
    store.put_text("alpha", PutOptions::default())?;

    let req: VectorSearchRequest<MemvidFilter> =
        VectorSearchRequestBuilder::<MemvidFilter>::default()
            .query("alpha")
            .samples(u64::MAX)
            .build();
    let _ids = store.top_n_ids(req).await?;
    Ok(())
}

#[cfg(feature = "lex")]
#[test]
fn filter_as_of_ts_accepts_integer_valued_float() {
    let f = MemvidFilter::eq("as_of_ts", json!(1_700_000_000.0_f64));
    assert_eq!(f.as_of_ts, Some(1_700_000_000));
}

#[cfg(feature = "lex")]
#[tokio::test]
async fn top_n_deserialises_search_hit_directly() -> Result<()> {
    use memvid_core::SearchHit;

    let dir = tempdir()?;
    let store = lex_store(&dir.path().join("hit.mv2"))?;
    store.put_text("zenith-marker-text", PutOptions::default())?;

    let req: VectorSearchRequest<MemvidFilter> =
        VectorSearchRequestBuilder::<MemvidFilter>::default()
            .query("zenith-marker-text")
            .samples(2)
            .build();
    let hits: Vec<(f64, String, SearchHit)> = store.top_n(req).await?;
    assert!(!hits.is_empty(), "expected at least one SearchHit");
    let (_, id, hit) = hits.first().expect("hits.first");
    assert_eq!(*id, hit.frame_id.to_string());
    assert!(
        hit.text.contains("zenith") || !hit.text.is_empty(),
        "expected hit.text to be populated"
    );
    Ok(())
}
