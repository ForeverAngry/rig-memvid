//! End-to-end smoke test for `MemvidStore` against a real memvid archive.
#![allow(clippy::panic_in_result_fn)]

use anyhow::Result;
use memvid_core::PutOptions;
use rig::{
    Embed, OneOrMany,
    embeddings::{Embedding, embed::EmbedError},
    vector_store::{
        InsertDocuments, VectorSearchRequest, VectorStoreIndex, request::VectorSearchRequestBuilder,
    },
};
use rig_memvid::{MemvidFilter, MemvidStore};
use serde::{Deserialize, Serialize};
use tempfile::tempdir;

#[cfg(feature = "lex")]
#[tokio::test]
async fn put_then_top_n_returns_hit() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("test.mv2");

    let store = MemvidStore::builder().path(&path).enable_lex().create()?;

    store.put_text(
        "The Tower of London was founded by William the Conqueror in 1066.",
        PutOptions::default(),
    )?;
    store.put_text(
        "Pluto was reclassified as a dwarf planet in 2006.",
        PutOptions::default(),
    )?;

    let req: VectorSearchRequest<MemvidFilter> =
        VectorSearchRequestBuilder::<MemvidFilter>::default()
            .query("Tower of London")
            .samples(5)
            .build();

    let hits: Vec<(f64, String, serde_json::Value)> = store.top_n(req).await?;
    assert!(!hits.is_empty(), "expected at least one hit for lex search");
    let combined = hits
        .iter()
        .map(|(_, _, v)| v.to_string())
        .collect::<String>();
    assert!(
        combined.to_lowercase().contains("tower"),
        "expected matching text in hits, got: {combined}"
    );
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Doc {
    id: String,
    body: String,
}

impl Embed for Doc {
    fn embed(&self, embedder: &mut rig::embeddings::TextEmbedder) -> Result<(), EmbedError> {
        embedder.embed(self.body.clone());
        Ok(())
    }
}

#[cfg(feature = "lex")]
#[tokio::test]
async fn insert_documents_then_top_n_ids() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("rig.mv2");
    let store = MemvidStore::builder().path(&path).enable_lex().create()?;

    let docs = vec![
        (
            Doc {
                id: "rig".into(),
                body: "Rig is a Rust library for building LLM apps.".into(),
            },
            OneOrMany::one(Embedding {
                document: "Rig is a Rust library for building LLM apps.".into(),
                vec: Vec::new(),
            }),
        ),
        (
            Doc {
                id: "memvid".into(),
                body: "Memvid is a high-performance vector memory file format.".into(),
            },
            OneOrMany::one(Embedding {
                document: "Memvid is a high-performance vector memory file format.".into(),
                vec: Vec::new(),
            }),
        ),
    ];

    store.insert_documents(docs).await?;

    let req: VectorSearchRequest<MemvidFilter> =
        VectorSearchRequestBuilder::<MemvidFilter>::default()
            .query("Rust library")
            .samples(1)
            .build();

    let ids = store.top_n_ids(req).await?;
    assert_eq!(ids.len(), 1, "expected exactly one hit");

    let req: VectorSearchRequest<MemvidFilter> =
        VectorSearchRequestBuilder::<MemvidFilter>::default()
            .query("Rust library")
            .samples(1)
            .build();
    let hits: Vec<(f64, String, serde_json::Value)> = store.top_n(req).await?;
    assert_eq!(hits.len(), 1);
    let payload = hits
        .first()
        .map(|(_, _, v)| v.to_string())
        .unwrap_or_default();
    assert!(
        payload.contains("Rust library"),
        "expected hit to contain inserted text, got: {payload}"
    );
    Ok(())
}

/// Semantic retrieval via the `vec` feature: query terms do *not* appear in
/// the indexed text, so a successful match proves the vector index — not
/// BM25 — is doing the work.
///
/// This test downloads the BGE-small ONNX model on first run (≈30 MB)
/// into the user's cache directory and is therefore marked `#[ignore]`.
/// Run it explicitly with:
///
/// ```bash
/// cargo test --features vec -- --ignored vec_semantic_search
/// ```
#[cfg(feature = "vec")]
#[tokio::test]
#[ignore = "downloads BGE-small ONNX model on first run"]
async fn vec_semantic_search_finds_paraphrase() -> Result<()> {
    use memvid_core::TextEmbedConfig;

    let dir = tempdir()?;
    let path = dir.path().join("vec.mv2");

    // Allow auto-download for CI/dev machines that haven't cached the model.
    let cfg = TextEmbedConfig {
        offline: false,
        ..TextEmbedConfig::bge_small()
    };

    let store = MemvidStore::builder()
        .path(&path)
        .with_embedder_config(cfg)?
        .create()?;

    store.put_text(
        "Photosynthesis converts sunlight into chemical energy in plants.",
        PutOptions::default(),
    )?;
    store.put_text(
        "The Eiffel Tower is a wrought-iron lattice tower in Paris.",
        PutOptions::default(),
    )?;

    let req: VectorSearchRequest<MemvidFilter> =
        VectorSearchRequestBuilder::<MemvidFilter>::default()
            .query("how do leaves make food from light")
            .samples(2)
            .build();

    let hits: Vec<(f64, String, serde_json::Value)> = store.top_n(req).await?;
    assert!(!hits.is_empty(), "expected at least one vector hit");
    let top = hits
        .first()
        .map(|(_, _, v)| v.to_string())
        .unwrap_or_default()
        .to_lowercase();
    assert!(
        top.contains("photosynthesis"),
        "expected top vector hit to be the photosynthesis frame, got: {top}"
    );
    Ok(())
}
