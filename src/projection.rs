//! Project memvid retrieval hits into [`rig_compose::ContextItem`].
//!
//! Memvid surfaces two retrieval shapes:
//!
//! - [`memvid_core::SearchHit`] from `.mv2`-backed [`crate::MemvidStore`]
//!   lookups (lexical or vector),
//! - [`memvid_core::MemoryCard`] and [`crate::CardDoc`] from structured
//!   memory-card context, and
//! - [`crate::InMemoryHit`] from the no-disk in-process
//!   [`crate::InMemoryStore`].
//!
//! Both carry the same conceptual payload — a ranked, scored snippet
//! plus enough provenance to trace where it came from — but neither
//! shape is directly consumable by a host that wants to fold memory
//! recall into a bounded [`rig_compose::ContextPack`]. This module
//! provides the projection vocabulary, mirroring the one in
//! `rig-resources`, so a coordinator can mix memvid candidates with
//! resources, files, and tool results under a single `Vec<ContextItem>`
//! API.
//!
//! All items emitted here use [`rig_compose::ContextSourceKind::Memory`]
//! and carry a structured `provenance` JSON object so downstream
//! evaluators (RAG harnesses, attribution UIs, signed-evidence ledgers)
//! can identify the originating frame or in-memory key without
//! re-running retrieval.
//!
//! ```no_run
//! # #[cfg(feature = "context-projection")]
//! # fn demo(hits: &[memvid_core::SearchHit]) {
//! use rig_memvid::projection::search_hits_to_context_items;
//!
//! let items = search_hits_to_context_items(hits);
//! assert!(items.iter().all(|i| matches!(
//!     i.source,
//!     rig_compose::ContextSourceKind::Memory
//! )));
//! # }
//! ```

use memvid_core::{MemoryCard, SearchHit};
use rig_compose::{ContextItem, ContextSourceKind};
use serde_json::{Map, Value, json};

use crate::cards_context::{CardDoc, format_card, kind_str, polarity_str};
use crate::inmem::{Episode, InMemoryHit};

/// Convert a domain-native retrieval hit into a backend-neutral
/// [`ContextItem`].
///
/// Implementors are responsible for choosing a stable `source_id`,
/// mapping their native score into `f64`, and emitting a JSON
/// `provenance` object that downstream evaluators can rely on. The
/// returned item carries [`ContextSourceKind::Memory`] for both
/// in-process and on-disk memvid retrieval.
pub trait IntoContextItem {
    /// Project `self` into a single context item with the requested
    /// `rank`. Callers (e.g. [`search_hits_to_context_items`]) enumerate
    /// the slice and pass an ascending zero-based rank.
    fn to_context_item(&self, rank: usize) -> ContextItem;
}

/// Map a [`SearchHit::score`] into the `f64` shape expected by
/// [`ContextItem::with_score`]. When the underlying engine did not
/// produce a score, we fall back to `1 / (rank + 1)` so hits remain
/// monotonically ordered for packers that key off `score` alone.
#[must_use]
fn search_hit_score(hit: &SearchHit) -> f64 {
    match hit.score {
        Some(score) => f64::from(score),
        None => {
            // `hit.rank` is `usize`; cap before promoting to f64 to avoid
            // silent precision loss on 64-bit platforms.
            let rank = u32::try_from(hit.rank).unwrap_or(u32::MAX);
            1.0 / f64::from(rank.saturating_add(1))
        }
    }
}

fn search_hit_provenance(hit: &SearchHit) -> Value {
    let mut provenance = Map::new();
    provenance.insert("resource".into(), Value::String("memvid.search".into()));
    provenance.insert("frame_id".into(), Value::String(hit.frame_id.to_string()));
    provenance.insert("uri".into(), Value::String(hit.uri.clone()));
    provenance.insert("rank".into(), json!(hit.rank));
    provenance.insert("matches".into(), json!(hit.matches));
    let (range_start, range_end) = hit.range;
    provenance.insert("range".into(), json!([range_start, range_end]));
    if let Some(score) = hit.score {
        provenance.insert("score".into(), json!(score));
    }
    if let Some(title) = hit.title.as_ref() {
        provenance.insert("title".into(), Value::String(title.clone()));
    }
    if let Some(chunk_range) = hit.chunk_range {
        let (start, end) = chunk_range;
        provenance.insert("chunk_range".into(), json!([start, end]));
    }
    if let Some(metadata) = hit.metadata.as_ref()
        && let Ok(value) = serde_json::to_value(metadata)
    {
        provenance.insert("metadata".into(), value);
    }
    Value::Object(provenance)
}

fn fallback_score(rank: usize) -> f64 {
    let rank = u32::try_from(rank).unwrap_or(u32::MAX);
    1.0 / f64::from(rank.saturating_add(1))
}

fn card_score(confidence: Option<f32>, rank: usize) -> f64 {
    confidence
        .map(f64::from)
        .unwrap_or_else(|| fallback_score(rank))
}

fn polarity_value(polarity: impl Into<Option<String>>) -> Value {
    match polarity.into() {
        Some(value) => Value::String(value),
        None => Value::Null,
    }
}

fn memory_card_source_id(card: &MemoryCard) -> String {
    if card.id == 0 {
        format!(
            "card/{entity}/{slot}/{frame}",
            entity = card.entity,
            slot = card.slot,
            frame = card.source_frame_id
        )
    } else {
        format!("card/{}", card.id)
    }
}

fn card_doc_source_id(doc: &CardDoc) -> String {
    format!(
        "card/{entity}/{slot}/{frame}",
        entity = doc.entity,
        slot = doc.slot,
        frame = doc.source_frame_id
    )
}

fn memory_card_provenance(card: &MemoryCard) -> Value {
    let mut provenance = Map::new();
    provenance.insert("schema_version".into(), json!(1));
    provenance.insert("resource".into(), Value::String("memvid.card".into()));
    provenance.insert("card_id".into(), json!(card.id));
    provenance.insert("entity".into(), Value::String(card.entity.clone()));
    provenance.insert("slot".into(), Value::String(card.slot.clone()));
    provenance.insert(
        "kind".into(),
        Value::String(kind_str(card.kind).to_string()),
    );
    provenance.insert("polarity".into(), json!(card.polarity.map(polarity_str)));
    provenance.insert("source_frame_id".into(), json!(card.source_frame_id));
    provenance.insert("source_uri".into(), json!(card.source_uri));
    provenance.insert("engine".into(), Value::String(card.engine.clone()));
    provenance.insert("confidence".into(), json!(card.confidence));
    Value::Object(provenance)
}

fn card_doc_provenance(doc: &CardDoc) -> Value {
    json!({
        "schema_version": 1,
        "resource": "memvid.card",
        "entity": doc.entity,
        "slot": doc.slot,
        "kind": doc.kind,
        "polarity": polarity_value(doc.polarity.clone()),
        "source_frame_id": doc.source_frame_id,
        "confidence": doc.confidence,
    })
}

impl IntoContextItem for SearchHit {
    fn to_context_item(&self, rank: usize) -> ContextItem {
        ContextItem::new(
            ContextSourceKind::Memory,
            self.frame_id.to_string(),
            self.text.clone(),
        )
        .with_rank(rank)
        .with_score(search_hit_score(self))
        .with_provenance(search_hit_provenance(self))
    }
}

impl<E: Episode> IntoContextItem for InMemoryHit<E> {
    fn to_context_item(&self, rank: usize) -> ContextItem {
        let provenance = json!({
            "resource": "memvid.inmem",
            "key": self.key,
            "score": self.score,
        });
        ContextItem::new(
            ContextSourceKind::Memory,
            self.key.clone(),
            self.episode.summary().to_string(),
        )
        .with_rank(rank)
        .with_score(f64::from(self.score))
        .with_provenance(provenance)
    }
}

impl IntoContextItem for MemoryCard {
    fn to_context_item(&self, rank: usize) -> ContextItem {
        ContextItem::new(
            ContextSourceKind::Memory,
            memory_card_source_id(self),
            format_card(self),
        )
        .with_rank(rank)
        .with_score(card_score(self.confidence, rank))
        .with_provenance(memory_card_provenance(self))
    }
}

impl IntoContextItem for CardDoc {
    fn to_context_item(&self, rank: usize) -> ContextItem {
        ContextItem::new(
            ContextSourceKind::Memory,
            card_doc_source_id(self),
            self.text.clone(),
        )
        .with_rank(rank)
        .with_score(card_score(self.confidence, rank))
        .with_provenance(card_doc_provenance(self))
    }
}

/// Project a slice of [`SearchHit`]s into ranked [`ContextItem`]s.
///
/// The returned vector preserves input order; rank is taken from the
/// slice position rather than [`SearchHit::rank`] so callers can
/// pre-filter or re-order before projection without re-numbering by
/// hand.
#[must_use]
pub fn search_hits_to_context_items(hits: &[SearchHit]) -> Vec<ContextItem> {
    hits.iter()
        .enumerate()
        .map(|(rank, hit)| hit.to_context_item(rank))
        .collect()
}

/// Project a slice of [`InMemoryHit`]s into ranked [`ContextItem`]s.
#[must_use]
pub fn inmem_hits_to_context_items<E: Episode>(hits: &[InMemoryHit<E>]) -> Vec<ContextItem> {
    hits.iter()
        .enumerate()
        .map(|(rank, hit)| hit.to_context_item(rank))
        .collect()
}

/// Project a slice of [`MemoryCard`]s into ranked [`ContextItem`]s.
#[must_use]
pub fn memory_cards_to_context_items(cards: &[MemoryCard]) -> Vec<ContextItem> {
    cards
        .iter()
        .enumerate()
        .map(|(rank, card)| card.to_context_item(rank))
        .collect()
}

/// Project a slice of [`CardDoc`]s into ranked [`ContextItem`]s.
#[must_use]
pub fn card_docs_to_context_items(docs: &[CardDoc]) -> Vec<ContextItem> {
    docs.iter()
        .enumerate()
        .map(|(rank, doc)| doc.to_context_item(rank))
        .collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use memvid_core::types::FrameId;
    use memvid_core::{MemoryKind, Polarity, VersionRelation};
    use rig_compose::{ContextOmissionReason, ContextPack, ContextPackConfig};

    #[derive(Clone)]
    struct StubEpisode(&'static str);

    impl Episode for StubEpisode {
        fn summary(&self) -> &str {
            self.0
        }
    }

    fn make_hit(rank: usize, score: Option<f32>, frame: FrameId) -> SearchHit {
        SearchHit {
            rank,
            frame_id: frame,
            uri: format!("memvid://frame/{frame}"),
            title: Some("note".into()),
            range: (0, 8),
            text: "snippet ".into(),
            matches: 1,
            chunk_range: Some((0, 8)),
            chunk_text: None,
            score,
            metadata: None,
        }
    }

    fn make_card(id: u64, confidence: Option<f32>) -> MemoryCard {
        MemoryCard {
            id,
            kind: MemoryKind::Preference,
            entity: "Ada".into(),
            slot: "drink".into(),
            value: "espresso".into(),
            polarity: Some(Polarity::Positive),
            event_date: None,
            document_date: None,
            version_key: Some("Ada:drink".into()),
            version_relation: VersionRelation::Sets,
            source_frame_id: 42,
            source_uri: Some("memvid://frame/42".into()),
            source_offset: Some((7, 15)),
            engine: "test-extractor".into(),
            engine_version: "1".into(),
            confidence,
            created_at: 99,
        }
    }

    fn make_card_doc() -> CardDoc {
        CardDoc {
            text: "pref Ada likes espresso".into(),
            kind: "pref".into(),
            entity: "Ada".into(),
            slot: "drink".into(),
            value: "espresso".into(),
            polarity: Some("positive".into()),
            source_frame_id: 42,
            confidence: Some(0.75),
        }
    }

    #[test]
    fn search_hit_projects_memory_item_with_score_and_provenance() {
        let hit = make_hit(0, Some(0.5), 42);
        let item = hit.to_context_item(0);
        assert!(matches!(item.source, ContextSourceKind::Memory));
        assert_eq!(item.source_id, "42");
        assert_eq!(item.rank, 0);
        assert!((item.score - 0.5).abs() < 1e-6);
        assert_eq!(item.text, "snippet ");
        let provenance = item.provenance.as_object().unwrap();
        assert_eq!(provenance["resource"], "memvid.search");
        assert_eq!(provenance["frame_id"], "42");
        assert_eq!(provenance["uri"], "memvid://frame/42");
        assert_eq!(provenance["title"], "note");
    }

    #[test]
    fn missing_score_falls_back_to_inverse_rank() {
        let hit = make_hit(2, None, 7);
        let item = hit.to_context_item(2);
        // 1 / (rank + 1) where rank = 2 → 1/3
        assert!((item.score - (1.0_f64 / 3.0)).abs() < 1e-6);
    }

    #[test]
    fn inmem_hit_projects_with_key_as_source_id() {
        let hit = InMemoryHit {
            episode: StubEpisode("maintenance window opens at noon"),
            score: 0.75,
            key: "ep-0000000000000001".into(),
        };
        let item = hit.to_context_item(3);
        assert!(matches!(item.source, ContextSourceKind::Memory));
        assert_eq!(item.source_id, "ep-0000000000000001");
        assert_eq!(item.rank, 3);
        assert_eq!(item.text, "maintenance window opens at noon");
        let provenance = item.provenance.as_object().unwrap();
        assert_eq!(provenance["resource"], "memvid.inmem");
        assert_eq!(provenance["key"], "ep-0000000000000001");
    }

    #[test]
    fn slice_projection_assigns_ascending_ranks() {
        let hits = vec![
            make_hit(0, Some(0.9), 1),
            make_hit(1, Some(0.5), 2),
            make_hit(2, Some(0.1), 3),
        ];
        let items = search_hits_to_context_items(&hits);
        assert_eq!(items.len(), 3);
        for (idx, item) in items.iter().enumerate() {
            assert_eq!(item.rank, idx);
        }
        assert_eq!(items[0].source_id, "1");
        assert_eq!(items[2].source_id, "3");
    }

    #[test]
    fn inmem_slice_projection_enumerates_ranks() {
        let hits = vec![
            InMemoryHit {
                episode: StubEpisode("alpha"),
                score: 0.9,
                key: "ep-a".into(),
            },
            InMemoryHit {
                episode: StubEpisode("beta"),
                score: 0.4,
                key: "ep-b".into(),
            },
        ];
        let items = inmem_hits_to_context_items(&hits);
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].rank, 0);
        assert_eq!(items[1].rank, 1);
        assert_eq!(items[1].source_id, "ep-b");
    }

    #[test]
    fn memory_card_projects_compact_text_and_provenance() {
        let card = make_card(77, Some(0.8));
        let item = card.to_context_item(4);
        assert!(matches!(item.source, ContextSourceKind::Memory));
        assert_eq!(item.source_id, "card/77");
        assert_eq!(item.rank, 4);
        assert!((item.score - 0.8).abs() < 1e-6);
        assert_eq!(item.text, "pref Ada likes espresso");

        let provenance = item.provenance.as_object().unwrap();
        assert_eq!(provenance["schema_version"], 1);
        assert_eq!(provenance["resource"], "memvid.card");
        assert_eq!(provenance["card_id"], 77);
        assert_eq!(provenance["entity"], "Ada");
        assert_eq!(provenance["slot"], "drink");
        assert_eq!(provenance["kind"], "pref");
        assert_eq!(provenance["polarity"], "positive");
        assert_eq!(provenance["source_frame_id"], 42);
        assert_eq!(provenance["source_uri"], "memvid://frame/42");
        assert_eq!(provenance["engine"], "test-extractor");
        let confidence = provenance["confidence"].as_f64().unwrap();
        assert!((confidence - 0.8).abs() < 1e-6);
    }

    #[test]
    fn memory_card_without_id_or_confidence_gets_stable_fallbacks() {
        let card = make_card(0, None);
        let item = card.to_context_item(2);
        assert_eq!(item.source_id, "card/Ada/drink/42");
        assert!((item.score - (1.0_f64 / 3.0)).abs() < 1e-6);
    }

    #[test]
    fn card_doc_projects_with_confidence_and_card_provenance() {
        let doc = make_card_doc();
        let item = doc.to_context_item(1);
        assert_eq!(item.source_id, "card/Ada/drink/42");
        assert_eq!(item.rank, 1);
        assert!((item.score - 0.75).abs() < 1e-6);
        assert_eq!(item.text, "pref Ada likes espresso");

        let provenance = item.provenance.as_object().unwrap();
        assert_eq!(provenance["resource"], "memvid.card");
        assert_eq!(provenance["schema_version"], 1);
        assert_eq!(provenance["entity"], "Ada");
        assert_eq!(provenance["polarity"], "positive");
        let confidence = provenance["confidence"].as_f64().unwrap();
        assert!((confidence - 0.75).abs() < 1e-6);
    }

    #[test]
    fn card_and_search_items_pack_with_stable_omissions() {
        let mut items = memory_cards_to_context_items(&[make_card(1, Some(0.9))]);
        items.extend(search_hits_to_context_items(&[make_hit(1, Some(0.4), 10)]));
        let mut docs = card_docs_to_context_items(&[CardDoc {
            text: "this card is deliberately too large for the remaining budget".into(),
            confidence: None,
            ..make_card_doc()
        }]);
        docs[0].rank = 2;
        items.extend(docs);

        items[0].rank = 0;
        items[1].rank = 1;
        let pack = ContextPack::pack(items, ContextPackConfig::new(50).with_max_items(8));

        assert_eq!(pack.selected.len(), 2);
        assert_eq!(pack.selected[0].source_id, "card/1");
        assert_eq!(pack.selected[1].source_id, "10");
        assert_eq!(pack.omitted.len(), 1);
        assert_eq!(pack.omitted[0].reason, ContextOmissionReason::OverBudget);
        let provenance = pack.selected[0].provenance.as_object().unwrap();
        assert_eq!(provenance["resource"], "memvid.card");
    }
}
