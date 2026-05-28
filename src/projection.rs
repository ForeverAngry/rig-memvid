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

const STATE_CANDIDATE: &str = "candidate";

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
    provenance.insert("source_frame_id".into(), json!(hit.frame_id));
    provenance.insert("uri".into(), Value::String(hit.uri.clone()));
    provenance.insert("source_uri".into(), Value::String(hit.uri.clone()));
    provenance.insert("rank".into(), json!(hit.rank));
    provenance.insert("matches".into(), json!(hit.matches));
    provenance.insert(
        "projection_state".into(),
        Value::String(STATE_CANDIDATE.into()),
    );
    let (range_start, range_end) = hit.range;
    provenance.insert("range".into(), json!([range_start, range_end]));
    if let Some(score) = hit.score {
        provenance.insert("score".into(), json!(score));
        provenance.insert("confidence".into(), json!(score));
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
    provenance.insert("principal".into(), Value::String(card.entity.clone()));
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
    provenance.insert("recorded_at_millis".into(), json!(card.created_at));
    provenance.insert(
        "projection_state".into(),
        Value::String(STATE_CANDIDATE.into()),
    );
    // Supersession-relevant fields. `version_key` groups two cards that
    // describe the same fact across writes; `effective_timestamp` is the
    // monotonic recency signal used to pick a survivor.
    provenance.insert(
        "version_key".into(),
        Value::String(
            card.version_key
                .clone()
                .unwrap_or_else(|| card.default_version_key()),
        ),
    );
    provenance.insert(
        "effective_timestamp".into(),
        json!(card.effective_timestamp()),
    );
    provenance.insert(
        "effective_at_millis".into(),
        json!(card.effective_timestamp()),
    );
    if let Some(event_date) = card.event_date {
        provenance.insert("event_date".into(), json!(event_date));
    }
    if let Some(document_date) = card.document_date {
        provenance.insert("document_date".into(), json!(document_date));
    }
    Value::Object(provenance)
}

fn card_doc_provenance(doc: &CardDoc) -> Value {
    json!({
        "schema_version": 1,
        "resource": "memvid.card",
        "entity": doc.entity,
        "principal": doc.entity,
        "slot": doc.slot,
        "kind": doc.kind,
        "polarity": polarity_value(doc.polarity.clone()),
        "source_frame_id": doc.source_frame_id,
        "confidence": doc.confidence,
        "projection_state": STATE_CANDIDATE,
        // CardDoc lacks explicit version/event timestamps; default the
        // supersession key to entity:slot and use source_frame_id as the
        // monotonic recency proxy (memvid frames are append-only).
        "version_key": format!("{}:{}", doc.entity, doc.slot),
        "effective_timestamp": doc.source_frame_id,
        "effective_at_millis": doc.source_frame_id,
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
            "source_uri": format!("memory://inmem/{}", self.key),
            "key": self.key,
            "source_frame_id": self.key,
            "score": self.score,
            "confidence": self.score,
            "projection_state": STATE_CANDIDATE,
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

// ── Memory candidates + supersession ────────────────────────────────────────

/// Typed wrapper around a memory-sourced [`ContextItem`].
///
/// Projections in this module emit `ContextItem`s with
/// [`ContextSourceKind::Memory`]; wrapping them as `MemoryCandidate`
/// records that fact in the type system and exposes accessors for the
/// supersession-relevant provenance fields (`version_key`,
/// `effective_timestamp`, `source_frame_id`).
///
/// Build candidates via [`MemoryCandidate::from_item`] or the bulk
/// conversion in [`MemoryContextPack::from_candidates`] / the projection
/// helpers in this module.
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryCandidate {
    item: ContextItem,
}

impl MemoryCandidate {
    /// Wrap an existing context item as a memory candidate.
    ///
    /// Callers are responsible for ensuring `item.source` is
    /// [`ContextSourceKind::Memory`]; the wrapper does not re-tag the
    /// underlying item.
    #[must_use]
    pub fn from_item(item: ContextItem) -> Self {
        Self { item }
    }

    /// Borrow the underlying context item.
    #[must_use]
    pub fn as_item(&self) -> &ContextItem {
        &self.item
    }

    /// Consume the wrapper and return the inner context item.
    #[must_use]
    pub fn into_item(self) -> ContextItem {
        self.item
    }

    /// Supersession-grouping key as emitted by this module's projection
    /// functions, or `None` if the candidate carries no version key.
    /// Candidates without a key never supersede or are superseded by
    /// any other candidate.
    #[must_use]
    pub fn version_key(&self) -> Option<&str> {
        self.item
            .provenance
            .as_object()
            .and_then(|map| map.get("version_key"))
            .and_then(Value::as_str)
    }

    /// Monotonic recency used to choose a survivor inside a
    /// supersession group. Resolved in priority order from
    /// `provenance.effective_timestamp` then `provenance.source_frame_id`.
    /// Returns `None` if neither is present, in which case the
    /// algorithm falls back to first-occurrence order.
    #[must_use]
    pub fn recency(&self) -> Option<i64> {
        let map = self.item.provenance.as_object()?;
        if let Some(v) = map.get("effective_timestamp").and_then(Value::as_i64) {
            return Some(v);
        }
        // `source_frame_id` is unsigned in memvid; clamp to i64 range.
        map.get("source_frame_id")
            .and_then(Value::as_u64)
            .map(|v| i64::try_from(v).unwrap_or(i64::MAX))
    }
}

impl From<MemoryCandidate> for ContextItem {
    fn from(candidate: MemoryCandidate) -> Self {
        candidate.into_item()
    }
}

/// Record of one candidate hidden by a newer survivor.
#[derive(Debug, Clone, PartialEq)]
pub struct SupersededCandidate {
    /// `source_id` of the surviving candidate that hides this one.
    pub survivor_source_id: String,
    /// `version_key` shared by survivor and hidden candidate.
    pub version_key: String,
    /// The hidden candidate.
    pub hidden: MemoryCandidate,
}

/// Result of running [`MemoryContextPack::from_candidates`] over a
/// ranked list of [`MemoryCandidate`]s.
///
/// `kept` is the deduplicated set in deterministic input order:
/// candidates without a `version_key` retain their original position;
/// each group of candidates sharing a `version_key` is collapsed to the
/// survivor (highest [`MemoryCandidate::recency`], ties broken by input
/// order) emitted at the position of the group's first occurrence.
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryContextPack {
    /// Candidates surviving supersession, in deterministic order.
    pub kept: Vec<MemoryCandidate>,
    /// Candidates hidden by a newer survivor.
    pub superseded: Vec<SupersededCandidate>,
}

impl MemoryContextPack {
    /// Apply supersession to `candidates` and return a deduplicated pack.
    ///
    /// Candidates without a `version_key` are always kept; candidates
    /// with the same `version_key` collapse to one survivor (highest
    /// `recency`, ties broken by lowest input index). Output order
    /// preserves first-occurrence positions so downstream
    /// [`rig_compose::ContextPack::pack`] sees the same surface even
    /// when an older version of a card was ranked above its newer
    /// replacement.
    #[must_use]
    pub fn from_candidates(candidates: Vec<MemoryCandidate>) -> Self {
        use std::collections::HashMap;
        use std::collections::hash_map::Entry;

        // Track per-group state by version_key. `first_idx` fixes the
        // output position; `survivor_idx` may be updated as we walk.
        struct GroupState {
            first_idx: usize,
            survivor_idx: usize,
            survivor_recency: Option<i64>,
            members: Vec<usize>,
        }
        let mut groups: HashMap<String, GroupState> = HashMap::new();
        for (idx, candidate) in candidates.iter().enumerate() {
            match candidate.version_key() {
                None => {}
                Some(key) => match groups.entry(key.to_string()) {
                    Entry::Vacant(slot) => {
                        let recency = candidate.recency();
                        slot.insert(GroupState {
                            first_idx: idx,
                            survivor_idx: idx,
                            survivor_recency: recency,
                            members: vec![idx],
                        });
                    }
                    Entry::Occupied(mut slot) => {
                        let g = slot.get_mut();
                        g.members.push(idx);
                        let new_recency = candidate.recency();
                        // Strictly greater recency wins; ties are kept
                        // by first-occurrence (i.e. we do not update).
                        let should_replace = match (new_recency, g.survivor_recency) {
                            (Some(n), Some(s)) => n > s,
                            (Some(_), None) => true,
                            (None, _) => false,
                        };
                        if should_replace {
                            g.survivor_idx = idx;
                            g.survivor_recency = new_recency;
                        }
                    }
                },
            }
        }

        // Build output in input-position order: at each position, emit
        // either the no-key candidate or the group survivor (only at
        // the group's first occurrence).
        let mut kept: Vec<MemoryCandidate> = Vec::new();
        let mut superseded: Vec<SupersededCandidate> = Vec::new();
        let mut emitted_groups: std::collections::HashSet<String> =
            std::collections::HashSet::new();

        for (idx, candidate) in candidates.iter().enumerate() {
            match candidate.version_key() {
                None => kept.push(candidate.clone()),
                Some(key) => {
                    if emitted_groups.contains(key) {
                        continue;
                    }
                    let Some(group) = groups.get(key) else {
                        continue;
                    };
                    if group.first_idx != idx {
                        continue;
                    }
                    emitted_groups.insert(key.to_string());
                    let Some(survivor) = candidates.get(group.survivor_idx) else {
                        continue;
                    };
                    let survivor_source_id = survivor.as_item().source_id.clone();
                    kept.push(survivor.clone());
                    for &member_idx in &group.members {
                        if member_idx == group.survivor_idx {
                            continue;
                        }
                        let Some(hidden) = candidates.get(member_idx) else {
                            continue;
                        };
                        superseded.push(SupersededCandidate {
                            survivor_source_id: survivor_source_id.clone(),
                            version_key: key.to_string(),
                            hidden: hidden.clone(),
                        });
                    }
                }
            }
        }

        Self { kept, superseded }
    }

    /// Convert the kept candidates into a `Vec<ContextItem>` ready for
    /// [`rig_compose::ContextPack::pack`].
    #[must_use]
    pub fn into_context_items(self) -> Vec<ContextItem> {
        self.kept
            .into_iter()
            .map(MemoryCandidate::into_item)
            .collect()
    }
}

/// Convenience: wrap owned context items as memory candidates.
#[must_use]
pub fn items_to_memory_candidates(items: Vec<ContextItem>) -> Vec<MemoryCandidate> {
    items.into_iter().map(MemoryCandidate::from_item).collect()
}

/// Convenience: run supersession over a `Vec<ContextItem>` directly.
#[must_use]
pub fn supersede(items: Vec<ContextItem>) -> MemoryContextPack {
    MemoryContextPack::from_candidates(items_to_memory_candidates(items))
}

// ── Frame-typed projection (compaction-aware) ───────────────────────────────

#[cfg(feature = "compaction")]
pub use frame_typed::{
    MemoryFrameRole, PartitionedHits, frame_role, partition_search_hits_by_role,
    typed_search_hit_to_context_item, typed_search_hits_to_context_items,
    typed_search_hits_to_memory_candidates,
};

#[cfg(feature = "compaction")]
mod frame_typed {
    //! Role-aware projection for search hits whose frames carry a
    //! [`MemvidFrameMetadata`](crate::metadata::MemvidFrameMetadata)
    //! envelope.
    //!
    //! `rig-memvid`'s compaction surface
    //! ([`crate::MemvidDemotionHook`], [`crate::MemvidStoringCompactor`])
    //! tags every written frame with `kind ∈ {demoted_message,
    //! compaction_summary}` plus `conversation_id`, `chat_role`,
    //! `dedup_key`, and optional `scope`. These helpers decode that
    //! envelope when present so downstream packers can distinguish
    //! "raw demoted turn" from "rolled-up summary" without re-running
    //! retrieval and so [`super::MemoryContextPack`] can supersede an
    //! older summary by a newer one with the same dedup key.

    use memvid_core::SearchHit;
    use rig_compose::{ContextItem, ContextSourceKind};
    use serde_json::{Map, Value, json};

    use super::{
        MemoryCandidate, STATE_CANDIDATE, fallback_score, search_hit_provenance, search_hit_score,
    };
    use crate::metadata::{FrameKind, MemvidFrameMetadata};

    /// Logical role of a memvid frame as recorded by the compaction
    /// envelope, or [`MemoryFrameRole::Raw`] when no envelope is present
    /// (e.g. ad-hoc `put_text` writes from outside the compaction path).
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    #[non_exhaustive]
    pub enum MemoryFrameRole {
        /// Frame without a `MemvidFrameMetadata` envelope.
        Raw,
        /// A single user/assistant turn evicted from the working set.
        DemotedMessage,
        /// A rolled-up summary of multiple evicted turns.
        CompactionSummary,
    }

    impl MemoryFrameRole {
        /// Snake-case discriminant suitable for provenance JSON.
        #[must_use]
        pub fn as_str(self) -> &'static str {
            match self {
                Self::Raw => "raw",
                Self::DemotedMessage => "demoted_message",
                Self::CompactionSummary => "compaction_summary",
            }
        }
    }

    impl From<FrameKind> for MemoryFrameRole {
        fn from(kind: FrameKind) -> Self {
            match kind {
                FrameKind::DemotedMessage => Self::DemotedMessage,
                FrameKind::CompactionSummary => Self::CompactionSummary,
            }
        }
    }

    /// Decode the [`MemvidFrameMetadata`] envelope attached to `hit`,
    /// returning [`MemoryFrameRole::Raw`] when the envelope is missing
    /// or fails to parse.
    #[must_use]
    pub fn frame_role(hit: &SearchHit) -> MemoryFrameRole {
        decode_envelope(hit)
            .map(|env| env.kind.into())
            .unwrap_or(MemoryFrameRole::Raw)
    }

    fn decode_envelope(hit: &SearchHit) -> Option<MemvidFrameMetadata> {
        let metadata = hit.metadata.as_ref()?;
        if metadata.extra_metadata.is_empty() {
            return None;
        }
        MemvidFrameMetadata::try_from_map(&metadata.extra_metadata).ok()
    }

    /// Project a single [`SearchHit`] into a [`ContextItem`] enriched
    /// with the frame-typed provenance keys.
    ///
    /// When the hit's `extra_metadata` decodes as
    /// [`MemvidFrameMetadata`], the returned item carries:
    /// - `provenance.frame_kind`: `demoted_message` | `compaction_summary`
    /// - `provenance.conversation_id`, `provenance.chat_role`,
    ///   `provenance.principal` (mirror of `chat_role`),
    ///   `provenance.dedup_key`, optional `provenance.scope`,
    /// - `provenance.version_key`: stable supersession key derived from
    ///   the role + dedup hash (so re-rolled summaries collapse),
    /// - `provenance.effective_at_millis`: the frame id (memvid frames
    ///   are append-only, so frame id is a monotonic recency proxy).
    ///
    /// A role-aware `source_id` (`summary/{frame_id}` or
    /// `frame/{frame_id}`) replaces the bare frame id so duplicate
    /// `source_id`s do not collide between a raw and a summary view of
    /// the same frame range.
    ///
    /// When the envelope is missing the function falls back to the
    /// untyped projection used by [`super::search_hits_to_context_items`].
    #[must_use]
    pub fn typed_search_hit_to_context_item(hit: &SearchHit, rank: usize) -> ContextItem {
        let Some(envelope) = decode_envelope(hit) else {
            return ContextItem::new(
                ContextSourceKind::Memory,
                hit.frame_id.to_string(),
                hit.text.clone(),
            )
            .with_rank(rank)
            .with_score(search_hit_score(hit))
            .with_provenance(search_hit_provenance(hit));
        };

        let role = MemoryFrameRole::from(envelope.kind);
        let source_id = match role {
            MemoryFrameRole::CompactionSummary => format!("summary/{}", hit.frame_id),
            MemoryFrameRole::DemotedMessage => format!("frame/{}", hit.frame_id),
            // Unreachable: `decode_envelope` only returns Some when the
            // envelope decodes, and `FrameKind` has no `Raw` variant
            // today. Kept as a safe fall-through if the enum grows.
            MemoryFrameRole::Raw => hit.frame_id.to_string(),
        };
        let provenance = typed_provenance(hit, &envelope, role, rank);

        ContextItem::new(ContextSourceKind::Memory, source_id, hit.text.clone())
            .with_rank(rank)
            .with_score(search_hit_score(hit))
            .with_provenance(provenance)
    }

    fn typed_provenance(
        hit: &SearchHit,
        envelope: &MemvidFrameMetadata,
        role: MemoryFrameRole,
        rank: usize,
    ) -> Value {
        let mut provenance = Map::new();
        provenance.insert("schema_version".into(), json!(1));
        provenance.insert(
            "resource".into(),
            Value::String(match role {
                MemoryFrameRole::CompactionSummary => "memvid.summary".into(),
                MemoryFrameRole::DemotedMessage => "memvid.frame".into(),
                MemoryFrameRole::Raw => "memvid.search".into(),
            }),
        );
        provenance.insert("frame_kind".into(), Value::String(role.as_str().into()));
        provenance.insert("frame_id".into(), Value::String(hit.frame_id.to_string()));
        provenance.insert("source_frame_id".into(), json!(hit.frame_id));
        provenance.insert("uri".into(), Value::String(hit.uri.clone()));
        provenance.insert("source_uri".into(), Value::String(hit.uri.clone()));
        provenance.insert("rank".into(), json!(rank));
        provenance.insert("matches".into(), json!(hit.matches));
        provenance.insert(
            "projection_state".into(),
            Value::String(STATE_CANDIDATE.into()),
        );
        let (range_start, range_end) = hit.range;
        provenance.insert("range".into(), json!([range_start, range_end]));
        let score = match hit.score {
            Some(score) => f64::from(score),
            None => fallback_score(rank),
        };
        provenance.insert("score".into(), json!(score));
        provenance.insert("confidence".into(), json!(score));
        if let Some(title) = hit.title.as_ref() {
            provenance.insert("title".into(), Value::String(title.clone()));
        }
        provenance.insert(
            "conversation_id".into(),
            Value::String(envelope.conversation_id.clone()),
        );
        provenance.insert(
            "chat_role".into(),
            Value::String(envelope.chat_role.clone()),
        );
        // Mirror chat_role into the shared `principal` slot so packers
        // that scope by principal (e.g. multi-tenant inspectors) see a
        // value for compacted memory just like they do for cards.
        provenance.insert(
            "principal".into(),
            Value::String(envelope.chat_role.clone()),
        );
        provenance.insert(
            "dedup_key".into(),
            Value::String(envelope.dedup_key.clone()),
        );
        if let Some(scope) = envelope.scope.as_ref() {
            provenance.insert("scope".into(), Value::String(scope.clone()));
        }
        // Build a stable supersession key. Two writes of the same
        // dedup_key are content-identical by construction (blake3 over
        // the body); pinning the role keeps a raw frame from
        // superseding its later summary or vice versa.
        let version_key = format!("{}:{}", role.as_str(), envelope.dedup_key);
        provenance.insert("version_key".into(), Value::String(version_key));
        // Memvid frames are append-only, so `frame_id` is a monotonic
        // recency proxy: a later write of the same dedup_key would
        // land at a higher frame id and naturally win supersession.
        provenance.insert("effective_at_millis".into(), json!(hit.frame_id));
        provenance.insert("effective_timestamp".into(), json!(hit.frame_id));
        Value::Object(provenance)
    }

    /// Bulk variant of [`typed_search_hit_to_context_item`].
    #[must_use]
    pub fn typed_search_hits_to_context_items(hits: &[SearchHit]) -> Vec<ContextItem> {
        hits.iter()
            .enumerate()
            .map(|(rank, hit)| typed_search_hit_to_context_item(hit, rank))
            .collect()
    }

    /// Bulk projection that wraps each typed item as a
    /// [`MemoryCandidate`] ready for [`super::MemoryContextPack::from_candidates`].
    #[must_use]
    pub fn typed_search_hits_to_memory_candidates(hits: &[SearchHit]) -> Vec<MemoryCandidate> {
        typed_search_hits_to_context_items(hits)
            .into_iter()
            .map(MemoryCandidate::from_item)
            .collect()
    }

    /// Hits split by their decoded [`MemoryFrameRole`].
    ///
    /// Returned vectors preserve input order within each bucket so
    /// callers can re-rank or budget each role independently before
    /// re-merging.
    #[derive(Debug, Clone, Default)]
    pub struct PartitionedHits {
        /// Hits whose envelope was missing or undecodable.
        pub raw: Vec<SearchHit>,
        /// Hits tagged [`crate::metadata::FrameKind::DemotedMessage`].
        pub demoted: Vec<SearchHit>,
        /// Hits tagged [`crate::metadata::FrameKind::CompactionSummary`].
        pub summaries: Vec<SearchHit>,
    }

    /// Partition `hits` by [`MemoryFrameRole`].
    #[must_use]
    pub fn partition_search_hits_by_role(hits: &[SearchHit]) -> PartitionedHits {
        let mut out = PartitionedHits::default();
        for hit in hits {
            match frame_role(hit) {
                MemoryFrameRole::Raw => out.raw.push(hit.clone()),
                MemoryFrameRole::DemotedMessage => out.demoted.push(hit.clone()),
                MemoryFrameRole::CompactionSummary => out.summaries.push(hit.clone()),
            }
        }
        out
    }
}

#[cfg(feature = "compaction")]
impl MemoryContextPack {
    /// Project `hits` through [`typed_search_hits_to_memory_candidates`]
    /// and apply supersession.
    ///
    /// This is the recommended entry point for callers that retrieve
    /// from a compaction-backed `.mv2` archive: it preserves the order
    /// of the underlying ranker, decodes the frame envelope, and
    /// collapses any pair of hits that share a `(role, dedup_key)` —
    /// keeping the higher-frame-id survivor — so a re-rolled summary
    /// does not appear twice in the same pack.
    #[must_use]
    pub fn from_search_hits(hits: &[memvid_core::SearchHit]) -> Self {
        Self::from_candidates(typed_search_hits_to_memory_candidates(hits))
    }
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
        assert_eq!(provenance["source_frame_id"], 42);
        assert_eq!(provenance["uri"], "memvid://frame/42");
        assert_eq!(provenance["source_uri"], "memvid://frame/42");
        assert_eq!(provenance["projection_state"], "candidate");
        assert_eq!(provenance["title"], "note");
        let confidence = provenance["confidence"].as_f64().unwrap();
        assert!((confidence - 0.5).abs() < 1e-6);
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
        assert_eq!(
            provenance["source_uri"],
            "memory://inmem/ep-0000000000000001"
        );
        assert_eq!(provenance["source_frame_id"], "ep-0000000000000001");
        assert_eq!(provenance["projection_state"], "candidate");
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
        assert_eq!(provenance["principal"], "Ada");
        assert_eq!(provenance["slot"], "drink");
        assert_eq!(provenance["kind"], "pref");
        assert_eq!(provenance["polarity"], "positive");
        assert_eq!(provenance["source_frame_id"], 42);
        assert_eq!(provenance["source_uri"], "memvid://frame/42");
        assert_eq!(provenance["engine"], "test-extractor");
        assert_eq!(provenance["recorded_at_millis"], 99);
        assert_eq!(provenance["projection_state"], "candidate");
        assert_eq!(provenance["effective_at_millis"], 99);
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
        assert_eq!(provenance["principal"], "Ada");
        assert_eq!(provenance["polarity"], "positive");
        assert_eq!(provenance["projection_state"], "candidate");
        assert_eq!(provenance["effective_at_millis"], 42);
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

    // ── Supersession tests ──────────────────────────────────────────────

    fn card_with(
        id: u64,
        entity: &str,
        slot: &str,
        version_key: Option<&str>,
        source_frame_id: u64,
        event_date: Option<i64>,
    ) -> MemoryCard {
        MemoryCard {
            id,
            kind: MemoryKind::Preference,
            entity: entity.into(),
            slot: slot.into(),
            value: "v".into(),
            polarity: None,
            event_date,
            document_date: None,
            version_key: version_key.map(str::to_string),
            version_relation: VersionRelation::Updates,
            source_frame_id,
            source_uri: None,
            source_offset: None,
            engine: "test".into(),
            engine_version: "1".into(),
            confidence: None,
            // `effective_timestamp` falls back to created_at when no
            // event/document_date is set, so align it with the frame
            // so newer-frame cards naturally rank as newer recency.
            created_at: source_frame_id.try_into().unwrap_or(i64::MAX),
        }
    }

    #[test]
    fn supersede_keeps_all_items_when_none_carry_version_key() {
        // Search hits never carry version_key → no supersession occurs.
        let items =
            search_hits_to_context_items(&[make_hit(0, Some(0.9), 1), make_hit(1, Some(0.4), 2)]);
        let pack = supersede(items.clone());
        assert_eq!(pack.kept.len(), 2);
        assert!(pack.superseded.is_empty());
        let kept_items = pack.into_context_items();
        assert_eq!(kept_items[0].source_id, items[0].source_id);
        assert_eq!(kept_items[1].source_id, items[1].source_id);
    }

    #[test]
    fn supersede_collapses_same_version_key_picking_newest_frame() {
        // Older card (lower frame_id, no event_date) ranked first;
        // newer card same version_key ranked second. Newer must win.
        let older = card_with(1, "Ada", "drink", Some("Ada:drink"), 10, None);
        let newer = card_with(2, "Ada", "drink", Some("Ada:drink"), 25, None);
        let cards = vec![older, newer];
        let items = memory_cards_to_context_items(&cards);
        let pack = supersede(items);
        assert_eq!(pack.kept.len(), 1);
        assert_eq!(pack.superseded.len(), 1);
        assert_eq!(pack.kept[0].as_item().source_id, "card/2");
        assert_eq!(pack.superseded[0].survivor_source_id, "card/2");
        assert_eq!(pack.superseded[0].version_key, "Ada:drink");
        assert_eq!(pack.superseded[0].hidden.as_item().source_id, "card/1");
    }

    #[test]
    fn supersede_emits_survivor_at_first_occurrence_position() {
        // Mix: hit (no key), older card (key), unrelated hit (no key),
        // newer card (same key). Output order should be:
        //   [hit#0, newer-card, hit#1]  — newer card emitted at the
        // position of the older card's first occurrence.
        let hit0_items = search_hits_to_context_items(&[make_hit(0, Some(0.9), 100)]);
        let cards_old = memory_cards_to_context_items(&[card_with(
            1,
            "Ada",
            "drink",
            Some("Ada:drink"),
            10,
            None,
        )]);
        let hit1_items = search_hits_to_context_items(&[make_hit(0, Some(0.4), 200)]);
        let cards_new = memory_cards_to_context_items(&[card_with(
            2,
            "Ada",
            "drink",
            Some("Ada:drink"),
            25,
            None,
        )]);
        let mut items = Vec::new();
        items.extend(hit0_items);
        items.extend(cards_old);
        items.extend(hit1_items);
        items.extend(cards_new);

        let pack = supersede(items);
        assert_eq!(pack.kept.len(), 3);
        assert_eq!(pack.kept[0].as_item().source_id, "100");
        assert_eq!(pack.kept[1].as_item().source_id, "card/2");
        assert_eq!(pack.kept[2].as_item().source_id, "200");
        assert_eq!(pack.superseded.len(), 1);
        assert_eq!(pack.superseded[0].hidden.as_item().source_id, "card/1");
    }

    #[test]
    fn supersede_breaks_recency_ties_by_first_occurrence() {
        // Two cards same version_key, same effective_timestamp (both
        // event_date=None, same source_frame_id). Earlier wins.
        let first = card_with(1, "Bob", "city", Some("Bob:city"), 5, None);
        let second = card_with(2, "Bob", "city", Some("Bob:city"), 5, None);
        let items = memory_cards_to_context_items(&[first, second]);
        let pack = supersede(items);
        assert_eq!(pack.kept.len(), 1);
        assert_eq!(pack.kept[0].as_item().source_id, "card/1");
        assert_eq!(pack.superseded.len(), 1);
        assert_eq!(pack.superseded[0].hidden.as_item().source_id, "card/2");
    }

    #[test]
    fn supersede_uses_event_date_when_available() {
        // Same version_key but the older-frame card has a newer
        // event_date → event_date wins because effective_timestamp
        // prefers event_date over document_date.
        let recent_frame_old_event = card_with(
            1,
            "Cleo",
            "role",
            Some("Cleo:role"),
            /* source_frame_id */ 50,
            Some(100),
        );
        let old_frame_recent_event = card_with(
            2,
            "Cleo",
            "role",
            Some("Cleo:role"),
            /* source_frame_id */ 10,
            Some(500),
        );
        let items =
            memory_cards_to_context_items(&[recent_frame_old_event, old_frame_recent_event]);
        let pack = supersede(items);
        assert_eq!(pack.kept.len(), 1);
        assert_eq!(pack.kept[0].as_item().source_id, "card/2");
    }

    #[test]
    fn supersede_handles_groups_independently() {
        let a_old = card_with(1, "Ada", "drink", Some("Ada:drink"), 5, None);
        let b_old = card_with(2, "Bob", "city", Some("Bob:city"), 8, None);
        let a_new = card_with(3, "Ada", "drink", Some("Ada:drink"), 25, None);
        let b_new = card_with(4, "Bob", "city", Some("Bob:city"), 30, None);
        let items = memory_cards_to_context_items(&[a_old, b_old, a_new, b_new]);
        let pack = supersede(items);
        assert_eq!(pack.kept.len(), 2);
        assert_eq!(pack.kept[0].as_item().source_id, "card/3"); // Ada survivor at Ada's first pos
        assert_eq!(pack.kept[1].as_item().source_id, "card/4"); // Bob survivor at Bob's first pos
        assert_eq!(pack.superseded.len(), 2);
    }

    #[test]
    fn memory_candidate_exposes_version_key_and_recency() {
        let card = card_with(1, "Ada", "drink", Some("Ada:drink"), 42, Some(900));
        let item = card.to_context_item(0);
        let candidate = MemoryCandidate::from_item(item);
        assert_eq!(candidate.version_key(), Some("Ada:drink"));
        assert_eq!(candidate.recency(), Some(900));

        // SearchHit-projected items have no version_key → None.
        let hit_item = make_hit(0, Some(0.5), 7).to_context_item(0);
        let hit_candidate = MemoryCandidate::from_item(hit_item);
        assert_eq!(hit_candidate.version_key(), None);
    }

    // ── Frame-typed projection tests (compaction-aware) ─────────────────

    #[cfg(feature = "compaction")]
    mod typed {
        use super::super::*;
        use crate::metadata::{FrameKind, MemvidFrameMetadata};
        use memvid_core::SearchHitMetadata;
        use memvid_core::types::FrameId;

        fn envelope(kind: FrameKind, dedup_key: &str, role: &str) -> MemvidFrameMetadata {
            MemvidFrameMetadata {
                schema_version: 1,
                kind,
                conversation_id: "conv-1".into(),
                chat_role: role.into(),
                dedup_key: dedup_key.into(),
                scope: Some("project-x".into()),
            }
        }

        fn typed_hit(
            rank: usize,
            score: Option<f32>,
            frame: FrameId,
            env: Option<MemvidFrameMetadata>,
        ) -> SearchHit {
            let metadata = env.map(|env| {
                let extra = env.into_map();
                SearchHitMetadata {
                    extra_metadata: extra,
                    ..SearchHitMetadata::default()
                }
            });
            SearchHit {
                rank,
                frame_id: frame,
                uri: format!("memvid://frame/{frame}"),
                title: Some("frame".into()),
                range: (0, 10),
                text: "compacted body".into(),
                matches: 1,
                chunk_range: Some((0, 10)),
                chunk_text: None,
                score,
                metadata,
            }
        }

        #[test]
        fn frame_role_defaults_to_raw_without_envelope() {
            let hit = typed_hit(0, Some(0.5), 1, None);
            assert_eq!(frame_role(&hit), MemoryFrameRole::Raw);
        }

        #[test]
        fn frame_role_decodes_demoted_and_summary() {
            let demoted = typed_hit(
                0,
                Some(0.4),
                2,
                Some(envelope(FrameKind::DemotedMessage, "k1", "user")),
            );
            assert_eq!(frame_role(&demoted), MemoryFrameRole::DemotedMessage);

            let summary = typed_hit(
                0,
                Some(0.6),
                3,
                Some(envelope(FrameKind::CompactionSummary, "k2", "assistant")),
            );
            assert_eq!(frame_role(&summary), MemoryFrameRole::CompactionSummary);
        }

        #[test]
        fn typed_item_uses_role_specific_source_id_and_provenance() {
            let summary = typed_hit(
                0,
                Some(0.9),
                42,
                Some(envelope(FrameKind::CompactionSummary, "abc", "assistant")),
            );
            let item = typed_search_hit_to_context_item(&summary, 0);
            assert!(matches!(item.source, ContextSourceKind::Memory));
            assert_eq!(item.source_id, "summary/42");
            let provenance = item.provenance.as_object().unwrap();
            assert_eq!(provenance["resource"], "memvid.summary");
            assert_eq!(provenance["frame_kind"], "compaction_summary");
            assert_eq!(provenance["conversation_id"], "conv-1");
            assert_eq!(provenance["chat_role"], "assistant");
            assert_eq!(provenance["principal"], "assistant");
            assert_eq!(provenance["dedup_key"], "abc");
            assert_eq!(provenance["scope"], "project-x");
            assert_eq!(provenance["version_key"], "compaction_summary:abc");
            assert_eq!(provenance["effective_at_millis"], 42);
            assert_eq!(provenance["projection_state"], "candidate");

            let demoted = typed_hit(
                0,
                Some(0.3),
                7,
                Some(envelope(FrameKind::DemotedMessage, "xyz", "user")),
            );
            let item = typed_search_hit_to_context_item(&demoted, 0);
            assert_eq!(item.source_id, "frame/7");
            let provenance = item.provenance.as_object().unwrap();
            assert_eq!(provenance["resource"], "memvid.frame");
            assert_eq!(provenance["frame_kind"], "demoted_message");
            assert_eq!(provenance["version_key"], "demoted_message:xyz");
        }

        #[test]
        fn typed_item_falls_back_to_untyped_when_envelope_missing() {
            let hit = typed_hit(0, Some(0.5), 9, None);
            let item = typed_search_hit_to_context_item(&hit, 0);
            assert_eq!(item.source_id, "9");
            let provenance = item.provenance.as_object().unwrap();
            assert_eq!(provenance["resource"], "memvid.search");
            assert!(provenance.get("frame_kind").is_none());
            assert!(provenance.get("version_key").is_none());
        }

        #[test]
        fn partition_buckets_by_role_in_input_order() {
            let hits = vec![
                typed_hit(0, Some(0.9), 1, None),
                typed_hit(
                    1,
                    Some(0.5),
                    2,
                    Some(envelope(FrameKind::DemotedMessage, "d", "user")),
                ),
                typed_hit(
                    2,
                    Some(0.4),
                    3,
                    Some(envelope(FrameKind::CompactionSummary, "s", "assistant")),
                ),
                typed_hit(3, Some(0.2), 4, None),
            ];
            let parts = partition_search_hits_by_role(&hits);
            assert_eq!(parts.raw.len(), 2);
            assert_eq!(parts.demoted.len(), 1);
            assert_eq!(parts.summaries.len(), 1);
            assert_eq!(parts.raw[0].frame_id, 1);
            assert_eq!(parts.raw[1].frame_id, 4);
            assert_eq!(parts.demoted[0].frame_id, 2);
            assert_eq!(parts.summaries[0].frame_id, 3);
        }

        #[test]
        fn summary_pack_collapses_resummarized_dedup_key_to_newest_frame() {
            // Same dedup_key written twice (e.g. compactor re-rolled the
            // same chunk) → newer frame must survive, older suppressed.
            let older = typed_hit(
                0,
                Some(0.5),
                10,
                Some(envelope(
                    FrameKind::CompactionSummary,
                    "chunk-1",
                    "assistant",
                )),
            );
            let newer = typed_hit(
                1,
                Some(0.5),
                25,
                Some(envelope(
                    FrameKind::CompactionSummary,
                    "chunk-1",
                    "assistant",
                )),
            );
            let pack = MemoryContextPack::from_search_hits(&[older, newer]);
            assert_eq!(pack.kept.len(), 1);
            assert_eq!(pack.kept[0].as_item().source_id, "summary/25");
            assert_eq!(pack.superseded.len(), 1);
            assert_eq!(pack.superseded[0].version_key, "compaction_summary:chunk-1");
            assert_eq!(pack.superseded[0].hidden.as_item().source_id, "summary/10");
            assert_eq!(pack.superseded[0].survivor_source_id, "summary/25");
        }

        #[test]
        fn raw_and_summary_with_same_dedup_key_do_not_collide() {
            // A raw demoted_message and a compaction_summary that share
            // a dedup_key (would be unusual but possible if the dedup
            // hashes lined up) must NOT collapse — role is part of the
            // supersession key.
            let demoted = typed_hit(
                0,
                Some(0.4),
                5,
                Some(envelope(FrameKind::DemotedMessage, "shared", "user")),
            );
            let summary = typed_hit(
                1,
                Some(0.6),
                6,
                Some(envelope(
                    FrameKind::CompactionSummary,
                    "shared",
                    "assistant",
                )),
            );
            let pack = MemoryContextPack::from_search_hits(&[demoted, summary]);
            assert_eq!(pack.kept.len(), 2);
            assert_eq!(pack.kept[0].as_item().source_id, "frame/5");
            assert_eq!(pack.kept[1].as_item().source_id, "summary/6");
            assert!(pack.superseded.is_empty());
        }

        #[test]
        fn typed_hits_to_candidates_preserves_input_order() {
            let hits = vec![
                typed_hit(
                    0,
                    Some(0.9),
                    1,
                    Some(envelope(FrameKind::DemotedMessage, "a", "user")),
                ),
                typed_hit(
                    1,
                    Some(0.5),
                    2,
                    Some(envelope(FrameKind::CompactionSummary, "b", "assistant")),
                ),
            ];
            let candidates = typed_search_hits_to_memory_candidates(&hits);
            assert_eq!(candidates.len(), 2);
            assert_eq!(candidates[0].as_item().source_id, "frame/1");
            assert_eq!(candidates[1].as_item().source_id, "summary/2");
            assert_eq!(candidates[0].version_key(), Some("demoted_message:a"));
            assert_eq!(candidates[1].version_key(), Some("compaction_summary:b"));
        }
    }
}
