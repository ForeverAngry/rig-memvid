//! [`MemoryCardContext`]: a [`VectorStoreIndex`] view over a
//! [`MemvidStore`]'s structured-memory track.
//!
//! Where [`MemvidStore`] returns raw frame text (BM25/vector hits over the
//! conversation), `MemoryCardContext` returns formatted
//! [`memvid_core::MemoryCard`]s — the entity / slot / value triples that
//! memvid extracts automatically from each frame. Wire it as a second
//! `dynamic_context` so the agent sees both episodic and structured
//! recall:
//!
//! ```rust,no_run
//! use rig_memvid::{CardSelection, MemoryCardContext, MemvidStore};
//!
//! # async fn run(store: MemvidStore) -> anyhow::Result<()> {
//! # let model: rig::providers::openai::CompletionModel = unimplemented!();
//! let cards = MemoryCardContext::new(store.clone(), CardSelection::EntityMentions);
//! let agent = rig::agent::AgentBuilder::new(model)
//!     .dynamic_context(4, store)   // episodic frames
//!     .dynamic_context(8, cards)   // structured cards
//!     .build();
//! # Ok(()) }
//! ```
//!
//! No agent cooperation is required: card selection runs purely against
//! the local store, so this works the same on small open-weight models
//! as on tool-using frontier models.

use memvid_core::{MemoryCard, MemoryKind, Polarity};
use rig::vector_store::{VectorSearchRequest, VectorStoreError, VectorStoreIndex};
use rig::wasm_compat::WasmCompatSend;
use serde::{Deserialize, Serialize};

use crate::error::MemvidError;
use crate::memory_graph::MemoryGraph;
use crate::store::{MemvidFilter, MemvidStore};

/// Strategy for choosing which memory cards to surface as context.
///
/// Each variant resolves a query into a list of cards using only the
/// local store — no LLM, no NER. Build your own [`MemoryCardContext`]
/// implementation if you need fancier extraction.
#[derive(Debug, Clone, Default)]
pub enum CardSelection {
    /// Pull cards for entities whose names appear (case-insensitive,
    /// word-boundary-aware) in the query string. Cheap, deterministic,
    /// and zero-dependency. The default and the right choice for most
    /// chatbots.
    #[default]
    EntityMentions,
    /// Always include the most recently-written cards, ignoring the
    /// query entirely. Useful as a "what does the agent know about the
    /// user right now" preamble that doesn't depend on lexical overlap.
    RecentCards,
    /// Always include cards for the named principal/entity, ignoring
    /// the query text. This is useful for personal assistants where
    /// follow-up questions often say "what should I avoid?" without
    /// repeating the user's name.
    ForPrincipal(String),
    /// Always include preference-kind cards for the listed entities.
    /// Combine with [`Self::EntityMentions`] in your own selection
    /// strategy if you need both.
    PreferencesFor(Vec<String>),
}

/// A [`VectorStoreIndex`] that returns formatted [`MemoryCard`]s instead
/// of frame text. Generic over any [`MemoryGraph`] backend; defaults to
/// [`MemvidStore`] so the common call site
/// `MemoryCardContext::new(store, _)` continues to work without naming
/// the type parameter.
///
/// Returned scores combine deterministic query/card relevance with
/// recency as a tie-breaker. They are intended for stable context
/// ordering, not as embedding similarity scores.
#[derive(Debug, Clone)]
pub struct MemoryCardContext<G = MemvidStore>
where
    G: MemoryGraph,
{
    graph: G,
    strategy: CardSelection,
    max_cards: usize,
}

impl<G> MemoryCardContext<G>
where
    G: MemoryGraph,
{
    /// Default cap on cards returned per query when the caller doesn't
    /// override via [`Self::with_max_cards`] or
    /// [`VectorSearchRequest::samples`].
    pub const DEFAULT_MAX_CARDS: usize = 8;

    /// Build a context view over `graph` using `strategy`.
    pub fn new(graph: G, strategy: CardSelection) -> Self {
        Self {
            graph,
            strategy,
            max_cards: Self::DEFAULT_MAX_CARDS,
        }
    }

    /// Cap on the number of cards returned per query. The effective cap
    /// is `min(max_cards, request.samples())`.
    #[must_use]
    pub fn with_max_cards(mut self, max_cards: usize) -> Self {
        self.max_cards = max_cards;
        self
    }

    /// Borrow the underlying graph backend.
    #[must_use]
    pub fn graph(&self) -> &G {
        &self.graph
    }

    /// Selection strategy currently in use.
    #[must_use]
    pub fn strategy(&self) -> &CardSelection {
        &self.strategy
    }

    /// Resolve `query` into the set of candidate cards according to the
    /// configured [`CardSelection`]. Public so tests and tools can
    /// inspect what the agent would see.
    pub fn select(&self, query: &str) -> Result<Vec<MemoryCard>, G::Error> {
        match &self.strategy {
            CardSelection::EntityMentions => self.select_entity_mentions(query),
            CardSelection::RecentCards => self.select_recent(),
            CardSelection::ForPrincipal(principal) => self.select_for_principal(principal),
            CardSelection::PreferencesFor(entities) => self.select_preferences(entities),
        }
    }

    fn select_entity_mentions(&self, query: &str) -> Result<Vec<MemoryCard>, G::Error> {
        if self.graph.memory_card_count()? == 0 {
            return Ok(Vec::new());
        }
        let needle = query.to_lowercase();

        // Snapshot every card once, then filter in pure Rust. Cards are
        // already in memory inside the backend; this avoids any
        // re-entrant per-entity reader calls.
        let all = self.graph.all_memory_cards()?;
        let mut hits: Vec<MemoryCard> = all
            .into_iter()
            .filter(|card| {
                let entity = card.entity.to_lowercase();
                if entity.is_empty() {
                    return false;
                }
                contains_word(&needle, &entity)
            })
            .collect();
        hits.sort_by_key(|c| std::cmp::Reverse(c.created_at));
        Ok(hits)
    }

    fn select_recent(&self) -> Result<Vec<MemoryCard>, G::Error> {
        if self.graph.memory_card_count()? == 0 {
            return Ok(Vec::new());
        }
        let mut all = self.graph.all_memory_cards()?;
        all.sort_by_key(|c| std::cmp::Reverse(c.created_at));
        Ok(all)
    }

    fn select_for_principal(&self, principal: &str) -> Result<Vec<MemoryCard>, G::Error> {
        if self.graph.memory_card_count()? == 0 {
            return Ok(Vec::new());
        }
        let mut hits = self.graph.entity_memories(principal)?;
        let lower = principal.to_lowercase();
        if hits.is_empty() && lower != principal {
            hits = self.graph.entity_memories(&lower)?;
        }
        for card in self.graph.all_memory_cards()? {
            if hits.iter().any(|existing| same_card(existing, &card)) {
                continue;
            }
            let entity = card.entity.to_lowercase();
            if contains_word(&entity, &lower) {
                hits.push(card);
            }
        }

        let related_entities: Vec<String> = hits
            .iter()
            .filter(|card| card.kind == MemoryKind::Relationship)
            .map(|card| card.value.clone())
            .collect();
        for entity in related_entities {
            for card in self.related_entity_memories(&entity)? {
                if hits.iter().any(|existing| same_card(existing, &card)) {
                    continue;
                }
                hits.push(card);
            }
        }

        hits.sort_by_key(|c| std::cmp::Reverse(c.created_at));
        Ok(hits)
    }

    fn related_entity_memories(&self, entity: &str) -> Result<Vec<MemoryCard>, G::Error> {
        let mut hits = self.graph.entity_memories(entity)?;
        let lower = entity.to_lowercase();
        if hits.is_empty() && lower != entity {
            hits = self.graph.entity_memories(&lower)?;
        }
        Ok(hits)
    }

    fn select_preferences(&self, entities: &[String]) -> Result<Vec<MemoryCard>, G::Error> {
        if self.graph.memory_card_count()? == 0 {
            return Ok(Vec::new());
        }
        let mut hits = Vec::new();
        for ent in entities {
            hits.extend(self.graph.entity_preferences(ent)?);
        }
        hits.sort_by_key(|c| std::cmp::Reverse(c.created_at));
        Ok(hits)
    }
}
fn same_card(left: &MemoryCard, right: &MemoryCard) -> bool {
    left.entity == right.entity
        && left.slot == right.slot
        && left.value == right.value
        && left.source_frame_id == right.source_frame_id
}

/// Case-insensitive word-boundary substring match.
///
/// Avoids matching `"art"` inside `"smart"` while staying dependency-free.
/// Both inputs must already be lowercased by the caller.
fn contains_word(haystack: &str, needle: &str) -> bool {
    if needle.is_empty() || haystack.len() < needle.len() {
        return false;
    }
    let bytes = haystack.as_bytes();
    let n = needle.as_bytes();
    let mut i = 0usize;
    while i + n.len() <= bytes.len() {
        let Some(window) = bytes.get(i..i + n.len()) else {
            break;
        };
        if window == n {
            let before_ok = match i.checked_sub(1).and_then(|j| bytes.get(j)) {
                None => true,
                Some(b) => !is_word_byte(*b),
            };
            let after_ok = match bytes.get(i + n.len()) {
                None => true,
                Some(b) => !is_word_byte(*b),
            };
            if before_ok && after_ok {
                return true;
            }
        }
        i += 1;
    }
    false
}

fn is_word_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Render a card as a single compact line for context injection.
///
/// Format: `<kind> <entity>/<slot> = <value>[ (<polarity>)]`. Kept short
/// on purpose: dense structured facts cost more in the model's context
/// budget than free-text frames.
fn format_card(card: &MemoryCard) -> String {
    let polarity = match card.polarity {
        Some(Polarity::Positive) => " (+)",
        Some(Polarity::Negative) => " (-)",
        Some(Polarity::Neutral) | None => "",
    };
    if card.kind == MemoryKind::Relationship {
        if card.slot == "reports_to" {
            return format!(
                "rel {entity}'s manager = {value}",
                entity = card.entity,
                value = card.value
            );
        }
        if card.slot == "manager" {
            return format!(
                "rel {entity}'s manager = {value}",
                entity = card.entity,
                value = card.value
            );
        }
    }
    if card.kind == MemoryKind::Fact && card.slot == "location" {
        return format!(
            "fact {entity} lives in {value}",
            entity = card.entity,
            value = card.value,
        );
    }
    if card.kind == MemoryKind::Fact && card.slot == "employer" {
        return format!(
            "fact {entity} works at {value}",
            entity = card.entity,
            value = card.value,
        );
    }
    if card.kind == MemoryKind::Profile && card.slot == "allergy" {
        return format!(
            "profile {entity} is allergic to {value}",
            entity = card.entity,
            value = card.value,
        );
    }
    if card.kind == MemoryKind::Preference {
        if card.polarity == Some(Polarity::Negative) {
            return format!(
                "pref {entity} dislikes {value}",
                entity = card.entity,
                value = card.value,
            );
        }
        if card.polarity == Some(Polarity::Positive) {
            return format!(
                "pref {entity} likes {value}",
                entity = card.entity,
                value = card.value,
            );
        }
    }
    if matches!(card.kind, MemoryKind::Fact | MemoryKind::Profile) {
        return format!(
            "{kind} {entity}'s {slot} = {value}",
            kind = kind_str(card.kind),
            entity = card.entity,
            slot = card.slot,
            value = card.value,
        );
    }
    format!(
        "{kind} {entity}/{slot} = {value}{polarity}",
        kind = kind_str(card.kind),
        entity = card.entity,
        slot = card.slot,
        value = card.value,
        polarity = polarity,
    )
}

fn kind_str(kind: MemoryKind) -> &'static str {
    match kind {
        MemoryKind::Fact => "fact",
        MemoryKind::Preference => "pref",
        MemoryKind::Event => "event",
        MemoryKind::Profile => "profile",
        MemoryKind::Relationship => "rel",
        MemoryKind::Goal => "goal",
        MemoryKind::Other => "other",
    }
}

/// Synthetic score: 1.0 for the newest card, decreasing linearly to 0.0
/// for the oldest in this batch. Stable under empty / single-element
/// inputs.
fn recency_scores(cards: &[MemoryCard]) -> Vec<f64> {
    let n = cards.len();
    if n <= 1 {
        return vec![1.0; n];
    }
    let denom = (n - 1) as f64;
    (0..n).map(|i| 1.0 - (i as f64 / denom)).collect()
}

fn rank_cards(query: &str, cards: Vec<MemoryCard>) -> Vec<(f64, MemoryCard)> {
    let query = query.to_lowercase();
    let recency = recency_scores(&cards);
    let mut ranked: Vec<(f64, MemoryCard)> = cards
        .into_iter()
        .zip(recency)
        .map(|(card, recency_score)| {
            let score = card_relevance_score(&query, &card) + recency_score * 0.01;
            (score, card)
        })
        .collect();
    ranked.sort_by(|left, right| {
        right
            .0
            .total_cmp(&left.0)
            .then_with(|| right.1.created_at.cmp(&left.1.created_at))
    });
    ranked
}

fn card_relevance_score(query: &str, card: &MemoryCard) -> f64 {
    let mut score = 0.0;
    let entity = card.entity.to_lowercase();
    let slot = card.slot.to_lowercase();
    let value = card.value.to_lowercase();

    let entity_matches = !entity.is_empty() && contains_word(query, &entity);
    if entity_matches {
        score += 5.0;
        if card.kind == MemoryKind::Relationship
            && query_matches(
                query,
                &["manager", "boss", "reports", "report", "relationship"],
            )
        {
            score += 4.0;
        }
    }
    if !slot.is_empty() && contains_word(query, &slot) {
        score += 4.0;
    }
    if !value.is_empty() && contains_word(query, &value) {
        score += 2.0;
    }

    score += slot_intent_score(query, &slot);
    score += kind_intent_score(query, card.kind);

    if query_terms_match(query, &slot) {
        score += 1.0;
    }
    if query_terms_match(query, &value) {
        score += 1.0;
    }

    score
}

fn slot_intent_score(query: &str, slot: &str) -> f64 {
    if slot_matches(slot, &["location", "city", "home", "address"])
        && query_matches(
            query,
            &[
                "where", "live", "lives", "located", "location", "city", "reside", "resides",
                "from", "grew",
            ],
        )
    {
        return 6.0;
    }
    if slot_matches(slot, &["allergy", "allergic", "avoidance"])
        && query_matches(
            query,
            &[
                "avoid", "serve", "food", "allergic", "allergy", "eat", "cannot", "can't", "safe",
            ],
        )
    {
        return 6.0;
    }
    if slot_matches(slot, &["preference", "drink", "food", "coffee"])
        && query_matches(
            query,
            &[
                "like",
                "likes",
                "prefer",
                "prefers",
                "preference",
                "preferences",
                "drink",
                "coffee",
                "dislike",
                "dislikes",
            ],
        )
    {
        return 6.0;
    }
    if slot_matches(slot, &["manager", "reports_to", "reports", "boss"])
        && query_matches(
            query,
            &["manager", "boss", "reports", "report", "supervisor"],
        )
    {
        return 6.0;
    }
    if slot_matches(slot, &["employer", "company", "work"])
        && query_matches(
            query,
            &["work", "works", "employer", "company", "job", "role"],
        )
    {
        return 6.0;
    }
    0.0
}

fn kind_intent_score(query: &str, kind: MemoryKind) -> f64 {
    match kind {
        MemoryKind::Preference
            if query_matches(
                query,
                &[
                    "like",
                    "likes",
                    "prefer",
                    "prefers",
                    "preference",
                    "preferences",
                    "dislike",
                    "dislikes",
                ],
            ) =>
        {
            2.0
        }
        MemoryKind::Profile
            if query_matches(
                query,
                &[
                    "allergic", "allergy", "avoid", "serve", "food", "profile", "about",
                ],
            ) =>
        {
            2.0
        }
        MemoryKind::Relationship
            if query_matches(
                query,
                &["manager", "boss", "reports", "report", "relationship"],
            ) =>
        {
            2.0
        }
        MemoryKind::Fact => 0.5,
        MemoryKind::Event | MemoryKind::Goal | MemoryKind::Other | MemoryKind::Profile => 0.0,
        MemoryKind::Preference | MemoryKind::Relationship => 0.0,
    }
}

fn slot_matches(slot: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| contains_word(slot, needle))
}

fn query_matches(query: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| contains_word(query, needle))
}

fn query_terms_match(query: &str, text: &str) -> bool {
    text.split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|term| term.len() > 2)
        .any(|term| contains_word(query, term))
}

impl<G> VectorStoreIndex for MemoryCardContext<G>
where
    G: MemoryGraph + WasmCompatSend + Sync,
{
    /// Card-context selection ignores filters; using [`MemvidFilter`]
    /// keeps the type aligned with [`MemvidStore`] so callers can stack
    /// both views under a single filter type when composing
    /// `dynamic_context`.
    type Filter = MemvidFilter;

    async fn top_n<T>(
        &self,
        req: VectorSearchRequest<Self::Filter>,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError>
    where
        T: for<'a> Deserialize<'a> + WasmCompatSend,
    {
        let query = req.query().to_owned();
        let limit = std::cmp::min(self.max_cards, req.samples() as usize);

        let mut ranked = rank_cards(&query, self.select(&query).map_err(Into::into)?);
        if ranked.len() > limit {
            ranked.truncate(limit);
        }

        let mut out = Vec::with_capacity(ranked.len());
        for (score, card) in ranked {
            let id = card.id.to_string();
            let payload = CardDoc {
                text: format_card(&card),
                kind: kind_str(card.kind).to_string(),
                entity: card.entity,
                slot: card.slot,
                value: card.value,
                polarity: card.polarity.map(polarity_str).map(str::to_owned),
                source_frame_id: card.source_frame_id,
                confidence: card.confidence,
            };
            let value = serde_json::to_value(&payload).map_err(MemvidError::from)?;
            let doc: T = serde_json::from_value(value).map_err(MemvidError::from)?;
            out.push((score, id, doc));
        }
        Ok(out)
    }

    async fn top_n_ids(
        &self,
        req: VectorSearchRequest<Self::Filter>,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        let query = req.query().to_owned();
        let limit = std::cmp::min(self.max_cards, req.samples() as usize);

        let mut ranked = rank_cards(&query, self.select(&query).map_err(Into::into)?);
        if ranked.len() > limit {
            ranked.truncate(limit);
        }
        Ok(ranked
            .into_iter()
            .map(|(score, card)| (score, card.id.to_string()))
            .collect())
    }
}

fn polarity_str(p: Polarity) -> &'static str {
    match p {
        Polarity::Positive => "positive",
        Polarity::Negative => "negative",
        Polarity::Neutral => "neutral",
    }
}

/// Wire-format for a single card as projected into agent context.
///
/// `text` is the human-readable line that
/// [`rig::agent::AgentBuilder::dynamic_context`] surfaces in the prompt;
/// the remaining fields are present for callers that deserialise into a
/// richer struct (e.g. tools that want the polarity or source frame id).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CardDoc {
    /// Compact one-line rendering of the card.
    pub text: String,
    /// Memory kind as a short string (`fact`, `pref`, `event`, …).
    pub kind: String,
    /// Subject of the SPO triple.
    pub entity: String,
    /// Predicate / attribute name.
    pub slot: String,
    /// Object / value as a string.
    pub value: String,
    /// `"positive"` / `"negative"` / `"neutral"` if recorded.
    pub polarity: Option<String>,
    /// Frame id this card was extracted from.
    pub source_frame_id: u64,
    /// Extractor confidence in `[0, 1]` if the engine reported one.
    pub confidence: Option<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn word_boundary_matches() {
        assert!(contains_word("alice loves rust", "alice"));
        assert!(contains_word("hi alice!", "alice"));
        assert!(contains_word("alice", "alice"));
        assert!(!contains_word("smart cookie", "art"));
        assert!(!contains_word("alicemarie", "alice"));
        assert!(!contains_word("", "alice"));
        assert!(!contains_word("alice", ""));
    }

    #[test]
    fn recency_scores_handles_edge_cases() {
        assert_eq!(recency_scores(&[]), Vec::<f64>::new());
        assert_eq!(recency_scores(&[stub_card("a")]), vec![1.0]);
        let two = recency_scores(&[stub_card("a"), stub_card("b")]);
        assert_eq!(two, vec![1.0, 0.0]);
    }

    fn stub_card(entity: &str) -> MemoryCard {
        MemoryCard {
            id: 0,
            kind: MemoryKind::Fact,
            entity: entity.into(),
            slot: "s".into(),
            value: "v".into(),
            polarity: None,
            event_date: None,
            document_date: None,
            version_key: None,
            version_relation: memvid_core::VersionRelation::default(),
            source_frame_id: 0,
            source_uri: None,
            source_offset: None,
            engine: "t".into(),
            engine_version: "0".into(),
            confidence: None,
            created_at: 0,
        }
    }
}
