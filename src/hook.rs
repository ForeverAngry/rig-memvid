//! [`MemvidPersistHook`]: a [`PromptHook`] that persists every turn of an
//! agent conversation into a [`MemvidStore`].

use std::marker::PhantomData;
use std::sync::Arc;

use memvid_core::{MemoryCard, MemoryCardBuilder, PutOptions};
use rig::{
    agent::{HookAction, PromptHook},
    completion::{CompletionModel, CompletionResponse, Message},
};

use crate::store::MemvidStore;

/// A function that decides what (if anything) to persist for a single
/// message. Returning `None` skips the message.
///
/// Returning `Some("")` is treated identically to `None`: empty payloads
/// are never written to the archive.
pub type WriteTransform = Arc<dyn Fn(&Message) -> Option<String> + Send + Sync + 'static>;

/// Strategy for what to write into the memvid archive on each turn.
#[derive(Clone, Default)]
pub enum WritePolicy {
    /// Do not persist anything. The hook becomes a no-op (useful for toggling
    /// memory at runtime without removing the hook).
    Disabled,
    /// Persist the verbatim text of every user prompt and assistant response.
    #[default]
    Raw,
    /// Apply the supplied transform to each message and persist its result
    /// (or nothing, if the transform returns `None`).
    ///
    /// This is the extension point for caller-defined summarisation, PII
    /// redaction, or selective filtering.
    Custom(WriteTransform),
}

impl std::fmt::Debug for WritePolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Disabled => f.write_str("WritePolicy::Disabled"),
            Self::Raw => f.write_str("WritePolicy::Raw"),
            Self::Custom(_) => f.write_str("WritePolicy::Custom(<fn>)"),
        }
    }
}

/// Configuration for [`MemvidPersistHook`].
#[derive(Clone, Debug)]
pub struct MemoryConfig {
    /// What to persist on each turn.
    pub policy: WritePolicy,
    /// If `true`, call `commit()` after every turn so the new frames are
    /// immediately searchable. If `false`, the caller is responsible for
    /// committing periodically.
    pub commit_each_turn: bool,
    /// Tags applied to every persisted frame, useful for later filtering.
    pub default_tags: Vec<String>,
    /// Logical scope written into the frame's URI prefix. When set, every
    /// frame produced by this hook is stored with `PutOptions.uri = Some(scope)`,
    /// which makes `MemvidFilter::eq("scope", scope)` match those
    /// frames at query time (memvid's `scope` is a URI prefix filter).
    pub scope: Option<String>,
    /// Stable identity for the human side of the conversation.
    ///
    /// When set, user turns are lightly rewritten before memvid sees
    /// them so first-person pronouns resolve to this principal. For
    /// example, with `principal = Some("Alice".into())`, `I like
    /// espresso` is persisted as `Alice likes espresso`. This improves
    /// memvid's entity / slot / value extraction without requiring an
    /// LLM or a new runtime dependency.
    pub principal: Option<String>,
    /// If `true`, persist assistant responses as well as user turns.
    ///
    /// Defaults to `true` to preserve the full conversation transcript.
    /// Set to `false` when the archive is primarily used for user profile
    /// memory and assistant paraphrases would add noisy duplicate cards.
    pub persist_assistant: bool,
    /// Add small deterministic cards for principal-bound user turns when
    /// memvid's built-in triplet extractor misses common user-profile or
    /// relationship facts.
    ///
    /// Currently covers allergy / avoidance statements and simple
    /// manager / reporting statements after [`Self::principal`] has
    /// bound first-person pronouns to the stable entity. Defaults to
    /// `true`; it is a no-op when `principal` is `None`.
    pub supplemental_profile_cards: bool,
    /// Run memvid's auto-tagger over each persisted frame to attach
    /// extracted entity / topic tags. Defaults to `true`, mirroring
    /// [`memvid_core::PutOptions::default`].
    pub auto_tag: bool,
    /// Run memvid's date extractor over each persisted frame so dates
    /// mentioned in conversation become queryable. Defaults to `true`.
    pub extract_dates: bool,
    /// Extract Subject-Predicate-Object triplets from each persisted
    /// frame and store them as [`memvid_core::MemoryCard`]s on the
    /// memories track. Cards become queryable through
    /// [`crate::MemvidStore::entity_memories`],
    /// [`crate::MemvidStore::current_memory`],
    /// [`crate::MemvidStore::entity_preferences`], and the rest of the
    /// memory-card surface. Defaults to `true`.
    pub extract_triplets: bool,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            policy: WritePolicy::default(),
            commit_each_turn: true,
            default_tags: Vec::new(),
            scope: None,
            principal: None,
            persist_assistant: true,
            supplemental_profile_cards: true,
            auto_tag: true,
            extract_dates: true,
            extract_triplets: true,
        }
    }
}

/// Hook that records every user prompt and assistant response into a
/// [`MemvidStore`].
///
/// The hook is generic over the [`CompletionModel`] so the same store can be
/// shared between agents that use different providers.
pub struct MemvidPersistHook<M> {
    store: MemvidStore,
    config: MemoryConfig,
    _model: PhantomData<fn() -> M>,
}

impl<M> Clone for MemvidPersistHook<M> {
    fn clone(&self) -> Self {
        Self {
            store: self.store.clone(),
            config: self.config.clone(),
            _model: PhantomData,
        }
    }
}

impl<M> std::fmt::Debug for MemvidPersistHook<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemvidPersistHook")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl<M> MemvidPersistHook<M> {
    /// Create a new hook persisting into `store` according to `config`.
    pub fn new(store: MemvidStore, config: MemoryConfig) -> Self {
        Self {
            store,
            config,
            _model: PhantomData,
        }
    }

    /// Convenience: build a hook with the default [`MemoryConfig`]
    /// ([`WritePolicy::Raw`], `commit_each_turn = true`).
    pub fn with_defaults(store: MemvidStore) -> Self {
        Self::new(store, MemoryConfig::default())
    }

    fn render(&self, msg: &Message) -> Option<String> {
        match &self.config.policy {
            WritePolicy::Disabled => None,
            WritePolicy::Raw => render_message_text(msg),
            WritePolicy::Custom(f) => f(msg),
        }
    }

    fn put_options(&self, chat_role: &str) -> PutOptions {
        let mut opts = PutOptions {
            tags: self.config.default_tags.clone(),
            auto_tag: self.config.auto_tag,
            extract_dates: self.config.extract_dates,
            extract_triplets: self.config.extract_triplets,
            ..PutOptions::default()
        };
        opts.extra_metadata
            .insert("chat_role".into(), chat_role.into());
        if let Some(scope) = self.config.scope.as_deref() {
            // Memvid's `scope` search filter matches against frame URIs by
            // prefix, so attach the scope as the URI. Also stash it under
            // `extra_metadata["scope"]` for ergonomic introspection by
            // tools that walk frames directly.
            opts.uri = Some(scope.to_string());
            opts.extra_metadata.insert("scope".into(), scope.into());
        }
        opts
    }

    fn write(&self, text: &str, chat_role: &str) {
        if text.is_empty() {
            return;
        }
        let text = if chat_role == "user" {
            self.config
                .principal
                .as_deref()
                .map(|principal| bind_principal(text, principal))
                .unwrap_or_else(|| text.to_string())
        } else {
            text.to_string()
        };
        let opts = self.put_options(chat_role);
        let scope = self.config.scope.clone();
        let frame_id = match self.store.put_text_uncommitted(&text, opts) {
            Ok(frame_id) => frame_id,
            Err(err) => {
                tracing::warn!(
                    target: "rig_memvid::hook",
                    error = %err,
                    role = chat_role,
                    "failed to persist message into memvid",
                );
                return;
            }
        };

        if chat_role == "user"
            && self.config.supplemental_profile_cards
            && let Some(principal) = self.config.principal.as_deref()
        {
            for card in supplemental_memory_cards(&text, principal, frame_id, scope.clone()) {
                if let Err(err) = self.store.put_memory_card(card) {
                    tracing::warn!(
                        target: "rig_memvid::hook",
                        error = %err,
                        role = chat_role,
                        "failed to persist supplemental memory card into memvid",
                    );
                }
            }
        }

        if self.config.commit_each_turn
            && let Err(err) = self.store.commit()
        {
            tracing::warn!(
                target: "rig_memvid::hook",
                error = %err,
                role = chat_role,
                "failed to persist message into memvid",
            );
        }
    }
}

fn supplemental_memory_cards(
    text: &str,
    principal: &str,
    frame_id: u64,
    source_uri: Option<String>,
) -> Vec<MemoryCard> {
    let mut cards = Vec::new();
    if let Some(value) = allergy_value(text)
        && let Some(card) = profile_card(
            &principal.to_lowercase(),
            "allergy",
            &value,
            frame_id,
            source_uri.clone(),
        )
    {
        cards.push(card);
    }
    cards.extend(relationship_cards(text, principal, frame_id, source_uri));
    cards
}

fn profile_card(
    entity: &str,
    slot: &str,
    value: &str,
    frame_id: u64,
    source_uri: Option<String>,
) -> Option<MemoryCard> {
    MemoryCardBuilder::new()
        .profile()
        .entity(normalize_entity(entity))
        .slot(slot)
        .value(value.trim())
        .source(frame_id, source_uri)
        .engine("rig-memvid:principal-rules", "1")
        .confidence(1.0)
        .build(0)
        .ok()
}

fn relationship_card(
    entity: &str,
    slot: &str,
    value: &str,
    frame_id: u64,
    source_uri: Option<String>,
) -> Option<MemoryCard> {
    MemoryCardBuilder::new()
        .relationship()
        .entity(normalize_entity(entity))
        .slot(slot)
        .value(value.trim())
        .source(frame_id, source_uri)
        .engine("rig-memvid:principal-rules", "1")
        .confidence(1.0)
        .build(0)
        .ok()
}

fn fact_card(
    entity: &str,
    slot: &str,
    value: &str,
    frame_id: u64,
    source_uri: Option<String>,
) -> Option<MemoryCard> {
    MemoryCardBuilder::new()
        .fact()
        .entity(normalize_entity(entity))
        .slot(slot)
        .value(value.trim())
        .source(frame_id, source_uri)
        .engine("rig-memvid:principal-rules", "1")
        .confidence(1.0)
        .build(0)
        .ok()
}

fn relationship_cards(
    text: &str,
    principal: &str,
    frame_id: u64,
    source_uri: Option<String>,
) -> Vec<MemoryCard> {
    let mut cards = Vec::new();
    let Some(manager) = manager_subject(text, principal) else {
        return cards;
    };

    if let Some(card) =
        relationship_card(principal, "manager", &manager, frame_id, source_uri.clone())
    {
        cards.push(card);
    }

    if let Some(employer) = manager_employer(text, principal)
        && let Some(card) = fact_card(
            &manager,
            "employer",
            &employer,
            frame_id,
            source_uri.clone(),
        )
    {
        cards.push(card);
    }

    if let Some(report) = reports_to(text, &manager) {
        if let Some(card) = relationship_card(
            &manager,
            "reports_to",
            &report.manager,
            frame_id,
            source_uri.clone(),
        ) {
            cards.push(card);
        }
        if let Some(title) = report.manager_title
            && let Some(card) = profile_card(
                &report.manager,
                "title",
                &title,
                frame_id,
                source_uri.clone(),
            )
        {
            cards.push(card);
        }
    }

    cards
}

fn manager_subject(text: &str, principal: &str) -> Option<String> {
    let lower = text.to_lowercase();
    let marker = format!(" is {}'s manager", principal.to_lowercase());
    let idx = lower.find(&marker)?;
    let before = text.get(..idx)?.trim();
    last_name(before)
}

fn manager_employer(text: &str, principal: &str) -> Option<String> {
    let lower = text.to_lowercase();
    let marker = format!(" is {}'s manager at ", principal.to_lowercase());
    let idx = lower.find(&marker)? + marker.len();
    let raw = text.get(idx..)?;
    clean_clause(raw, &['.', '!', '?', ';', ',', '\n'])
}

struct ReportsTo {
    manager: String,
    manager_title: Option<String>,
}

fn reports_to(text: &str, subject: &str) -> Option<ReportsTo> {
    let lower = text.to_lowercase();
    let subject_marker = format!("{} reports to ", subject.to_lowercase());
    let start = if let Some(idx) = lower.find(&subject_marker) {
        idx + subject_marker.len()
    } else if let Some(idx) = lower.find(" he reports to ") {
        idx + " he reports to ".len()
    } else if let Some(idx) = lower.find(" she reports to ") {
        idx + " she reports to ".len()
    } else {
        return None;
    };
    let raw = text.get(start..)?;
    let sentence = clean_clause(raw, &['.', '!', '?', ';', '\n'])?;
    let mut parts = sentence.splitn(2, ',');
    let manager = clean_name(parts.next()?)?;
    let manager_title = parts.next().and_then(clean_title);
    Some(ReportsTo {
        manager,
        manager_title,
    })
}

fn last_name(text: &str) -> Option<String> {
    text.split_whitespace().rev().find_map(clean_name)
}

fn clean_name(text: &str) -> Option<String> {
    let value = text
        .trim()
        .trim_matches(|c: char| !c.is_alphanumeric() && c != '_' && c != '-' && c != '\'')
        .trim();
    (!value.is_empty()).then(|| value.to_string())
}

fn clean_title(text: &str) -> Option<String> {
    let value = text
        .trim()
        .strip_prefix("the ")
        .unwrap_or_else(|| text.trim())
        .trim()
        .trim_matches(|c: char| !c.is_alphanumeric() && c != ' ' && c != '_' && c != '-')
        .trim();
    (!value.is_empty()).then(|| value.to_string())
}

fn clean_clause(text: &str, delimiters: &[char]) -> Option<String> {
    let value = text
        .split(|c| delimiters.contains(&c))
        .next()?
        .trim()
        .trim_matches(|c: char| !c.is_alphanumeric() && c != ' ' && c != '_' && c != '-')
        .trim();
    (!value.is_empty()).then(|| value.to_string())
}

fn normalize_entity(entity: &str) -> String {
    entity.trim().to_lowercase()
}

fn allergy_value(text: &str) -> Option<String> {
    let lower = text.to_lowercase();
    let start = if let Some(idx) = lower.find(" allergic to ") {
        idx + " allergic to ".len()
    } else if let Some(idx) = lower.find(" allergy to ") {
        idx + " allergy to ".len()
    } else if let Some(idx) = lower.find(" cannot have ") {
        idx + " cannot have ".len()
    } else if let Some(idx) = lower.find(" can't have ") {
        idx + " can't have ".len()
    } else {
        return None;
    };
    let raw = text.get(start..)?;
    let value = raw
        .split(['.', '!', '?', ';', ',', '\n'])
        .next()?
        .trim()
        .trim_matches(|c: char| matches!(c, '.' | '!' | '?' | ';' | ',' | ':' | ' '));
    (!value.is_empty()).then(|| value.to_string())
}

fn bind_principal(text: &str, principal: &str) -> String {
    let principal = principal.trim();
    if principal.is_empty() {
        return text.to_string();
    }

    let lower = text.to_lowercase();
    let name_prefix = format!("my name is {} and i ", principal.to_lowercase());
    if lower.starts_with(&name_prefix)
        && let Some(rest) = text.get(name_prefix.len() - "i ".len()..)
    {
        return bind_principal(rest, principal);
    }

    let mut output = Vec::new();
    let mut tokens = text.split_whitespace().peekable();
    while let Some(token) = tokens.next() {
        let core = token_core_lower(token);
        if core != "i" {
            output.push(bind_token(token, principal));
            continue;
        }

        if let Some(next) = tokens.peek() {
            let next_core = token_core_lower(next);
            if next_core == "really" {
                let really = tokens.next();
                if let (Some(really_token), Some(verb_token)) = (really, tokens.peek()) {
                    let verb_core = token_core_lower(verb_token);
                    if let Some(verb) = principal_verb(&verb_core) {
                        let suffix = token_suffix(verb_token);
                        let _ = tokens.next();
                        output.push(format!("{principal} {really_token} {verb}{suffix}"));
                        continue;
                    }
                }
                output.push(principal.to_string());
                if let Some(really_token) = really {
                    output.push(really_token.to_string());
                }
                continue;
            }
            if let Some(verb) = principal_verb(&next_core) {
                let suffix = token_suffix(next);
                let _ = tokens.next();
                output.push(format!("{principal} {verb}{suffix}"));
                continue;
            }
        }
        output.push(bind_token(token, principal));
    }
    output.join(" ")
}

fn token_core_lower(token: &str) -> String {
    token
        .trim_matches(|c: char| !c.is_alphanumeric() && c != '\'')
        .to_lowercase()
}

fn token_suffix(token: &str) -> String {
    token
        .chars()
        .rev()
        .take_while(|c| !c.is_alphanumeric() && *c != '\'')
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect()
}

fn principal_verb(core: &str) -> Option<&'static str> {
    match core {
        "like" => Some("likes"),
        "dislike" => Some("dislikes"),
        "live" => Some("lives"),
        "work" => Some("works"),
        "grew" => Some("grew"),
        "prefer" => Some("prefers"),
        "love" => Some("loves"),
        "hate" => Some("hates"),
        "want" => Some("wants"),
        "need" => Some("needs"),
        "am" => Some("is"),
        "have" => Some("has"),
        _ => None,
    }
}

fn bind_token(token: &str, principal: &str) -> String {
    let suffix = token_suffix(token);
    let core = token_core_lower(token);
    let replacement = match core.as_str() {
        "i" => Some(principal.to_string()),
        "me" | "myself" => Some(principal.to_string()),
        "my" | "mine" => Some(format!("{principal}'s")),
        "i'm" | "im" => Some(format!("{principal} is")),
        "i've" | "ive" => Some(format!("{principal} has")),
        "i'd" | "id" => Some(format!("{principal} would")),
        "i'll" | "ill" => Some(format!("{principal} will")),
        _ => None,
    };
    match replacement {
        Some(mut value) => {
            value.push_str(&suffix);
            value
        }
        None => token.to_string(),
    }
}

/// Pull a textual representation out of a [`Message`].
///
/// `Message::rag_text` is `pub(crate)` in rig-core, so we re-implement the
/// equivalent walk here over the public content enums.
fn render_message_text(msg: &Message) -> Option<String> {
    use rig::completion::message::{
        AssistantContent, Message as Msg, ReasoningContent, UserContent,
    };

    match msg {
        Msg::System { content } => Some(content.clone()),
        Msg::User { content } => {
            let mut buf = String::new();
            for item in content.iter() {
                if let UserContent::Text(text) = item {
                    if !buf.is_empty() {
                        buf.push('\n');
                    }
                    buf.push_str(&text.text);
                }
            }
            (!buf.is_empty()).then_some(buf)
        }
        Msg::Assistant { content, .. } => {
            let mut buf = String::new();
            for item in content.iter() {
                match item {
                    AssistantContent::Text(text) => {
                        if !buf.is_empty() {
                            buf.push('\n');
                        }
                        buf.push_str(&text.text);
                    }
                    AssistantContent::Reasoning(reasoning) => {
                        for entry in reasoning.content.iter() {
                            if let ReasoningContent::Text { text, .. } = entry {
                                if !buf.is_empty() {
                                    buf.push('\n');
                                }
                                buf.push_str(text);
                            }
                        }
                    }
                    AssistantContent::ToolCall(_) | AssistantContent::Image(_) => {}
                }
            }
            (!buf.is_empty()).then_some(buf)
        }
    }
}

impl<M> PromptHook<M> for MemvidPersistHook<M>
where
    M: CompletionModel,
{
    async fn on_completion_call(&self, prompt: &Message, _history: &[Message]) -> HookAction {
        if let Some(text) = self.render(prompt) {
            self.write(&text, "user");
        }
        HookAction::cont()
    }

    async fn on_completion_response(
        &self,
        _prompt: &Message,
        response: &CompletionResponse<M::Response>,
    ) -> HookAction {
        if !self.config.persist_assistant {
            return HookAction::cont();
        }
        for content in response.choice.iter() {
            let synthetic = Message::Assistant {
                id: None,
                content: rig::OneOrMany::one(content.clone()),
            };
            if let Some(text) = self.render(&synthetic) {
                self.write(&text, "assistant");
            }
        }
        HookAction::cont()
    }
}

#[cfg(test)]
mod tests {
    use super::{allergy_value, bind_principal, supplemental_memory_cards};

    #[test]
    fn bind_principal_rewrites_first_person_tokens() {
        let rewritten = bind_principal(
            "My name is Alice. I'm allergic to peanuts, and I like espresso.",
            "Alice",
        );
        assert_eq!(
            rewritten,
            "Alice's name is Alice. Alice is allergic to peanuts, and Alice likes espresso."
        );
    }

    #[test]
    fn bind_principal_rewrites_common_verbs_after_adverbs() {
        assert_eq!(
            bind_principal("I really dislike instant coffee.", "Alice"),
            "Alice really dislikes instant coffee."
        );
    }

    #[test]
    fn bind_principal_collapses_name_intro_before_verbs() {
        assert_eq!(
            bind_principal(
                "My name is Alice and I work at Acme as a staff engineer.",
                "Alice",
            ),
            "Alice works at Acme as a staff engineer."
        );
    }

    #[test]
    fn bind_principal_ignores_empty_principal() {
        assert_eq!(bind_principal("I like rust", "  "), "I like rust");
    }

    #[test]
    fn allergy_value_extracts_common_forms() {
        assert_eq!(
            allergy_value("Alice is allergic to peanuts."),
            Some("peanuts".to_string())
        );
        assert_eq!(
            allergy_value("Alice cannot have shellfish, thanks"),
            Some("shellfish".to_string())
        );
    }

    #[test]
    fn supplemental_cards_build_allergy_profile() {
        let cards = supplemental_memory_cards(
            "Alice is allergic to peanuts.",
            "Alice",
            42,
            Some("scope".to_string()),
        );
        assert_eq!(cards.len(), 1);
        for card in &cards {
            assert_eq!(card.kind, memvid_core::MemoryKind::Profile);
            assert_eq!(card.entity, "alice");
            assert_eq!(card.slot, "allergy");
            assert_eq!(card.value, "peanuts");
            assert_eq!(card.source_frame_id, 42);
        }
    }

    #[test]
    fn supplemental_cards_build_manager_relationships() {
        let cards = supplemental_memory_cards(
            "Bob is Alice's manager at Acme. He reports to Carol, the VP.",
            "Alice",
            42,
            Some("scope".to_string()),
        );
        assert!(cards.iter().any(|card| {
            card.kind == memvid_core::MemoryKind::Relationship
                && card.entity == "alice"
                && card.slot == "manager"
                && card.value == "Bob"
        }));
        assert!(cards.iter().any(|card| {
            card.kind == memvid_core::MemoryKind::Relationship
                && card.entity == "bob"
                && card.slot == "reports_to"
                && card.value == "Carol"
        }));
        assert!(cards.iter().any(|card| {
            card.kind == memvid_core::MemoryKind::Fact
                && card.entity == "bob"
                && card.slot == "employer"
                && card.value == "Acme"
        }));
        assert!(cards.iter().any(|card| {
            card.kind == memvid_core::MemoryKind::Profile
                && card.entity == "carol"
                && card.slot == "title"
                && card.value == "VP"
        }));
    }
}
