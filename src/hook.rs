//! [`MemvidPersistHook`]: a [`PromptHook`] that persists every turn of an
//! agent conversation into a [`MemvidStore`].

use std::marker::PhantomData;
use std::sync::Arc;

use memvid_core::PutOptions;
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
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            policy: WritePolicy::default(),
            commit_each_turn: true,
            default_tags: Vec::new(),
            scope: None,
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
        let opts = self.put_options(chat_role);
        let result = if self.config.commit_each_turn {
            self.store.put_text(text, opts)
        } else {
            self.store.put_text_uncommitted(text, opts)
        };
        if let Err(err) = result {
            tracing::warn!(
                target: "rig_memvid::hook",
                error = %err,
                role = chat_role,
                "failed to persist message into memvid",
            );
        }
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
