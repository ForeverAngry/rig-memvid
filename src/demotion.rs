//! [`MemvidDemotionHook`]: a [`rig::memory::DemotionHook`] that drains
//! messages evicted from an agent's active window into a [`MemvidStore`].
//!
//! Pair with `rig_memory::DemotingPolicyMemory` (or any composing adapter
//! over [`rig::memory::ConversationMemory`]) to turn truncation into
//! durable, queryable demotion: the eviction policy drops old turns from
//! the active prompt, this hook lands them in the same `.mv2` archive
//! you use for retrieval.
//!
//! This module is gated on the `compaction` feature so callers that only
//! want plain RAG over `MemvidStore` are not pulled into the
//! `rig_core::memory` surface.

use std::sync::Arc;

use rig::completion::Message;
use rig::memory::{DemotionHook, MemoryError};
use rig::wasm_compat::WasmBoxedFuture;

use crate::dedup::DedupSet;
use crate::frame_writer::{commit_if_each_turn, write_frame};
use crate::hook::{MemoryConfig, WritePolicy, render_message_text};
use crate::store::MemvidStore;

/// Adapts a [`MemvidStore`] to [`rig::memory::DemotionHook`], persisting
/// every demoted message into the underlying `.mv2` archive.
///
/// Behaviour mirrors [`crate::MemvidPersistHook`]:
///
/// - [`WritePolicy::Disabled`] → drop demoted turns silently (the hook
///   becomes a no-op).
/// - [`WritePolicy::Raw`] → persist the verbatim text of each message.
/// - [`WritePolicy::Custom`] → run the caller transform on each message.
///
/// Tags from [`MemoryConfig::default_tags`] are attached to every frame.
/// The hook adds these extra metadata keys for downstream filtering and
/// dedup:
///
/// - `chat_role` — `"user"`, `"assistant"`, or `"system"`.
///   `Message` does not carry a dedicated tool role; tool-call payloads
///   arrive as `Assistant` / `User` content variants.
/// - `kind` — always `"demoted_message"`.
/// - `conversation_id` — the value passed to [`Self::on_demote`].
/// - `dedup_key` — 64-character hex blake3 fingerprint of
///   `(kind, conversation_id, role, scope, text)`.
///
/// `MemoryConfig::scope` and `MemoryConfig::commit_each_turn` are honoured
/// the same way [`crate::MemvidPersistHook`] honours them.
///
/// # Idempotency
///
/// Satisfies the upstream contract on `(conversation_id, messages)`
/// **within a single process lifetime** via an in-memory content-hash
/// gate. Calling [`Self::on_demote`] more than once with the same input
/// produces exactly one frame per message. Cross-restart idempotency is
/// opt-in via [`Self::dedup_snapshot`] /
/// [`Self::load_dedup_snapshot`]: snapshot before shutdown, replay on
/// startup.
///
/// # Errors
///
/// All write failures surface as [`MemoryError::Backend`]. A failed
/// `put_text_uncommitted` aborts the batch immediately so the upstream
/// `CompactingMemory` does not evict messages that did not durably
/// land.
///
/// # Example
///
/// ```no_run
/// use rig_memvid::{
///     MemoryConfig, MemvidDemotionHook, MemvidStore, MemvidStoreBuilder,
/// };
///
/// # fn build() -> Result<(), Box<dyn std::error::Error>> {
/// let store: MemvidStore = MemvidStore::builder().path("memory.mv2").open_or_create()?;
/// let _hook = MemvidDemotionHook::with_defaults(store);
/// # Ok(()) }
/// ```
pub struct MemvidDemotionHook {
    store: MemvidStore,
    config: MemoryConfig,
    dedup: Arc<DedupSet>,
}

impl Clone for MemvidDemotionHook {
    fn clone(&self) -> Self {
        Self {
            store: self.store.clone(),
            config: self.config.clone(),
            dedup: Arc::clone(&self.dedup),
        }
    }
}

impl std::fmt::Debug for MemvidDemotionHook {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemvidDemotionHook")
            .field("config", &self.config)
            .field("dedup", &self.dedup)
            .finish_non_exhaustive()
    }
}

impl MemvidDemotionHook {
    /// Build a hook that persists demoted messages into `store` according
    /// to `config`.
    pub fn new(store: MemvidStore, config: MemoryConfig) -> Self {
        Self {
            store,
            config,
            dedup: Arc::new(DedupSet::new()),
        }
    }

    /// Build a hook with the default [`MemoryConfig`]
    /// ([`WritePolicy::Raw`], `commit_each_turn = true`).
    pub fn with_defaults(store: MemvidStore) -> Self {
        Self::new(store, MemoryConfig::default())
    }

    /// Borrow the underlying store. Useful for sharing the same `.mv2`
    /// archive between recall (`MemvidStore` as a `VectorStoreIndex`) and
    /// demotion (this hook).
    pub fn store(&self) -> &MemvidStore {
        &self.store
    }

    /// Borrow the active configuration.
    pub fn config(&self) -> &MemoryConfig {
        &self.config
    }

    /// Snapshot of the in-memory dedup set as sorted hex-encoded keys.
    ///
    /// Persist this to a sidecar file before shutdown to extend the
    /// idempotency contract across process restarts; restore via
    /// [`Self::load_dedup_snapshot`] on the next instance.
    pub fn dedup_snapshot(&self) -> Result<Vec<String>, MemoryError> {
        self.dedup
            .snapshot()
            .map_err(|err| MemoryError::backend(Box::new(err)))
    }

    /// Replay a snapshot produced by [`Self::dedup_snapshot`] into this
    /// hook. Malformed entries are skipped with a warning.
    pub fn load_dedup_snapshot(&self, hexes: &[String]) -> Result<(), MemoryError> {
        self.dedup
            .extend_from_snapshot(hexes)
            .map_err(|err| MemoryError::backend(Box::new(err)))
    }

    fn render(&self, msg: &Message) -> Option<String> {
        match &self.config.policy {
            WritePolicy::Disabled => None,
            WritePolicy::Raw => render_message_text(msg),
            WritePolicy::Custom(f) => f(msg),
        }
    }

    fn write(&self, conversation_id: &str, msg: &Message) -> Result<(), MemoryError> {
        let Some(text) = self.render(msg) else {
            return Ok(());
        };
        write_frame(
            &self.store,
            &self.config,
            &self.dedup,
            crate::metadata::FrameKind::DemotedMessage,
            conversation_id,
            role_label(msg),
            &text,
        )
        .map(|_| ())
    }
}

impl DemotionHook for MemvidDemotionHook {
    fn on_demote<'a>(
        &'a self,
        conversation_id: &'a str,
        messages: Vec<Message>,
    ) -> WasmBoxedFuture<'a, Result<(), MemoryError>> {
        Box::pin(async move {
            if matches!(self.config.policy, WritePolicy::Disabled) || messages.is_empty() {
                return Ok(());
            }
            for msg in &messages {
                self.write(conversation_id, msg)?;
            }
            commit_if_each_turn(&self.store, &self.config)?;
            #[cfg(feature = "observe")]
            rig_tap::emit_kind(
                conversation_id,
                rig_tap::EventKind::MemoryDemoted {
                    demoted_count: messages.len(),
                    tags: self.config.default_tags.clone(),
                },
            );
            Ok(())
        })
    }
}

fn role_label(msg: &Message) -> &'static str {
    match msg {
        Message::System { .. } => "system",
        Message::User { .. } => "user",
        Message::Assistant { .. } => "assistant",
    }
}
