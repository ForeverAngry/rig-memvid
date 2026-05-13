//! [`MemvidStoringCompactor`]: a [`rig::memory::Compactor`] decorator that
//! persists every produced summary into a [`MemvidStore`] before
//! returning it to the composing memory adapter.
//!
//! The inner compactor (typically [`rig_memory::TemplateCompactor`] or an
//! LLM-backed implementation) produces the artifact spliced back into the
//! active prompt. This wrapper shadow-writes the same artifact into the
//! `.mv2` archive so the rolled-up history remains queryable later, even
//! after the in-process `CompactingMemory` is dropped.
//!
//! Gated on the `compaction` feature.

use std::sync::Arc;

use rig::completion::Message;
use rig::memory::{Compactor, MemoryError};
use rig::wasm_compat::WasmBoxedFuture;

use crate::dedup::DedupSet;
use crate::frame_writer::{commit_if_each_turn, write_frame};
use crate::hook::{MemoryConfig, render_message_text};
use crate::store::MemvidStore;

/// Wraps any [`rig::memory::Compactor`] and persists each produced
/// artifact into a [`MemvidStore`].
///
/// The inner compactor's [`Compactor::Artifact`] type is preserved
/// unchanged; the composing memory adapter still receives the original
/// artifact instance. Persistence happens *between* the inner call and
/// the return, which means an I/O failure surfaces as a
/// [`MemoryError::Backend`] rather than silently dropping the rollup.
///
/// Each persisted frame carries these extra metadata keys:
///
/// - `kind` — always `"compaction_summary"`.
/// - `conversation_id` — the value passed to [`Compactor::compact`].
/// - `chat_role` — `"system"`, the role used by all known
///   `rig_memory` artifacts (`TextSummary` converts into
///   [`Message::System`]).
/// - `dedup_key` — 64-character hex blake3 fingerprint of
///   `(kind, conversation_id, role, scope, rendered_text)`.
///
/// Tags from [`MemoryConfig::default_tags`] and the URI prefix from
/// [`MemoryConfig::scope`] are honoured the same way
/// [`crate::MemvidPersistHook`] honours them. `commit_each_turn` is
/// honoured too: when `true`, the archive is committed after each
/// summary is written so subsequent searches see the rollup.
///
/// # Idempotency
///
/// Satisfies the upstream contract on `(conversation_id, evicted,
/// carry_over)` **within a single process lifetime** via an in-memory
/// content-hash gate on the *rendered artifact text*. If the inner
/// compactor is deterministic, repeated calls land exactly one frame.
/// If the inner compactor is non-deterministic (LLM-backed without a
/// fixed seed), each distinct output produces its own frame — by
/// design.
///
/// # Example
///
/// ```no_run
/// use rig_memvid::{
///     MemoryConfig, MemvidStore, MemvidStoreBuilder, MemvidStoringCompactor,
/// };
/// use rig_memory::TemplateCompactor;
///
/// # fn build() -> Result<(), Box<dyn std::error::Error>> {
/// let store: MemvidStore = MemvidStore::builder().path("memory.mv2").open_or_create()?;
/// let inner = TemplateCompactor::new();
/// let _compactor = MemvidStoringCompactor::new(store, inner, MemoryConfig::default());
/// # Ok(()) }
/// ```
pub struct MemvidStoringCompactor<C> {
    store: MemvidStore,
    inner: C,
    config: MemoryConfig,
    dedup: Arc<DedupSet>,
}

impl<C: Clone> Clone for MemvidStoringCompactor<C> {
    fn clone(&self) -> Self {
        Self {
            store: self.store.clone(),
            inner: self.inner.clone(),
            config: self.config.clone(),
            dedup: Arc::clone(&self.dedup),
        }
    }
}

impl<C: std::fmt::Debug> std::fmt::Debug for MemvidStoringCompactor<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemvidStoringCompactor")
            .field("inner", &self.inner)
            .field("config", &self.config)
            .field("dedup", &self.dedup)
            .finish_non_exhaustive()
    }
}

impl<C> MemvidStoringCompactor<C> {
    /// Wrap `inner` so every artifact it produces is persisted into
    /// `store` according to `config`.
    pub fn new(store: MemvidStore, inner: C, config: MemoryConfig) -> Self {
        Self {
            store,
            inner,
            config,
            dedup: Arc::new(DedupSet::new()),
        }
    }

    /// Wrap `inner` with the default [`MemoryConfig`].
    pub fn with_defaults(store: MemvidStore, inner: C) -> Self {
        Self::new(store, inner, MemoryConfig::default())
    }

    /// Borrow the inner compactor.
    pub fn inner(&self) -> &C {
        &self.inner
    }

    /// Borrow the underlying store.
    pub fn store(&self) -> &MemvidStore {
        &self.store
    }

    /// Borrow the active configuration.
    pub fn config(&self) -> &MemoryConfig {
        &self.config
    }

    /// Snapshot of the in-memory dedup set as sorted hex-encoded keys.
    pub fn dedup_snapshot(&self) -> Result<Vec<String>, MemoryError> {
        self.dedup
            .snapshot()
            .map_err(|err| MemoryError::backend(Box::new(err)))
    }

    /// Replay a snapshot produced by [`Self::dedup_snapshot`].
    pub fn load_dedup_snapshot(&self, hexes: &[String]) -> Result<(), MemoryError> {
        self.dedup
            .extend_from_snapshot(hexes)
            .map_err(|err| MemoryError::backend(Box::new(err)))
    }
}

impl<C> Compactor for MemvidStoringCompactor<C>
where
    C: Compactor,
{
    type Artifact = C::Artifact;

    fn compact<'a>(
        &'a self,
        conversation_id: &'a str,
        evicted: &'a [Message],
        carry_over: Option<&'a Self::Artifact>,
    ) -> WasmBoxedFuture<'a, Result<Self::Artifact, MemoryError>> {
        Box::pin(async move {
            let artifact = self
                .inner
                .compact(conversation_id, evicted, carry_over)
                .await?;

            // Render the artifact via its `Into<Message>` representation.
            // We clone first because the trait consumes the artifact and
            // we still want to return the original to the caller.
            // Non-text artifacts (tool calls, images) render as `None`
            // and are skipped — we never silently coerce them into an
            // empty frame.
            if let Some(rendered) = render_message_text(&artifact.clone().into()) {
                let written = write_frame(
                    &self.store,
                    &self.config,
                    &self.dedup,
                    crate::metadata::FrameKind::CompactionSummary,
                    conversation_id,
                    "system",
                    &rendered,
                )?;
                if written {
                    commit_if_each_turn(&self.store, &self.config)?;
                }
            }

            Ok(artifact)
        })
    }
}
