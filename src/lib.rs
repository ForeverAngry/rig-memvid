//! Memvid-backed persistent memory for [Rig](https://crates.io/crates/rig-core).
//!
//! See the crate-level [README](../README.md) for a quick-start. The two
//! main types are:
//!
//! - [`MemvidStore`] — implements [`rig::vector_store::VectorStoreIndex`].
//!   Wire it into an agent with `agent.dynamic_context(n, store.clone())`.
//! - [`MemvidPersistHook`] — implements
//!   [`rig::agent::PromptHook`]. Attach with
//!   `prompt.with_hook(hook)` to record every turn into the same store.
//!
//! Sharing a single [`MemvidStore`] between recall and persistence is the
//! intended pattern: writes through the hook are visible to subsequent
//! searches in the same process.
//!
//! # Re-exports
//!
//! [`memvid_core`] is re-exported so callers do not need to add it as a
//! direct dependency to construct [`memvid_core::PutOptions`],
//! [`memvid_core::AclContext`], or [`memvid_core::SearchRequest`] when
//! reaching for [`MemvidStore::search`] / [`MemvidStore::put_text`].
//!
//! # Platform support
//!
//! The crate is `cfg`-gated off on `wasm`: memvid relies on synchronous
//! file I/O and a process-level file lock that have no analogue under
//! `wasm32-unknown-unknown`. On wasm targets the crate is intentionally
//! empty so that downstream crates can still depend on it transitively
//! without a build break.

#![cfg_attr(not(target_family = "wasm"), deny(missing_docs))]

#[cfg(target_family = "wasm")]
compile_error!(
    "rig-memvid does not currently support `wasm` targets: memvid-core requires \
     synchronous file I/O and OS-level file locks. Disable rig-memvid for wasm \
     builds via Cargo target-specific dependencies."
);

#[cfg(not(target_family = "wasm"))]
mod cards_context;
#[cfg(all(not(target_family = "wasm"), feature = "compaction"))]
mod compactor;
#[cfg(all(not(target_family = "wasm"), feature = "compaction"))]
mod dedup;
#[cfg(all(not(target_family = "wasm"), feature = "compaction"))]
mod demotion;
#[cfg(not(target_family = "wasm"))]
mod error;
#[cfg(all(not(target_family = "wasm"), feature = "compaction"))]
mod frame_writer;
#[cfg(not(target_family = "wasm"))]
mod hook;
#[cfg(not(target_family = "wasm"))]
pub mod inmem;
#[cfg(not(target_family = "wasm"))]
mod memory_graph;
#[cfg(all(not(target_family = "wasm"), feature = "compaction"))]
pub mod metadata;
#[cfg(all(not(target_family = "wasm"), feature = "context-projection"))]
pub mod projection;
#[cfg(all(not(target_family = "wasm"), feature = "context-projection"))]
pub use projection::{
    IntoContextItem, MemoryCandidate, MemoryContextPack, SupersededCandidate,
    card_docs_to_context_items, inmem_hits_to_context_items, items_to_memory_candidates,
    memory_cards_to_context_items, search_hits_to_context_items, supersede,
};
#[cfg(all(
    not(target_family = "wasm"),
    feature = "context-projection",
    feature = "compaction"
))]
pub use projection::{
    MemoryFrameRole, PartitionedHits, frame_role, partition_search_hits_by_role,
    typed_search_hit_to_context_item, typed_search_hits_to_context_items,
    typed_search_hits_to_memory_candidates,
};
#[cfg(not(target_family = "wasm"))]
mod store;

#[cfg(not(target_family = "wasm"))]
pub use cards_context::{CardDoc, CardSelection, MemoryCardContext};
#[cfg(all(not(target_family = "wasm"), feature = "compaction"))]
pub use compactor::MemvidStoringCompactor;
#[cfg(all(not(target_family = "wasm"), feature = "compaction"))]
pub use demotion::MemvidDemotionHook;
#[cfg(not(target_family = "wasm"))]
pub use error::MemvidError;
#[cfg(not(target_family = "wasm"))]
pub use hook::{
    MemoryConfig, MemoryConfigBuilder, MemvidPersistHook, WriteFailure, WriteFailureAction,
    WriteFailurePhase, WritePolicy, WriteTransform,
};
#[cfg(not(target_family = "wasm"))]
pub use inmem::{Episode, InMemoryError, InMemoryHit, InMemoryStore};
#[cfg(not(target_family = "wasm"))]
pub use memory_graph::MemoryGraph;
#[cfg(all(not(target_family = "wasm"), feature = "compaction"))]
pub use metadata::{FrameKind, MemvidFrameMetadata};
/// Re-exports of the `rig-memory` compaction surface so callers can
/// build a `CompactingMemory<M, P, MemvidStoringCompactor<C>>` without
/// adding `rig-memory` as a direct dependency.
#[cfg(all(not(target_family = "wasm"), feature = "compaction"))]
pub use rig::memory::{Compactor, DemotionHook, MemoryError};
#[cfg(all(not(target_family = "wasm"), feature = "compaction"))]
pub use rig_memory::{CompactingMemory, TemplateCompactor, TextSummary};
#[cfg(not(target_family = "wasm"))]
pub use store::{MemvidFilter, MemvidStore, MemvidStoreBuilder};

#[cfg(not(target_family = "wasm"))]
pub use memvid_core::types::{LogicMeshStats, SearchHitEntity};
/// Re-exports of the memvid-core memory-card surface that
/// [`MemvidStore`] returns from
/// [`MemvidStore::entity_memories`] and friends. Re-exported here so
/// callers do not need a direct `memvid-core` dependency just to name
/// the structured-memory types.
#[cfg(not(target_family = "wasm"))]
pub use memvid_core::{
    EntityKind, FollowResult, MemoryCard, MemoryCardId, MemoryKind, MeshEdge, MeshNode, Polarity,
    VersionRelation,
};

#[cfg(not(target_family = "wasm"))]
pub use memvid_core;
