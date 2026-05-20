//! Backend-agnostic structured-memory abstraction.
//!
//! [`MemoryGraph`] is the read-side trait that
//! [`crate::MemoryCardContext`] (and any future graph-aware adapters) is
//! generic over. It models a memory store as a typed entity-relationship
//! graph: nodes are *entities* (subjects), edges are *slots* (predicates)
//! pointing at *values* (objects), each carrying polarity, timestamps,
//! and provenance.
//!
//! # Why a trait
//!
//! Memvid is the most complete implementation today (it ships frames +
//! Logic-Mesh + memory cards in a single `.mv2` file), but the same
//! abstraction fits Neo4j (graph-native), Postgres / SQLite (cards
//! table), and the in-memory test fake here. Keeping the trait local
//! to `rig-memvid` for now is deliberate — the next backend that
//! implements it will tell us where the upstream home should be (most
//! likely `rig-core::memory` next to
//! [`rig::vector_store::VectorStoreIndex`]).
//!
//! # Semantics
//!
//! - All methods are **read-only** and **synchronous**: implementations
//!   are expected to be backed by an in-memory snapshot, a memory-mapped
//!   file, or a thread-pool-bound database connection. Async backends
//!   should `block_in_place` or expose a separate async trait above
//!   this one.
//! - The error type is implementation-defined but must convert into
//!   [`rig::vector_store::VectorStoreError`] so adapters can surface
//!   failures uniformly.
//! - `entity_memories` returning an empty `Vec` is **not** an error;
//!   unknown entities are valid.
//!
//! # Implementing for a new backend
//!
//! ```rust,no_run
//! use memvid_core::MemoryCard;
//! use rig::vector_store::VectorStoreError;
//! use rig_memvid::MemoryGraph;
//!
//! struct PgGraph; // imagine a postgres-backed cards table
//!
//! impl MemoryGraph for PgGraph {
//!     type Error = VectorStoreError;
//!     fn memory_card_count(&self) -> Result<usize, Self::Error> { Ok(0) }
//!     fn all_memory_cards(&self) -> Result<Vec<MemoryCard>, Self::Error> { Ok(vec![]) }
//!     fn entity_memories(&self, _: &str) -> Result<Vec<MemoryCard>, Self::Error> { Ok(vec![]) }
//!     fn current_memory(&self, _: &str, _: &str) -> Result<Option<MemoryCard>, Self::Error> { Ok(None) }
//!     fn entity_preferences(&self, _: &str) -> Result<Vec<MemoryCard>, Self::Error> { Ok(vec![]) }
//!     fn memory_timeline(&self, _: &str) -> Result<Vec<MemoryCard>, Self::Error> { Ok(vec![]) }
//! }
//! ```

use memvid_core::MemoryCard;

/// Read-side abstraction over a structured-memory store.
///
/// Implementations expose entity / slot / value cards with versioning,
/// polarity, and provenance. The default
/// [`crate::MemvidStore`] implementation backs all six methods with the
/// memvid `.mv2` memories track.
pub trait MemoryGraph {
    /// Error returned by graph queries. Must be convertible to
    /// [`rig::vector_store::VectorStoreError`] when used with an
    /// adapter that exposes a [`rig::vector_store::VectorStoreIndex`]
    /// (such as [`crate::MemoryCardContext`]).
    type Error: Into<rig::vector_store::VectorStoreError>;

    /// Total number of cards currently stored.
    fn memory_card_count(&self) -> Result<usize, Self::Error>;

    /// Snapshot of every card. Used by selection strategies that need
    /// to filter / sort across the whole set.
    fn all_memory_cards(&self) -> Result<Vec<MemoryCard>, Self::Error>;

    /// All cards for `entity`. Empty `Vec` for unknown entities.
    fn entity_memories(&self, entity: &str) -> Result<Vec<MemoryCard>, Self::Error>;

    /// Most recent non-retracted value for `entity`/`slot`, if any.
    fn current_memory(&self, entity: &str, slot: &str) -> Result<Option<MemoryCard>, Self::Error>;

    /// Preference-kind cards for `entity`.
    fn entity_preferences(&self, entity: &str) -> Result<Vec<MemoryCard>, Self::Error>;

    /// Event-kind cards for `entity` in chronological order.
    fn memory_timeline(&self, entity: &str) -> Result<Vec<MemoryCard>, Self::Error>;

    /// Cards whose `entity` mentions appear in `query` (case-insensitive
    /// whole-word match). The default implementation snapshots the entire
    /// archive via [`MemoryGraph::all_memory_cards`] and filters in pure
    /// Rust, which is correct but clones every card; backends backed by a
    /// graph store should override this to filter behind their own
    /// locking / indexing and avoid the intermediate full-archive
    /// allocation.
    fn cards_for_query(&self, query: &str) -> Result<Vec<MemoryCard>, Self::Error> {
        let needle = query.to_lowercase();
        let all = self.all_memory_cards()?;
        Ok(all
            .into_iter()
            .filter(|card| {
                let entity = card.entity.to_lowercase();
                !entity.is_empty() && crate::cards_context::contains_word(&needle, &entity)
            })
            .collect())
    }
}

impl MemoryGraph for crate::MemvidStore {
    type Error = crate::MemvidError;

    fn memory_card_count(&self) -> Result<usize, Self::Error> {
        Self::memory_card_count(self)
    }

    fn all_memory_cards(&self) -> Result<Vec<MemoryCard>, Self::Error> {
        Self::all_memory_cards(self)
    }

    fn entity_memories(&self, entity: &str) -> Result<Vec<MemoryCard>, Self::Error> {
        Self::entity_memories(self, entity)
    }

    fn current_memory(&self, entity: &str, slot: &str) -> Result<Option<MemoryCard>, Self::Error> {
        Self::current_memory(self, entity, slot)
    }

    fn entity_preferences(&self, entity: &str) -> Result<Vec<MemoryCard>, Self::Error> {
        Self::entity_preferences(self, entity)
    }

    fn memory_timeline(&self, entity: &str) -> Result<Vec<MemoryCard>, Self::Error> {
        Self::memory_timeline(self, entity)
    }

    fn cards_for_query(&self, query: &str) -> Result<Vec<MemoryCard>, Self::Error> {
        Self::cards_for_query(self, query)
    }
}
