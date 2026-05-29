//! Compatibility re-exports for the backend-agnostic in-memory reference store.
//!
//! The implementation lives in [`rig_memory_policy::inmem`] so other memory
//! adapters can reuse the same no-disk lexical reference backend. These
//! re-exports preserve the historic `rig_memvid::inmem::*` import path.

pub use rig_memory_policy::inmem::{Episode, InMemoryHit, InMemoryStore};

/// Error type returned by [`InMemoryStore`] operations.
pub type InMemoryError = rig_memory_policy::PolicyError;
