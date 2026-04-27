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

#![cfg(not(target_family = "wasm"))]
#![deny(missing_docs)]

mod error;
mod hook;
mod store;

pub use error::MemvidError;
pub use hook::{MemoryConfig, MemvidPersistHook, WritePolicy, WriteTransform};
pub use store::{MemvidFilter, MemvidStore, MemvidStoreBuilder};

pub use memvid_core;
