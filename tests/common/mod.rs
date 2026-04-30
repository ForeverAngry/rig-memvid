//! Shared helpers for `rig-memvid` integration & smoke tests.
//!
//! Lints are relaxed here (and re-relaxed in the consuming test files) so
//! tests can use `unwrap`/`expect` without having to pepper every assertion.
#![allow(
    dead_code,
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::panic_in_result_fn
)]

#[cfg(feature = "lex")]
use anyhow::Result;
#[cfg(feature = "lex")]
use rig_memvid::MemvidStore;

/// Build a fresh lex-enabled [`MemvidStore`] at `path`. Errors if the file
/// already exists. Used by both [`tests/integration.rs`] and
/// [`tests/smoke.rs`] to keep test bring-up uniform.
#[cfg(feature = "lex")]
pub fn lex_store(path: &std::path::Path) -> Result<MemvidStore> {
    Ok(MemvidStore::builder().path(path).enable_lex().create()?)
}
