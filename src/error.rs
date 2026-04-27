//! Error types for `rig-memvid`.

use rig::vector_store::VectorStoreError;

/// Errors produced by `rig-memvid`.
#[derive(Debug, thiserror::Error)]
pub enum MemvidError {
    /// Underlying memvid-core failure.
    #[error("memvid error: {0}")]
    Memvid(#[from] memvid_core::MemvidError),

    /// I/O error while opening, creating, or locking a memvid file.
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),

    /// Serde failure while round-tripping a hit into the caller's document
    /// type, or while encoding metadata into JSON for storage.
    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),

    /// A filter clause was constructed that memvid cannot represent (for
    /// example a `gt`/`lt` predicate, an `or` combination, or an `eq` against
    /// an unknown key).
    #[error("unsupported filter clause: {0}")]
    UnsupportedFilter(String),

    /// The store's mutex was poisoned by a previous panic.
    #[error("memvid store mutex poisoned")]
    Poisoned,
}

impl From<MemvidError> for VectorStoreError {
    fn from(err: MemvidError) -> Self {
        VectorStoreError::DatastoreError(Box::new(err))
    }
}
