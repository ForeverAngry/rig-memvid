//! In-process content-hash dedup for the [`MemvidDemotionHook`] and
//! [`MemvidStoringCompactor`].
//!
//! The upstream `rig::memory::{DemotionHook, Compactor}` traits require
//! implementations to be idempotent on `(conversation_id, messages)`.
//! Memvid's `.mv2` log is append-only with no unique-key enforcement, so
//! this module provides a small content-hash gate that lives alongside
//! the hook / compactor instance and prevents the same frame from being
//! appended twice within a single process lifetime.
//!
//! Scope of the guarantee:
//!
//! - **Within a process:** invoking `on_demote` / `compact` more than
//!   once with the same `(conversation_id, kind, role, scope, text)`
//!   produces exactly one frame in the archive.
//! - **Across process restarts:** dedup state is not persisted by
//!   default. Callers that need cross-restart idempotency can snapshot
//!   the set via [`DedupSet::snapshot`] before shutdown and replay it
//!   into a fresh instance via [`DedupSet::extend_from_snapshot`].
//!
//! This module is `pub(crate)`; the public surface lives on
//! [`MemvidDemotionHook`] and [`MemvidStoringCompactor`].

use std::collections::HashSet;
use std::sync::RwLock;

use crate::error::MemvidError;

/// 32-byte content fingerprint produced by [`blake3::hash`].
pub(crate) type DedupKey = [u8; 32];

/// Compute the dedup key for a single frame.
///
/// Inputs are joined by NUL so that two distinct field tuples cannot
/// collide via concatenation (e.g. `("ab", "c")` vs `("a", "bc")`).
pub(crate) fn compute_key(
    kind: &str,
    conversation_id: &str,
    role: &str,
    scope: Option<&str>,
    text: &str,
) -> DedupKey {
    let mut hasher = blake3::Hasher::new();
    hasher.update(kind.as_bytes());
    hasher.update(&[0]);
    hasher.update(conversation_id.as_bytes());
    hasher.update(&[0]);
    hasher.update(role.as_bytes());
    hasher.update(&[0]);
    hasher.update(scope.unwrap_or("").as_bytes());
    hasher.update(&[0]);
    hasher.update(text.as_bytes());
    *hasher.finalize().as_bytes()
}

/// In-memory set of dedup keys with snapshot / restore for opt-in
/// cross-process persistence.
#[derive(Default)]
pub(crate) struct DedupSet {
    seen: RwLock<HashSet<DedupKey>>,
}

impl DedupSet {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Returns `true` if `key` has already been recorded.
    ///
    /// Returns [`MemvidError::Poisoned`] if a previous holder of the
    /// internal lock panicked.
    pub(crate) fn contains(&self, key: &DedupKey) -> Result<bool, MemvidError> {
        let guard = self.seen.read().map_err(|_| MemvidError::Poisoned)?;
        Ok(guard.contains(key))
    }

    /// Record `key` as seen. Idempotent.
    pub(crate) fn insert(&self, key: DedupKey) -> Result<(), MemvidError> {
        let mut guard = self.seen.write().map_err(|_| MemvidError::Poisoned)?;
        guard.insert(key);
        Ok(())
    }

    /// Snapshot the current set as a sorted list of hex-encoded keys,
    /// suitable for writing to a sidecar file.
    pub(crate) fn snapshot(&self) -> Result<Vec<String>, MemvidError> {
        let guard = self.seen.read().map_err(|_| MemvidError::Poisoned)?;
        let mut out: Vec<String> = guard.iter().map(hex_encode).collect();
        out.sort();
        Ok(out)
    }

    /// Replay a snapshot produced by [`Self::snapshot`] back into this
    /// set. Malformed entries are skipped with a `tracing::warn!`.
    pub(crate) fn extend_from_snapshot(&self, hexes: &[String]) -> Result<(), MemvidError> {
        let mut guard = self.seen.write().map_err(|_| MemvidError::Poisoned)?;
        for hex in hexes {
            match hex_decode(hex) {
                Some(key) => {
                    guard.insert(key);
                }
                None => {
                    tracing::warn!(
                        target: "rig_memvid::dedup",
                        invalid = %hex,
                        "skipping malformed dedup snapshot entry",
                    );
                }
            }
        }
        Ok(())
    }

    /// Number of recorded keys. Test/diagnostic surface.
    #[cfg(test)]
    pub(crate) fn len(&self) -> Result<usize, MemvidError> {
        let guard = self.seen.read().map_err(|_| MemvidError::Poisoned)?;
        Ok(guard.len())
    }
}

impl std::fmt::Debug for DedupSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let count = self.seen.read().map(|g| g.len()).unwrap_or_default();
        f.debug_struct("DedupSet").field("entries", &count).finish()
    }
}

fn hex_encode(key: &DedupKey) -> String {
    let mut out = String::with_capacity(64);
    for b in key {
        out.push(nibble_to_hex(b >> 4));
        out.push(nibble_to_hex(b & 0x0f));
    }
    out
}

/// Hex-encode a [`DedupKey`] as 64 lowercase ASCII chars. Public to the
/// crate so the hook and compactor can stamp the same hex form into
/// `extra_metadata`.
pub(crate) fn hex_encode_key(key: &DedupKey) -> String {
    hex_encode(key)
}

fn nibble_to_hex(n: u8) -> char {
    // `n` is masked to 0..=15 before reaching this function (or is
    // already a high-nibble shifted into 0..=15). The branch keeps
    // clippy::indexing_slicing happy without sacrificing readability.
    let n = n & 0x0f;
    if n < 10 {
        (b'0' + n) as char
    } else {
        (b'a' + n - 10) as char
    }
}

fn hex_decode(hex: &str) -> Option<DedupKey> {
    if hex.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    let bytes = hex.as_bytes();
    for i in 0..32 {
        let hi = nibble(bytes.get(i * 2).copied()?)?;
        let lo = nibble(bytes.get(i * 2 + 1).copied()?)?;
        if let Some(slot) = out.get_mut(i) {
            *slot = (hi << 4) | lo;
        }
    }
    Some(out)
}

fn nibble(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
    use super::*;

    #[test]
    fn distinct_inputs_produce_distinct_keys() {
        let a = compute_key("demoted_message", "c1", "user", None, "hello");
        let b = compute_key("demoted_message", "c1", "user", None, "hello world");
        let c = compute_key("compaction_summary", "c1", "user", None, "hello");
        let d = compute_key("demoted_message", "c2", "user", None, "hello");
        let e = compute_key("demoted_message", "c1", "assistant", None, "hello");
        let f = compute_key("demoted_message", "c1", "user", Some("s"), "hello");
        assert_ne!(a, b);
        assert_ne!(a, c);
        assert_ne!(a, d);
        assert_ne!(a, e);
        assert_ne!(a, f);
    }

    #[test]
    fn identical_inputs_produce_identical_keys() {
        let a = compute_key("demoted_message", "c1", "user", None, "hello");
        let b = compute_key("demoted_message", "c1", "user", None, "hello");
        assert_eq!(a, b);
    }

    #[test]
    fn boundary_collision_resistance() {
        // ("ab","c") vs ("a","bc") would collide under naive concat.
        let a = compute_key("ab", "c", "user", None, "");
        let b = compute_key("a", "bc", "user", None, "");
        assert_ne!(a, b);
    }

    #[test]
    fn set_round_trips_via_snapshot() {
        let set = DedupSet::new();
        let k1 = compute_key("kind", "conv", "user", None, "one");
        let k2 = compute_key("kind", "conv", "user", None, "two");
        set.insert(k1).unwrap();
        set.insert(k2).unwrap();
        let snap = set.snapshot().unwrap();
        assert_eq!(snap.len(), 2);

        let restored = DedupSet::new();
        restored.extend_from_snapshot(&snap).unwrap();
        assert!(restored.contains(&k1).unwrap());
        assert!(restored.contains(&k2).unwrap());
    }

    #[test]
    fn malformed_snapshot_entries_are_skipped() {
        let set = DedupSet::new();
        let good = compute_key("k", "c", "user", None, "x");
        let bad = "not-hex".to_string();
        let snap = vec![hex_encode(&good), bad];
        set.extend_from_snapshot(&snap).unwrap();
        assert_eq!(set.len().unwrap(), 1);
        assert!(set.contains(&good).unwrap());
    }
}
