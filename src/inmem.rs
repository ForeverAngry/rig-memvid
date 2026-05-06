//! In-memory append-only episode store with lexical retrieval.
//!
//! [`InMemoryStore`] is the no-disk companion to [`crate::MemvidStore`]:
//! it keeps episodes in a process-local `Vec` and ranks them with a
//! deterministic token-overlap score. It is intended for tests,
//! examples, and offline modes that don't want to spin up a memvid
//! `.mv2` archive but still need an [`Episode`]-shaped retrieval
//! surface.
//!
//! Compared to [`crate::MemvidStore`], this store
//!
//! - has no I/O, no file lock, and no embedder;
//! - returns at most `k` hits sorted by descending lexical score; and
//! - is generic over a user-defined [`Episode`] payload, so callers
//!   keep their own domain types (alert envelopes, hunter findings,
//!   etc.) without forcing them through memvid's serialisation.
//!
//! ```no_run
//! use rig_memvid::inmem::{Episode, InMemoryStore};
//!
//! #[derive(Clone)]
//! struct MyEpisode {
//!     summary: String,
//! }
//!
//! impl Episode for MyEpisode {
//!     fn summary(&self) -> &str { &self.summary }
//! }
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let store = InMemoryStore::<MyEpisode>::new();
//! let key = store
//!     .append(MyEpisode { summary: "scheduled maintenance".into() })
//!     .await?;
//! let hits = store.retrieve_similar("maintenance", 5).await?;
//! assert_eq!(hits.first().map(|h| h.episode.summary.as_str()),
//!            Some("scheduled maintenance"));
//! let _ = (key, hits);
//! # Ok(()) }
//! ```

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

/// Errors returned by [`InMemoryStore`] operations.
#[derive(Debug, thiserror::Error)]
pub enum InMemoryError {
    /// Direct lookup against an unknown key.
    #[error("episode not found: {0}")]
    NotFound(String),

    /// The store's mutex was poisoned by a previous panic.
    #[error("in-memory store mutex poisoned")]
    Poisoned,
}

/// User-defined episode payload with a searchable natural-language
/// summary.
///
/// Implementors keep their own domain shape — only the summary string
/// is observed by the lexical scorer.
pub trait Episode: Clone + Send + Sync + 'static {
    /// The natural-language summary used to rank hits against a query.
    /// Tokens are split on whitespace; longer summaries score higher
    /// for queries with broader vocabulary overlap.
    fn summary(&self) -> &str;
}

/// A retrieval hit returned by [`InMemoryStore::retrieve_similar`].
#[derive(Debug, Clone)]
pub struct InMemoryHit<E> {
    /// The stored episode.
    pub episode: E,
    /// Lexical similarity score in `[0, 1]`. Higher is more similar.
    pub score: f32,
    /// Stable storage key assigned by [`InMemoryStore::append`].
    pub key: String,
}

/// Append-only in-memory episode store with deterministic lexical
/// retrieval.
///
/// `E` is the caller's episode payload; only [`Episode::summary`] is
/// inspected during retrieval.
#[derive(Debug)]
pub struct InMemoryStore<E: Episode> {
    inner: Mutex<Inner<E>>,
}

#[derive(Debug)]
struct Inner<E: Episode> {
    next_key: u64,
    /// Insertion-ordered (key, episode) pairs. Order is preserved so
    /// retrieval ties break deterministically by insertion time.
    episodes: Vec<(String, E)>,
}

impl<E: Episode> Default for InMemoryStore<E> {
    fn default() -> Self {
        Self {
            inner: Mutex::new(Inner {
                next_key: 0,
                episodes: Vec::new(),
            }),
        }
    }
}

impl<E: Episode> InMemoryStore<E> {
    /// Create a fresh store wrapped in [`Arc`] for cheap cloning across
    /// hunter / agent tasks.
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Append a new episode and return its storage key.
    ///
    /// Keys are stable across the lifetime of the store and follow the
    /// `ep-{:016x}` template so they sort lexicographically by insertion
    /// order.
    pub async fn append(&self, episode: E) -> Result<String, InMemoryError> {
        let mut inner = self.inner.lock().map_err(|_| InMemoryError::Poisoned)?;
        inner.next_key = inner.next_key.saturating_add(1);
        let key = format!("ep-{:016x}", inner.next_key);
        inner.episodes.push((key.clone(), episode));
        Ok(key)
    }

    /// Return up to `k` episodes most similar to `query`, sorted by
    /// descending lexical score. Hits with score `0.0` are skipped.
    pub async fn retrieve_similar(
        &self,
        query: &str,
        k: usize,
    ) -> Result<Vec<InMemoryHit<E>>, InMemoryError> {
        if k == 0 {
            return Ok(Vec::new());
        }
        let snapshot: Vec<(String, E)> = {
            let inner = self.inner.lock().map_err(|_| InMemoryError::Poisoned)?;
            inner.episodes.clone()
        };
        let mut hits: Vec<InMemoryHit<E>> = snapshot
            .into_iter()
            .filter_map(|(key, ep)| {
                let score = lexical_score(query, ep.summary());
                if score > 0.0 {
                    Some(InMemoryHit {
                        episode: ep,
                        score,
                        key,
                    })
                } else {
                    None
                }
            })
            .collect();
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(k);
        Ok(hits)
    }

    /// Direct lookup by storage key.
    pub async fn get(&self, key: &str) -> Result<E, InMemoryError> {
        let inner = self.inner.lock().map_err(|_| InMemoryError::Poisoned)?;
        inner
            .episodes
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, ep)| ep.clone())
            .ok_or_else(|| InMemoryError::NotFound(key.to_string()))
    }

    /// Number of episodes currently stored.
    pub async fn len(&self) -> Result<usize, InMemoryError> {
        let inner = self.inner.lock().map_err(|_| InMemoryError::Poisoned)?;
        Ok(inner.episodes.len())
    }

    /// Whether the store is empty.
    pub async fn is_empty(&self) -> Result<bool, InMemoryError> {
        Ok(self.len().await? == 0)
    }
}

/// Token-overlap score in `[0, 1]`.
///
/// Returns the fraction of distinct normalized query tokens that appear
/// in `summary`. Normalization is intentionally simple and
/// deterministic: Unicode-aware lowercase via [`str::to_lowercase`] on
/// each whitespace-delimited token, and trim leading/trailing
/// non-alphanumeric characters (a Unicode-aware superset of ASCII
/// punctuation). An empty query yields `0.0`.
fn lexical_score(query: &str, summary: &str) -> f32 {
    let query_tokens = normalized_tokens(query);
    if query_tokens.is_empty() {
        return 0.0;
    }
    let summary_tokens = normalized_tokens(summary);
    let intersection = query_tokens.intersection(&summary_tokens).count() as f32;
    intersection / query_tokens.len() as f32
}

fn normalized_tokens(input: &str) -> HashSet<String> {
    input
        .split_whitespace()
        .map(|token| token.trim_matches(|ch: char| !ch.is_alphanumeric()))
        .filter(|token| !token.is_empty())
        .map(str::to_lowercase)
        .collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[derive(Debug, Clone)]
    struct E(&'static str);

    impl Episode for E {
        fn summary(&self) -> &str {
            self.0
        }
    }

    #[tokio::test]
    async fn append_assigns_unique_ordered_keys() {
        let s = InMemoryStore::<E>::new();
        let k1 = s.append(E("a")).await.unwrap();
        let k2 = s.append(E("b")).await.unwrap();
        assert!(k1 < k2);
    }

    #[tokio::test]
    async fn retrieve_returns_top_k_by_score() {
        let s = InMemoryStore::<E>::new();
        s.append(E("powershell maintenance scheduled task"))
            .await
            .unwrap();
        s.append(E("ddos amplification spike")).await.unwrap();
        let hits = s.retrieve_similar("powershell scheduled", 5).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert!(hits.first().unwrap().episode.0.contains("powershell"));
    }

    #[tokio::test]
    async fn retrieve_skips_zero_score_hits() {
        let s = InMemoryStore::<E>::new();
        s.append(E("alpha bravo")).await.unwrap();
        let hits = s.retrieve_similar("zulu", 5).await.unwrap();
        assert!(hits.is_empty());
    }

    #[tokio::test]
    async fn retrieve_matches_case_insensitively() {
        let s = InMemoryStore::<E>::new();
        s.append(E("PowerShell scheduled task")).await.unwrap();
        let hits = s.retrieve_similar("powershell", 5).await.unwrap();
        assert_eq!(hits.len(), 1);
    }

    #[tokio::test]
    async fn retrieve_trims_simple_punctuation() {
        let s = InMemoryStore::<E>::new();
        s.append(E("powershell, scheduled-task beacon"))
            .await
            .unwrap();
        let hits = s
            .retrieve_similar("powershell scheduled-task", 5)
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
    }

    #[tokio::test]
    async fn retrieve_handles_unicode_case_folding() {
        // Cyrillic case folding requires Unicode-aware lowercase.
        let s = InMemoryStore::<E>::new();
        s.append(E("ПОЛЬЗОВАТЕЛЬ logged in")).await.unwrap();
        let hits = s.retrieve_similar("пользователь", 5).await.unwrap();
        assert_eq!(hits.len(), 1);
    }

    #[tokio::test]
    async fn retrieve_trims_unicode_punctuation() {
        // The trailing 」 is Unicode punctuation, not ASCII; an
        // ASCII-only trim would leave it attached and miss the match.
        let s = InMemoryStore::<E>::new();
        s.append(E("「scheduled-task」 beacon")).await.unwrap();
        let hits = s.retrieve_similar("scheduled-task", 5).await.unwrap();
        assert_eq!(hits.len(), 1);
    }

    #[tokio::test]
    async fn get_returns_not_found_for_unknown_key() {
        let s = InMemoryStore::<E>::new();
        let err = s.get("nope").await.unwrap_err();
        assert!(matches!(err, InMemoryError::NotFound(_)));
    }

    #[tokio::test]
    async fn len_and_is_empty_track_inserts() {
        let s = InMemoryStore::<E>::new();
        assert!(s.is_empty().await.unwrap());
        s.append(E("x")).await.unwrap();
        assert_eq!(s.len().await.unwrap(), 1);
        assert!(!s.is_empty().await.unwrap());
    }

    #[tokio::test]
    async fn k_zero_returns_empty() {
        let s = InMemoryStore::<E>::new();
        s.append(E("alpha")).await.unwrap();
        assert!(s.retrieve_similar("alpha", 0).await.unwrap().is_empty());
    }
}
