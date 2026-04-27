//! [`MemvidStore`]: a [`VectorStoreIndex`] backed by a single `.mv2` file.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use memvid_core::{Memvid, PutOptions, SearchHit, SearchRequest};
use rig::{
    Embed, OneOrMany,
    embeddings::Embedding,
    vector_store::{
        InsertDocuments, VectorSearchRequest, VectorStoreError, VectorStoreIndex,
        request::SearchFilter,
    },
    wasm_compat::WasmCompatSend,
};
use serde::{Deserialize, Serialize};

use crate::error::MemvidError;

/// A persistent, file-backed vector / lexical index over a memvid `.mv2`
/// archive.
///
/// `MemvidStore` is cheap to clone (it shares an `Arc<Mutex<Memvid>>` with
/// every clone) and can be both read from and written to concurrently from
/// multiple async tasks. Writes are serialised through the inner mutex.
///
/// Unlike most rig vector stores, `MemvidStore` is **not** parameterised over
/// an [`EmbeddingModel`]: memvid embeds queries internally using whichever
/// engine its file is configured with (BM25/Tantivy when the `lex` feature is
/// enabled, HNSW + BGE-small when `vec` is enabled). Pass plain text in
/// [`VectorSearchRequest::query`] and let memvid do the rest.
///
/// [`EmbeddingModel`]: rig::embeddings::EmbeddingModel
#[derive(Clone)]
pub struct MemvidStore {
    inner: Arc<Mutex<Memvid>>,
}

impl std::fmt::Debug for MemvidStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemvidStore").finish_non_exhaustive()
    }
}

impl MemvidStore {
    /// Wraps an already-open [`Memvid`] handle.
    pub fn from_memvid(memvid: Memvid) -> Self {
        Self {
            inner: Arc::new(Mutex::new(memvid)),
        }
    }

    /// Begin building a new store. See [`MemvidStoreBuilder`].
    pub fn builder() -> MemvidStoreBuilder {
        MemvidStoreBuilder::default()
    }

    /// Acquire the inner mutex. Returns [`MemvidError::Poisoned`] if a prior
    /// holder of the lock panicked.
    fn lock(&self) -> Result<std::sync::MutexGuard<'_, Memvid>, MemvidError> {
        self.inner.lock().map_err(|_| MemvidError::Poisoned)
    }

    /// Append a UTF-8 text payload to the archive and immediately commit.
    ///
    /// Returns the assigned `frame_id`.
    pub fn put_text(&self, text: &str, options: PutOptions) -> Result<u64, MemvidError> {
        let mut guard = self.lock()?;
        let id = guard.put_bytes_with_options(text.as_bytes(), options)?;
        guard.commit()?;
        Ok(id)
    }

    /// Append a payload without committing. The caller is responsible for
    /// invoking [`MemvidStore::commit`] before a subsequent search will see
    /// the new frame.
    pub fn put_text_uncommitted(
        &self,
        text: &str,
        options: PutOptions,
    ) -> Result<u64, MemvidError> {
        let mut guard = self.lock()?;
        let id = guard.put_bytes_with_options(text.as_bytes(), options)?;
        Ok(id)
    }

    /// Flush any pending writes to disk.
    pub fn commit(&self) -> Result<(), MemvidError> {
        let mut guard = self.lock()?;
        guard.commit()?;
        Ok(())
    }

    /// Run a [`SearchRequest`] directly. Useful for callers that need
    /// memvid-native features (cursors, ACL contexts, etc.) that do not map
    /// onto [`VectorSearchRequest`].
    pub fn search(
        &self,
        request: SearchRequest,
    ) -> Result<memvid_core::SearchResponse, MemvidError> {
        let mut guard = self.lock()?;
        let resp = guard.search(request)?;
        Ok(resp)
    }
}

/// Builder for [`MemvidStore`].
#[derive(Debug, Default)]
pub struct MemvidStoreBuilder {
    path: Option<PathBuf>,
    enable_lex: bool,
}

impl MemvidStoreBuilder {
    /// Path to the `.mv2` file.
    pub fn path<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.path = Some(path.into());
        self
    }

    /// Enable BM25 / Tantivy lexical search on the underlying archive.
    pub fn enable_lex(mut self) -> Self {
        self.enable_lex = true;
        self
    }

    fn require_path(&self) -> Result<&Path, MemvidError> {
        self.path.as_deref().ok_or_else(|| {
            MemvidError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "MemvidStoreBuilder requires a path",
            ))
        })
    }

    fn finish(self, memvid: Memvid) -> Result<MemvidStore, MemvidError> {
        let mut memvid = memvid;
        if self.enable_lex {
            memvid.enable_lex()?;
        }
        Ok(MemvidStore::from_memvid(memvid))
    }

    /// Open an existing `.mv2` file. Errors if the file does not exist.
    pub fn open(self) -> Result<MemvidStore, MemvidError> {
        let path = self.require_path()?.to_path_buf();
        let memvid = Memvid::open(&path)?;
        self.finish(memvid)
    }

    /// Create a new `.mv2` file. Errors if the file already exists.
    pub fn create(self) -> Result<MemvidStore, MemvidError> {
        let path = self.require_path()?.to_path_buf();
        let memvid = Memvid::create(&path)?;
        self.finish(memvid)
    }

    /// Open the file if it exists, otherwise create it.
    pub fn open_or_create(self) -> Result<MemvidStore, MemvidError> {
        let path = self.require_path()?.to_path_buf();
        let memvid = if path.exists() {
            Memvid::open(&path)?
        } else {
            Memvid::create(&path)?
        };
        self.finish(memvid)
    }

    /// Open the file read-only.
    pub fn open_read_only(self) -> Result<MemvidStore, MemvidError> {
        let path = self.require_path()?.to_path_buf();
        let memvid = Memvid::open_read_only(&path)?;
        self.finish(memvid)
    }
}

/// A filter clause supported by [`MemvidStore`].
///
/// Memvid's query model does not support arbitrary boolean predicates;
/// this filter only carries the four restriction parameters that map onto
/// fields of [`SearchRequest`]:
///
/// | Predicate                       | Effect on the search request  |
/// | ------------------------------- | ----------------------------- |
/// | `eq("uri", "...")`              | `request.uri = Some(value)`   |
/// | `eq("scope", "...")`            | `request.scope = Some(value)` |
/// | `eq("as_of_frame", n)`          | `request.as_of_frame`         |
/// | `eq("as_of_ts", n)`             | `request.as_of_ts`            |
///
/// `gt`, `lt`, and `or` are not representable; constructing such a filter
/// produces an error at query time
/// ([`MemvidError::UnsupportedFilter`]).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemvidFilter {
    /// Optional URI prefix restriction.
    pub uri: Option<String>,
    /// Optional logical scope.
    pub scope: Option<String>,
    /// Optional point-in-time frame id.
    pub as_of_frame: Option<u64>,
    /// Optional point-in-time unix-millis timestamp.
    pub as_of_ts: Option<i64>,
    /// Reasons this filter cannot be applied. Populated when the user calls
    /// `gt`, `lt`, `or`, or `eq` with an unknown key.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    invalid: Vec<String>,
}

impl MemvidFilter {
    fn unsupported(reason: impl Into<String>) -> Self {
        Self {
            invalid: vec![reason.into()],
            ..Self::default()
        }
    }

    fn merge(mut self, rhs: Self) -> Self {
        if rhs.uri.is_some() {
            self.uri = rhs.uri;
        }
        if rhs.scope.is_some() {
            self.scope = rhs.scope;
        }
        if rhs.as_of_frame.is_some() {
            self.as_of_frame = rhs.as_of_frame;
        }
        if rhs.as_of_ts.is_some() {
            self.as_of_ts = rhs.as_of_ts;
        }
        self.invalid.extend(rhs.invalid);
        self
    }

    fn into_validated(self) -> Result<Self, MemvidError> {
        if self.invalid.is_empty() {
            Ok(self)
        } else {
            Err(MemvidError::UnsupportedFilter(self.invalid.join("; ")))
        }
    }

    fn apply_to(self, request: &mut SearchRequest) {
        request.uri = self.uri;
        request.scope = self.scope;
        request.as_of_frame = self.as_of_frame;
        request.as_of_ts = self.as_of_ts;
    }
}

fn json_as_string(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::String(s) => Some(s.clone()),
        other => Some(other.to_string()),
    }
}

impl SearchFilter for MemvidFilter {
    type Value = serde_json::Value;

    fn eq(key: impl AsRef<str>, value: Self::Value) -> Self {
        let key = key.as_ref();
        match key {
            "uri" => Self {
                uri: json_as_string(&value),
                ..Self::default()
            },
            "scope" => Self {
                scope: json_as_string(&value),
                ..Self::default()
            },
            "as_of_frame" => match value.as_u64() {
                Some(n) => Self {
                    as_of_frame: Some(n),
                    ..Self::default()
                },
                None => Self::unsupported(format!("as_of_frame must be a u64, got {value}")),
            },
            "as_of_ts" => match value.as_i64() {
                Some(n) => Self {
                    as_of_ts: Some(n),
                    ..Self::default()
                },
                None => Self::unsupported(format!("as_of_ts must be an i64, got {value}")),
            },
            other => Self::unsupported(format!(
                "unsupported filter key '{other}' (allowed: uri, scope, as_of_frame, as_of_ts)"
            )),
        }
    }

    fn gt(key: impl AsRef<str>, _value: Self::Value) -> Self {
        Self::unsupported(format!(
            "memvid does not support gt() on '{}'",
            key.as_ref()
        ))
    }

    fn lt(key: impl AsRef<str>, _value: Self::Value) -> Self {
        Self::unsupported(format!(
            "memvid does not support lt() on '{}'",
            key.as_ref()
        ))
    }

    fn and(self, rhs: Self) -> Self {
        self.merge(rhs)
    }

    fn or(self, rhs: Self) -> Self {
        self.merge(rhs)
            .merge(Self::unsupported("memvid does not support or() in filters"))
    }
}

/// Default snippet size when memvid is asked for context around a hit.
///
/// Tuned to be roughly one paragraph; callers who want different behaviour
/// should call [`MemvidStore::search`] directly with their own
/// [`SearchRequest`].
const DEFAULT_SNIPPET_CHARS: usize = 400;

fn build_search_request(
    query: String,
    samples: u64,
    filter: Option<MemvidFilter>,
) -> Result<SearchRequest, MemvidError> {
    let filter = match filter {
        Some(f) => f.into_validated()?,
        None => MemvidFilter::default(),
    };
    let mut req = SearchRequest {
        query,
        top_k: usize::try_from(samples).unwrap_or(usize::MAX),
        snippet_chars: DEFAULT_SNIPPET_CHARS,
        uri: None,
        scope: None,
        cursor: None,
        #[cfg(feature = "temporal")]
        temporal: None,
        as_of_frame: None,
        as_of_ts: None,
        no_sketch: false,
        acl_context: None,
        acl_enforcement_mode: memvid_core::AclEnforcementMode::default(),
    };
    filter.apply_to(&mut req);
    Ok(req)
}

fn hit_score(hit: &SearchHit) -> f64 {
    match hit.score {
        Some(s) => f64::from(s),
        // Lexical hits often arrive without a numeric score; fall back to
        // rank-derived order-preserving values so callers can still sort.
        None => 1.0 / (hit.rank as f64 + 1.0),
    }
}

impl VectorStoreIndex for MemvidStore {
    type Filter = MemvidFilter;

    async fn top_n<T>(
        &self,
        req: VectorSearchRequest<Self::Filter>,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError>
    where
        T: for<'a> Deserialize<'a> + WasmCompatSend,
    {
        let query = req.query().to_owned();
        let samples = req.samples();
        let filter = req.filter().clone();
        let request =
            build_search_request(query, samples, filter).map_err(VectorStoreError::from)?;

        let response = {
            let mut guard = self
                .inner
                .lock()
                .map_err(|_| VectorStoreError::from(MemvidError::Poisoned))?;
            guard.search(request).map_err(MemvidError::from)?
        };

        let mut out = Vec::with_capacity(response.hits.len());
        for hit in response.hits {
            let score = hit_score(&hit);
            let id = hit.frame_id.to_string();
            let value = serde_json::to_value(&hit).map_err(MemvidError::from)?;
            let doc: T = serde_json::from_value(value).map_err(MemvidError::from)?;
            out.push((score, id, doc));
        }
        Ok(out)
    }

    async fn top_n_ids(
        &self,
        req: VectorSearchRequest<Self::Filter>,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        let query = req.query().to_owned();
        let samples = req.samples();
        let filter = req.filter().clone();
        let request =
            build_search_request(query, samples, filter).map_err(VectorStoreError::from)?;

        let response = {
            let mut guard = self
                .inner
                .lock()
                .map_err(|_| VectorStoreError::from(MemvidError::Poisoned))?;
            guard.search(request).map_err(MemvidError::from)?
        };

        Ok(response
            .hits
            .into_iter()
            .map(|hit| (hit_score(&hit), hit.frame_id.to_string()))
            .collect())
    }
}

impl InsertDocuments for MemvidStore {
    async fn insert_documents<Doc>(
        &self,
        documents: Vec<(Doc, OneOrMany<Embedding>)>,
    ) -> Result<(), VectorStoreError>
    where
        Doc: Serialize + Embed + WasmCompatSend,
    {
        // We deliberately ignore the externally-supplied embeddings: memvid
        // owns its own embedding pipeline and embeds at the segment level.
        // Round-tripping the document through JSON gives us a stable byte
        // payload that is also what `serde_json::from_value::<T>` will
        // recover during search.
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| VectorStoreError::from(MemvidError::Poisoned))?;
        for (doc, _embeddings) in documents {
            let bytes = serde_json::to_vec(&doc).map_err(MemvidError::from)?;
            guard
                .put_bytes_with_options(&bytes, PutOptions::default())
                .map_err(MemvidError::from)?;
        }
        guard.commit().map_err(MemvidError::from)?;
        Ok(())
    }
}
