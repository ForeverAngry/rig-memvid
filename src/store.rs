//! [`MemvidStore`]: a [`VectorStoreIndex`] backed by a single `.mv2` file.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use memvid_core::{AclContext, AclEnforcementMode, Memvid, PutOptions, SearchHit, SearchRequest};
#[cfg(feature = "vec")]
use memvid_core::{LocalTextEmbedder, TextEmbedConfig};
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
/// ## Concurrency
///
/// Every public method on the underlying [`Memvid`] handle — including
/// `search`, `vec_search_with_embedding`, `frame_count`, and the various
/// `put_*` writers — takes `&mut self`. Reads cannot run in parallel with
/// other reads, so the inner lock is a [`Mutex`] rather than an
/// `RwLock`. Workloads that require concurrent reads should open separate
/// read-only handles via [`MemvidStoreBuilder::open_read_only`].
///
/// The lock is [`std::sync::Mutex`] (not `tokio::sync::Mutex`): the crate
/// is intentionally runtime-agnostic and the clippy `await_holding_lock`
/// lint enforces that no `.await` ever happens while a guard is live. Every
/// guard in this module is scope-dropped before any async boundary.
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
    #[cfg(feature = "vec")]
    embedder: Option<Arc<LocalTextEmbedder>>,
    /// Default `snippet_chars` applied to [`VectorStoreIndex`] queries.
    /// Configurable via [`MemvidStoreBuilder::snippet_chars`].
    snippet_chars: usize,
    /// Default ACL context applied to every search. `None` means no ACL
    /// filtering. Configurable via [`MemvidStoreBuilder::acl_context`].
    acl_context: Option<AclContext>,
    /// ACL enforcement mode (`Audit` or `Enforce`).
    acl_enforcement_mode: AclEnforcementMode,
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
            #[cfg(feature = "vec")]
            embedder: None,
            snippet_chars: DEFAULT_SNIPPET_CHARS,
            acl_context: None,
            acl_enforcement_mode: AclEnforcementMode::default(),
        }
    }

    /// Number of frames currently stored in the underlying `.mv2` file.
    pub fn frame_count(&self) -> Result<usize, MemvidError> {
        Ok(self.lock()?.frame_count())
    }

    /// Aggregate statistics for the underlying memory.
    pub fn stats(&self) -> Result<memvid_core::types::frame::Stats, MemvidError> {
        Ok(self.lock()?.stats()?)
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

    /// Whether this store will route writes/queries through a local
    /// embedding model.
    #[cfg(feature = "vec")]
    #[must_use]
    pub fn has_embedder(&self) -> bool {
        self.embedder.is_some()
    }

    /// Encode `text` with the configured embedder, if any.
    #[cfg(feature = "vec")]
    fn encode(&self, text: &str) -> Result<Option<Vec<f32>>, MemvidError> {
        match &self.embedder {
            Some(embedder) => Ok(Some(embedder.encode_text(text)?)),
            None => Ok(None),
        }
    }

    /// Append a UTF-8 text payload to the archive and immediately commit.
    ///
    /// Returns the assigned `frame_id`. When the store has been built with
    /// an embedder (`vec` feature), the text is embedded and stored
    /// alongside its frame so that subsequent
    /// [`VectorStoreIndex::top_n`] calls perform semantic search.
    pub fn put_text(&self, text: &str, options: PutOptions) -> Result<u64, MemvidError> {
        #[cfg(feature = "vec")]
        let embedding = self.encode(text)?;
        let mut guard = self.lock()?;
        #[cfg(feature = "vec")]
        let id = if let Some(emb) = embedding {
            guard.put_with_embedding_and_options(text.as_bytes(), emb, options)?
        } else {
            guard.put_bytes_with_options(text.as_bytes(), options)?
        };
        #[cfg(not(feature = "vec"))]
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
        #[cfg(feature = "vec")]
        let embedding = self.encode(text)?;
        let mut guard = self.lock()?;
        #[cfg(feature = "vec")]
        let id = if let Some(emb) = embedding {
            guard.put_with_embedding_and_options(text.as_bytes(), emb, options)?
        } else {
            guard.put_bytes_with_options(text.as_bytes(), options)?
        };
        #[cfg(not(feature = "vec"))]
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
    ///
    /// # Concurrency
    ///
    /// Acquires the store's inner [`Mutex`] for the duration of the call.
    /// Do **not** invoke this (or any other `MemvidStore` method) from
    /// within a [`crate::WriteTransform`] closure: hook writes already hold
    /// a path through `put_text` and a re-entrant call would deadlock.
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
#[derive(Default)]
pub struct MemvidStoreBuilder {
    path: Option<PathBuf>,
    enable_lex: bool,
    snippet_chars: Option<usize>,
    acl_context: Option<AclContext>,
    acl_enforcement_mode: Option<AclEnforcementMode>,
    #[cfg(feature = "vec")]
    enable_vec: bool,
    #[cfg(feature = "vec")]
    vec_model: Option<String>,
    #[cfg(feature = "vec")]
    embedder: Option<Arc<LocalTextEmbedder>>,
}

impl std::fmt::Debug for MemvidStoreBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut d = f.debug_struct("MemvidStoreBuilder");
        d.field("path", &self.path)
            .field("enable_lex", &self.enable_lex);
        #[cfg(feature = "vec")]
        {
            d.field("enable_vec", &self.enable_vec)
                .field("vec_model", &self.vec_model)
                .field("embedder", &self.embedder.as_ref().map(|_| "<embedder>"));
        }
        d.finish()
    }
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

    /// Number of context characters to capture around each search hit.
    /// Defaults to 400 characters. Applies to queries issued
    /// via [`VectorStoreIndex::top_n`] and the `vec` search path; callers
    /// who need per-query control should use [`MemvidStore::search`]
    /// directly with a hand-built [`SearchRequest`].
    pub fn snippet_chars(mut self, n: usize) -> Self {
        self.snippet_chars = Some(n);
        self
    }

    /// Default [`AclContext`] attached to every search performed through
    /// the [`VectorStoreIndex`] / vector-search interfaces. When unset,
    /// ACL filtering is disabled.
    pub fn acl_context(mut self, ctx: AclContext) -> Self {
        self.acl_context = Some(ctx);
        self
    }

    /// ACL enforcement mode for default-attached contexts. Defaults to
    /// [`AclEnforcementMode::Audit`].
    pub fn acl_enforcement_mode(mut self, mode: AclEnforcementMode) -> Self {
        self.acl_enforcement_mode = Some(mode);
        self
    }

    /// Enable HNSW vector search on the underlying archive.
    ///
    /// Available only when this crate is built with the `vec` feature, which
    /// pulls in `memvid-core/vec` (ONNX Runtime + bundled BGE-small).
    /// Mutually compatible with [`Self::enable_lex`]; both can be on at once
    /// for hybrid retrieval.
    #[cfg(feature = "vec")]
    pub fn enable_vec(mut self) -> Self {
        self.enable_vec = true;
        self
    }

    /// Bind (or validate) the embedding model identifier on the vector
    /// index. See [`memvid_core::Memvid::set_vec_model`].
    #[cfg(feature = "vec")]
    pub fn vec_model(mut self, model: impl Into<String>) -> Self {
        self.vec_model = Some(model.into());
        self
    }

    /// Attach a local text embedder. Writes performed via
    /// [`MemvidStore::put_text`] and queries performed via
    /// [`VectorStoreIndex::top_n`] will be embedded with this model and
    /// routed through memvid's HNSW vector index.
    ///
    /// Implies [`Self::enable_vec`]. If [`Self::vec_model`] has not been
    /// set, the model identifier reported by the embedder is bound
    /// automatically.
    #[cfg(feature = "vec")]
    pub fn embedder(mut self, embedder: LocalTextEmbedder) -> Self {
        if self.vec_model.is_none() {
            self.vec_model = Some(embedder.model_info().name.to_string());
        }
        self.embedder = Some(Arc::new(embedder));
        self.enable_vec = true;
        self
    }

    /// Convenience: attach the default local embedder (BGE-small,
    /// 384-dimensional). The model is loaded from
    /// [`TextEmbedConfig::default`]'s on-disk cache; if absent and
    /// `offline` is `false` it will be downloaded.
    #[cfg(feature = "vec")]
    pub fn with_default_embedder(self) -> Result<Self, MemvidError> {
        let embedder = LocalTextEmbedder::new(TextEmbedConfig::bge_small())?;
        Ok(self.embedder(embedder))
    }

    /// Convenience: attach a local embedder built from an explicit
    /// [`TextEmbedConfig`].
    #[cfg(feature = "vec")]
    pub fn with_embedder_config(self, config: TextEmbedConfig) -> Result<Self, MemvidError> {
        let embedder = LocalTextEmbedder::new(config)?;
        Ok(self.embedder(embedder))
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
        #[cfg(feature = "vec")]
        {
            if self.enable_vec {
                memvid.enable_vec()?;
            }
            if let Some(model) = self.vec_model.as_deref() {
                memvid.set_vec_model(model)?;
            }
        }
        #[cfg_attr(not(feature = "vec"), allow(unused_mut))]
        let mut store = MemvidStore::from_memvid(memvid);
        if let Some(s) = self.snippet_chars {
            store.snippet_chars = s;
        }
        if let Some(ctx) = self.acl_context {
            store.acl_context = Some(ctx);
        }
        if let Some(mode) = self.acl_enforcement_mode {
            store.acl_enforcement_mode = mode;
        }
        #[cfg(feature = "vec")]
        {
            store.embedder = self.embedder;
        }
        Ok(store)
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
/// this filter only carries the restriction parameters that map onto
/// fields of [`SearchRequest`]:
///
/// | Predicate                       | Effect on the search request    |
/// | ------------------------------- | ------------------------------- |
/// | `eq("uri", "...")`              | `request.uri = Some(value)`     |
/// | `eq("scope", "...")`            | `request.scope = Some(value)`   |
/// | `eq("as_of_frame", n)`          | `request.as_of_frame`           |
/// | `eq("as_of_ts", n)`             | `request.as_of_ts`              |
/// | `eq("cursor", "...")`           | `request.cursor` (pagination)   |
/// | `eq("no_sketch", true/false)`   | disable sketch pre-filtering    |
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
    /// Optional pagination cursor (opaque token returned by a prior search).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cursor: Option<String>,
    /// If `Some(true)`, disable the sketch pre-filter for this query.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub no_sketch: Option<bool>,
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
        if rhs.cursor.is_some() {
            self.cursor = rhs.cursor;
        }
        if rhs.no_sketch.is_some() {
            self.no_sketch = rhs.no_sketch;
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
        if let Some(c) = self.cursor {
            request.cursor = Some(c);
        }
        if let Some(b) = self.no_sketch {
            request.no_sketch = b;
        }
    }
}

fn json_as_string(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::String(s) => Some(s.clone()),
        other => Some(other.to_string()),
    }
}

/// Coerce a JSON value into an `i64` for `as_of_ts`.
///
/// Accepts integer JSON numbers and integer-valued floats (which is the
/// default representation for many JSON producers).
fn as_of_ts_from_value(value: &serde_json::Value) -> Option<i64> {
    if let Some(n) = value.as_i64() {
        return Some(n);
    }
    let f = value.as_f64()?;
    if f.is_finite() && f.fract() == 0.0 && f >= i64::MIN as f64 && f <= i64::MAX as f64 {
        Some(f as i64)
    } else {
        None
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
            "as_of_ts" => match as_of_ts_from_value(&value) {
                Some(n) => Self {
                    as_of_ts: Some(n),
                    ..Self::default()
                },
                None => Self::unsupported(format!("as_of_ts must be an i64, got {value}")),
            },
            "cursor" => Self {
                cursor: json_as_string(&value),
                ..Self::default()
            },
            "no_sketch" => match value.as_bool() {
                Some(b) => Self {
                    no_sketch: Some(b),
                    ..Self::default()
                },
                None => Self::unsupported(format!("no_sketch must be a bool, got {value}")),
            },
            other => Self::unsupported(format!(
                "unsupported filter key '{other}' (allowed: uri, scope, as_of_frame, as_of_ts, \
                 cursor, no_sketch)"
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

    fn or(self, _rhs: Self) -> Self {
        // Memvid's filter model is a flat conjunction; representing a true
        // disjunction would require widening the search request. Discard
        // both operands and return a bare unsupported marker — the
        // resulting filter is rejected by `into_validated()` regardless.
        let _ = self;
        Self::unsupported("memvid does not support or() in filters")
    }
}

/// Default snippet size when memvid is asked for context around a hit.
///
/// Tuned to be roughly one paragraph; callers who want different behaviour
/// should call [`MemvidStore::search`] directly with their own
/// [`SearchRequest`].
const DEFAULT_SNIPPET_CHARS: usize = 400;

/// Hard cap applied to `samples` (a.k.a. `top_k`) so callers cannot request
/// `usize::MAX` worth of hits — both as a defensive measure on 32-bit
/// targets where `u64 -> usize` may saturate, and to keep memvid from
/// allocating absurdly large result vectors.
const MAX_SAMPLES: usize = 1024;

fn samples_to_top_k(samples: u64) -> usize {
    let n = usize::try_from(samples).unwrap_or(MAX_SAMPLES);
    n.min(MAX_SAMPLES)
}

fn build_search_request(
    query: String,
    samples: u64,
    snippet_chars: usize,
    filter: Option<MemvidFilter>,
    acl_context: Option<AclContext>,
    acl_enforcement_mode: AclEnforcementMode,
) -> Result<SearchRequest, MemvidError> {
    let filter = match filter {
        Some(f) => f.into_validated()?,
        None => MemvidFilter::default(),
    };
    let mut req = SearchRequest {
        query,
        top_k: samples_to_top_k(samples),
        snippet_chars,
        uri: None,
        scope: None,
        cursor: None,
        #[cfg(feature = "temporal")]
        temporal: None,
        as_of_frame: None,
        as_of_ts: None,
        no_sketch: false,
        acl_context,
        acl_enforcement_mode,
    };
    filter.apply_to(&mut req);
    Ok(req)
}

fn hit_score(hit: &SearchHit) -> f64 {
    match hit.score {
        Some(s) => f64::from(s),
        // Lexical hits often arrive without a numeric score; fall back to
        // rank-derived order-preserving values so callers can still sort.
        // `hit.rank` is `usize`; cap at `u32::MAX` before promoting to f64
        // to avoid lossy `as` casts that clippy would otherwise reject.
        None => {
            let rank = u32::try_from(hit.rank).unwrap_or(u32::MAX);
            1.0 / (f64::from(rank) + 1.0)
        }
    }
}

#[cfg(feature = "vec")]
fn ensure_vec_filter_supported(filter: &MemvidFilter) -> Result<(), MemvidError> {
    if filter.uri.is_some() {
        return Err(MemvidError::UnsupportedFilter(
            "`uri` filter is not supported when querying through the embedder; use lex search"
                .into(),
        ));
    }
    if filter.as_of_frame.is_some() || filter.as_of_ts.is_some() {
        return Err(MemvidError::UnsupportedFilter(
            "point-in-time filters (`as_of_frame`, `as_of_ts`) are not supported under vector \
             search; use lex or `MemvidStore::search` directly"
                .into(),
        ));
    }
    Ok(())
}

impl MemvidStore {
    /// Run an embedding-driven search through memvid's HNSW index.
    /// Pre-validated by the caller; returns the raw memvid response.
    #[cfg(feature = "vec")]
    fn vec_search(
        &self,
        query: &str,
        samples: u64,
        filter: &MemvidFilter,
    ) -> Result<memvid_core::SearchResponse, MemvidError> {
        let embedder = self
            .embedder
            .as_ref()
            .ok_or_else(|| MemvidError::UnsupportedFilter("no embedder configured".into()))?;
        let embedding = embedder.encode_text(query)?;
        let top_k = samples_to_top_k(samples);
        let mut guard = self.lock()?;
        let resp = if self.acl_context.is_some() {
            guard.vec_search_with_embedding_acl(
                query,
                &embedding,
                top_k,
                self.snippet_chars,
                filter.scope.as_deref(),
                self.acl_context.as_ref(),
                self.acl_enforcement_mode,
            )?
        } else {
            guard.vec_search_with_embedding(
                query,
                &embedding,
                top_k,
                self.snippet_chars,
                filter.scope.as_deref(),
            )?
        };
        Ok(resp)
    }
}

impl MemvidStore {
    /// Internal: dispatch a `VectorSearchRequest` to either the embedder-driven
    /// vector path (if a local embedder is configured) or the lex/raw search
    /// path. Centralises the `cfg(feature = "vec")` plumbing so the public
    /// `VectorStoreIndex` methods stay small and free of duplication.
    fn run_search(
        &self,
        query: String,
        samples: u64,
        filter: Option<MemvidFilter>,
    ) -> Result<memvid_core::SearchResponse, MemvidError> {
        #[cfg(feature = "vec")]
        {
            if self.embedder.is_some() {
                let validated = match filter {
                    Some(f) => f.into_validated()?,
                    None => MemvidFilter::default(),
                };
                ensure_vec_filter_supported(&validated)?;
                return self.vec_search(&query, samples, &validated);
            }
        }
        let request = build_search_request(
            query,
            samples,
            self.snippet_chars,
            filter,
            self.acl_context.clone(),
            self.acl_enforcement_mode,
        )?;
        let mut guard = self.lock()?;
        Ok(guard.search(request)?)
    }
}

impl VectorStoreIndex for MemvidStore {
    type Filter = MemvidFilter;

    /// Run a search and deserialise each hit's JSON representation into `T`.
    ///
    /// # Contract
    ///
    /// The type `T` must be deserialisable from a [`SearchHit`] JSON object —
    /// i.e. either `T = SearchHit` itself, or a struct whose fields are a
    /// subset of `SearchHit`'s public fields (`frame_id`, `text`, `score`,
    /// `metadata`, …). Use `serde_json::Value` for an opaque view.
    ///
    /// **This method does not round-trip user-defined document types.**
    /// If you persisted JSON documents through [`InsertDocuments`] and want
    /// them back, use [`VectorStoreIndex::top_n_ids`] for the frame ids and
    /// then [`MemvidStore::search`] for full-fidelity access via the
    /// memvid-native [`SearchRequest`] API.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use memvid_core::SearchHit;
    /// use rig::vector_store::{
    ///     VectorSearchRequest, VectorStoreIndex,
    ///     request::VectorSearchRequestBuilder,
    /// };
    /// use rig_memvid::{MemvidFilter, MemvidStore};
    ///
    /// # async fn run(store: MemvidStore) -> anyhow::Result<()> {
    /// let req: VectorSearchRequest<MemvidFilter> =
    ///     VectorSearchRequestBuilder::<MemvidFilter>::default()
    ///         .query("hello")
    ///         .samples(5)
    ///         .build();
    /// let hits: Vec<(f64, String, SearchHit)> = store.top_n(req).await?;
    /// # Ok(())
    /// # }
    /// ```
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

        let response = self.run_search(query, samples, filter)?;

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

        let response = self.run_search(query, samples, filter)?;

        Ok(response
            .hits
            .into_iter()
            .map(|hit| (hit_score(&hit), hit.frame_id.to_string()))
            .collect())
    }
}

impl InsertDocuments for MemvidStore {
    /// Persist `documents` into the underlying `.mv2` file.
    ///
    /// **Note:** caller-supplied embeddings are intentionally ignored.
    /// On the lex-only path the document JSON is written as bytes and
    /// embeddings are dropped. When this store is configured with a
    /// local embedder (`vec` feature) every document is **re-embedded**
    /// with that model so memvid's vector index stays consistent with its
    /// bound model identifier.
    async fn insert_documents<Doc>(
        &self,
        documents: Vec<(Doc, OneOrMany<Embedding>)>,
    ) -> Result<(), VectorStoreError>
    where
        Doc: Serialize + Embed + WasmCompatSend,
    {
        // We deliberately ignore the externally-supplied embeddings (rig
        // computes them with its own model, but memvid validates the
        // dimension against its bound model and would reject mismatches).
        // When this store has its own embedder, embed each document with
        // the local model. Round-tripping the document through JSON gives
        // us a stable byte payload that `serde_json::from_value::<T>` can
        // recover during search.
        #[cfg(feature = "vec")]
        let local_embedder = self.embedder.clone();
        let mut prepared: Vec<(Vec<u8>, Option<Vec<f32>>)> = Vec::with_capacity(documents.len());
        for (doc, _embeddings) in documents {
            let bytes = serde_json::to_vec(&doc).map_err(MemvidError::from)?;
            #[cfg(feature = "vec")]
            let emb = match &local_embedder {
                Some(embedder) => {
                    // `serde_json::to_vec` always returns valid UTF-8, so this
                    // path is fully infallible today. Use `from_utf8` (not
                    // `from_utf8_unchecked`) to keep the invariant explicit:
                    // if a future refactor swaps the encoder, we surface the
                    // problem as a typed error instead of silently embedding
                    // an empty string.
                    let text = std::str::from_utf8(&bytes).map_err(|e| {
                        MemvidError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, e))
                    })?;
                    Some(embedder.encode_text(text).map_err(MemvidError::from)?)
                }
                None => None,
            };
            #[cfg(not(feature = "vec"))]
            let emb: Option<Vec<f32>> = None;
            prepared.push((bytes, emb));
        }

        let mut guard = self
            .inner
            .lock()
            .map_err(|_| VectorStoreError::from(MemvidError::Poisoned))?;
        for (bytes, emb) in prepared {
            match emb {
                Some(embedding) => {
                    guard
                        .put_with_embedding_and_options(&bytes, embedding, PutOptions::default())
                        .map_err(MemvidError::from)?;
                }
                None => {
                    guard
                        .put_bytes_with_options(&bytes, PutOptions::default())
                        .map_err(MemvidError::from)?;
                }
            }
        }
        guard.commit().map_err(MemvidError::from)?;
        Ok(())
    }
}
