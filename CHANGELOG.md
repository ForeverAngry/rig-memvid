# Changelog

All notable changes to `rig-memvid` will be documented in this file.

## [Unreleased]

### Added

- `inmem` module: `Episode` trait + generic `InMemoryStore<E>` with
  deterministic lexical retrieval. The no-disk companion to
  `MemvidStore` for tests, examples, and offline modes that don't want
  to spin up a `.mv2` archive.
- `InMemoryStore` lexical retrieval now normalizes tokens with
  Unicode-aware lowercase and trims leading/trailing non-alphanumeric
  characters (a Unicode-aware superset of ASCII punctuation), so
  non-ASCII summaries and queries no longer fall through the offline
  fallback while it remains deterministic and dependency-free.

## [0.1.2](https://github.com/ForeverAngry/rig-memvid/compare/v0.1.1...v0.1.2) - 2026-04-30

### Changed

- *(store)* Extract run_search; add samples cap, safer casts, doc contract

### Documentation

- Add Copilot/AGENTS instructions and scoped test/example rules
- Add Copilot instructions for rig-memvid project

### Changed

- `MemvidPersistHook` now writes the configured `MemoryConfig.scope` into
  `PutOptions.uri` so that `MemvidFilter::eq("scope", ...)` (which memvid
  treats as a URI prefix filter) actually narrows queries to hook-written
  frames. The scope is also stashed under `extra_metadata["scope"]` for
  introspection.
- `top_n` / `top_n_ids` no longer duplicate their lex/vec branches; both go
  through a single `MemvidStore::run_search` helper.
- `samples` is now clamped to a sensible upper bound (1024) before being
  passed to memvid as `top_k`, preventing accidental `usize::MAX` requests.
- `MemvidFilter::eq("as_of_ts", ...)` accepts integer-valued JSON floats
  in addition to JSON integers.

### Documented

- `top_n`'s deserialization contract: `T` must match `SearchHit` (or a
  structural subset / `serde_json::Value`). User-defined document types
  written through `InsertDocuments` are not round-tripped here; use
  `MemvidStore::search` for full-fidelity raw access.
- `MemvidStore::search` reentrancy / deadlock guidance.
- `From<MemvidError> for VectorStoreError` collapse semantics.
- `WritePolicy::Custom` returning `Some("")` is treated as `None`.

### Internal

- Stricter UTF-8 handling in `InsertDocuments` (no silent fallback to the
  empty string when embedding).
- Safer rank-derived score computation (no lossy `usize as f64` cast).
- Simplified `MemvidFilter::or` to discard merged operands (the resulting
  filter is rejected at validation time anyway).
- Shared `tests/common/mod.rs` helper for lex-store bring-up.
- `wasm` targets now emit a `compile_error!` with a clear message instead
  of an empty crate body.

### Tests

- Scope filter retrieves frames written through `MemvidPersistHook`.
- `samples = u64::MAX` does not panic.
- `as_of_ts` accepts integer-valued floats.
- `top_n::<SearchHit>` round-trips memvid hits.

### Examples

- Initialise `tracing-subscriber` so library warnings (e.g. failed persist
  writes) are visible on stderr.

## [0.1.1](https://github.com/ForeverAngry/rig-memvid/compare/v0.1.0...v0.1.1) - 2026-04-29

### Added

- Granular error mapping, cursor/no_sketch filters, ACL pass-through
- *(store)* Configurable snippet_chars, frame_count/stats accessors
- Add reqwest dependency and examples for inspecting memory and running chatbot with Ollama

### CI

- Fix CI matrix and bump MSRV to 1.88
- Add release-plz semver automation

### Fixed

- Gate lex smoke tests behind `lex` feature

### Tests

- Add integration test suite covering filters, lifecycle, errors

## [0.1.0] - Unreleased

### Added

- Initial release.
- `MemvidStore` implementing `rig_core::vector_store::VectorStoreIndex` and
  `InsertDocuments`, backed by a single-file `.mv2` memvid archive.
- `MemvidFilter` supporting `uri`, `scope`, `as_of_frame`, and `as_of_ts`
  predicates via the `SearchFilter` trait.
- `MemvidPersistHook` implementing `rig_core::agent::prompt_request::hooks::PromptHook`
  for automatic per-turn persistence of user prompts and assistant responses.
- `WritePolicy` (`Disabled`, `Raw`, `Custom`) controlling persistence behaviour.
- Feature flags: `lex` (default), `temporal`, `encryption`.

### Notes

- v0.1 ships lexical (BM25/Tantivy) search only. The `vec` and `api-embed`
  backends are deferred until `memvid-core` and `rig-fastembed` agree on an
  `ort` major version (currently `=2.0.0-rc.10` vs `=2.0.0-rc.9`).
