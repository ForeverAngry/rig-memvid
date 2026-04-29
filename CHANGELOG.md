# Changelog

All notable changes to `rig-memvid` will be documented in this file.

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
