# Changelog

<!-- markdownlint-disable MD024 -->

All notable changes to `rig-memvid` will be documented in this file.

## [Unreleased]

## [0.2.0] - 2026-05-19

### Breaking

- **`MemoryConfig` is now `#[non_exhaustive]`.** Construct it through
  the new fluent [`MemoryConfig::builder()`] or by starting from
  `MemoryConfig::default()` and mutating fields. Struct-literal
  construction (`MemoryConfig { … }`) outside this crate is no longer
  allowed — that is the SemVer-major break and why this is `0.2.0`
  rather than `0.1.6`. New fields will land additively from here.
- **`MemvidFrameMetadata` is now `#[non_exhaustive]`.** Decoded the
  same way (`serde_json::from_value`); direct struct-literal
  construction outside this crate is no longer allowed — downstream
  callers must deserialise (or avoid constructing it directly). This
  is forward-compatible: new fields can ship without another major
  bump.

### Added

- **`WriteFailure` policy on `MemoryConfig`.** New
  `MemoryConfig::on_write_failure` field controls what
  `MemvidPersistHook` does when a frame fails to write. Defaults to
  `WriteFailure::Warn` (log + continue — matches pre-0.2 behaviour).
  `WriteFailure::Halt` logs at `ERROR` and returns
  `HookAction::terminate` on the next hook return point so the agent
  loop stops. `WriteFailure::Custom(Arc<dyn Fn(...) -> ...>)` accepts
  a caller-provided callback for metrics or alerting and returns a
  `WriteFailureAction` to keep or halt the turn. The matching
  `WriteFailurePhase` enum distinguishes `Put`, `PutCard`, and
  `Commit` failures.
- **`MemoryConfigBuilder`.** Fluent builder for every `MemoryConfig`
  field. Re-exported from the crate root.

### Changed

- **`clean_clause` strips common corporate-entity suffixes.** Card
  values now drop trailing `Inc`, `Corp`, `LLC`, `Ltd`, `Co`,
  `Company`, `Corporation`, `Incorporated`, and `Limited` so an
  extractor seeing `Acme Corp.` materialises a card value of `Acme`.
  `engine_version` for principal-rules cards bumps from `"1"` to
  `"2"` so consumers can detect the normalised values. Existing
  cards on disk are unchanged.
- **`MemoryGraph::cards_for_query`.** New trait method with a
  default implementation; backends with index support can override
  to filter inside their own locking. `MemvidStore` overrides to
  filter behind the inner mutex and clone only matching cards,
  removing the per-query full-archive snapshot that
  `select_entity_mentions` previously paid.
- **`MemvidFilter::is_valid` / `MemvidFilter::errors`.** Inspect
  validity programmatically before issuing a search; pairs with the
  existing `MemvidError::UnsupportedFilter` return path.
  `SearchFilter::or` now also logs a `tracing::warn!` so silent
  rejections are observable.
- **Optional `observe` feature.** Pulls `rig-tap` as a runtime
  dependency and emits structured observability events from three tap
  points: `memory.frame_written` whenever the persist hook,
  compactor, or demotion hook lands a new (non-dedup) frame;
  `context.compacted` from `MemvidStoringCompactor::compact` after the
  inner compactor's summary is persisted; and `memory.demoted` from
  `MemvidDemotionHook::on_demote` after the batch commits. The feature
  is off by default — the standard build is byte-identical to the
  prior release.
- **`MemoryConfig::observe_conversation_id`.** New optional field
  carrying the conversation ID stamped on observe events emitted by
  `MemvidPersistHook`. Decouples telemetry correlation from memvid's
  URI-prefix `scope`. When unset, the hook still falls back to `scope`
  then `"default"`, so existing setups behave unchanged.

### Changed

- The `chatbot_with_memory_ollama` example now defaults to profile-memory
  behaviour: `MEMVID_PRINCIPAL=User` and `MEMVID_PERSIST_ASSISTANT=false`.
  First-person user turns bind to a stable principal entity and assistant
  paraphrases are excluded from recall. Set `MEMVID_PRINCIPAL=` (empty
  string) to opt out of principal binding and fall back to entity-mention
  card selection. Set `MEMVID_PERSIST_ASSISTANT=1` to re-enable full
  transcript archiving.

- **`rig-core` dependency bumped from `0.36.0` to `0.37.0`.** Picks up
  PR [#1748](https://github.com/0xPlaygrounds/rig/pull/1748) which
  introduces the `Compactor` and `DemotionHook` memory traits this
  release wires into the Memvid surface. We depend on `rig-core`
  directly but rename it back to `rig` in [Cargo.toml](Cargo.toml) so
  the historic `use rig::...` import paths across this crate continue
  to work unchanged. Downstream consumers see no change to the
  published surface.
- **`rig-compose` dependency bumped from `0.2.0` to `0.3`.** The
  optional `context-projection` feature and dev fixtures now align with
  the current companion-kernel release.
- **MSRV bumped from 1.88 to 1.89.** Required by `memvid-core`'s
  `wide`/`safe_arch` SIMD dependencies, which moved their MSRV to 1.89
  across the entire 1.x line. Pinning is not possible: `memvid-core`
  requires `wide = "1"` and no published `wide` 1.x supports 1.88.

### Added

- **`context-projection` feature (off by default; optional
  `rig-compose` dep).** New `projection` module exposes an
  `IntoContextItem` trait plus `search_hits_to_context_items` and
  `inmem_hits_to_context_items` helpers that project
  `memvid_core::SearchHit` and `InMemoryHit<E>` into
  `rig_compose::ContextItem`s tagged with `ContextSourceKind::Memory`.
  Each item carries rank, score, and a structured `provenance` JSON
  object (`resource`, `frame_id`/`key`, `uri`, `range`, `matches`,
  optional `title`/`chunk_range`/`metadata`) mirroring the
  `rig-resources` projection vocabulary so coordinators can fold memvid
  recall, resource lookups, and tool results into a single bounded
  context pack. Missing engine scores fall back to `1 / (rank + 1)` so
  packers that key off `score` alone stay monotonic. Projection unit tests
  plus the module doctest live alongside the module.
- `context-projection` now also projects structured memory cards via
  `memory_cards_to_context_items` and `card_docs_to_context_items`.
  `MemoryCard` projection reuses the compact `MemoryCardContext` card
  rendering, scores from extractor confidence when present, falls back
  to `1 / (rank + 1)`, and records card provenance (`entity`, `slot`,
  `kind`, `polarity`, `source_frame_id`, `source_uri`, `engine`,
  `confidence`, `schema_version`) for downstream packers and evals.

- **Compaction integration with `rig-core` memory traits** behind a new
  optional `compaction` feature (off by default; pulls
  `rig-memory = "0.1"`). Two primitives:
  - `MemvidDemotionHook` implements `rig::memory::DemotionHook` and
    drains messages evicted from an active conversation window into a
    shared `MemvidStore`. Honours `MemoryConfig` (`WritePolicy`,
    `default_tags`, `scope`, `commit_each_turn`, `auto_tag`,
    `extract_dates`, `extract_triplets`) and tags every persisted frame
    with `kind = "demoted_message"` plus the `conversation_id`.
  - `MemvidStoringCompactor<C>` decorates any `rig::memory::Compactor`
    (e.g. `rig_memory::TemplateCompactor` or an LLM-backed compactor)
    and persists each produced summary into `MemvidStore` as a
    `kind = "compaction_summary"` frame before returning the artifact
    to the composing `CompactingMemory` adapter. Preserves the inner
    `Artifact` type so callers see no API surface change.
  Integration tests in
  [tests/demotion_hook.rs](tests/demotion_hook.rs) and
  [tests/storing_compactor.rs](tests/storing_compactor.rs).
- New `simd` feature (enabled by default) that forwards
  `memvid-core/simd` and restores the upstream-default vector kernels
  that were silently being dropped by our `default-features = false`
  setting on the `memvid-core` dep. Pulls in `wide` (pure Rust, no MSRV
  or platform cost). `vec` and `api_embed` chain `simd` so any build
  that performs embedding-based retrieval gets the SIMD path regardless
  of `default-features`. At the current `MemvidStore::top_n` boundary
  the visible per-query delta is below measurement noise — embedding
  the query and deserializing hits dominate — so this is a
  default-restoration / future-proofing change rather than a measured
  perf win on today's `memvid-core`. See
  [examples/bench_vec_search.rs](examples/bench_vec_search.rs) for a
  reproducer.
- Add crate-local `ROADMAP.md` documenting maturity status, next work, and
  non-goals for the Memvid memory adapter.
- `MemoryGraph` trait — backend-agnostic read-side abstraction over
  structured-memory stores (entity / slot / value cards with versioning,
  polarity, provenance). `MemvidStore` is the first impl; the trait
  exists so the same `MemoryCardContext` can wrap a Postgres, SQLite, or
  Neo4j-backed cards table behind a single interface. Phase 1 lives
  in-crate; Phase 2 will be an upstream PR to `rig-core` next to
  `VectorStoreIndex` once a second backend lands.
- `MemoryCardContext` is now generic: `MemoryCardContext<G: MemoryGraph>`,
  defaulting to `G = MemvidStore` so existing call sites keep compiling.
- Logic-Mesh (graph track) pass-through on `MemvidStore`:
  `mesh_node_count`, `mesh_edge_count`, `find_entity`, `frame_entities`,
  `entities_by_kind`, `follow_relationship`. Surfaces memvid's typed
  entity-relationship graph (auto-populated when frames are written with
  `PutOptions::extract_triplets`) for `who-reports-to-whom`-style
  traversal queries without a direct `memvid-core` dependency.
- Re-exports: `EntityKind`, `MeshNode`, `MeshEdge`, `FollowResult`,
  `LogicMeshStats`, `SearchHitEntity` from `memvid-core`.
- `MemoryConfig::principal`: optional user identity binding for the
  persistence hook. When set, first-person user turns are lightly
  rewritten before memvid extraction (`I` / `my` / `I'm` -> the named
  principal), improving card entity coalescing for chatbot memory.
- `MemoryConfig::persist_assistant`: opt out of writing assistant
  responses when the archive should capture user profile facts without
  assistant paraphrase noise. Defaults to `true` for backwards
  compatibility.
- `MemoryConfig::supplemental_profile_cards`: principal-aware deterministic
  profile / relationship-card extraction for common facts memvid can miss,
  starting with allergy / avoidance statements (`Alice is allergic to
  peanuts` -> `profile alice/allergy = peanuts`) and simple manager /
  reporting statements (`Bob is Alice's manager at Acme. He reports to
  Carol, the VP.` -> `relationship alice/manager = Bob`,
  `relationship bob/reports_to = Carol`, `profile carol/title = VP`).
  Defaults to `true` and is a no-op unless `principal` is set.
- `MemoryCardContext` + `CardSelection`: a `VectorStoreIndex`-shaped view
  over a `MemvidStore`'s structured-memory track. Wire as a second
  `dynamic_context(n, _)` to surface entity / slot / value cards
  alongside episodic frames, with no agent-side cooperation required.
  Selection strategies: `EntityMentions` (default; case-insensitive
  word-boundary entity match against the query), `RecentCards`,
  `ForPrincipal(entity)`, `PreferencesFor(entities)`. Synthetic
  recency-based score keeps Rig's downstream sorting stable.
- `CardDoc` wire-format (re-exported) for the per-hit payload.
- `MemvidStore::all_memory_cards` snapshot accessor.
- `chatbot_with_memory_ollama` example now stacks
  `dynamic_context(8, MemoryCardContext::new(store, EntityMentions))`
  on top of the frame-based context.
- `livetest_relationships_mlx` example exercises the structured-memory
  relationship scenario against an OpenAI-compatible `mlx-lm` server,
  defaulting to `LiquidAI/LFM2.5-1.2B-Thinking-MLX-8bit` for Apple
  Silicon validation.

### Changed

- `MemoryCardContext` now ranks selected cards by deterministic query/card
  relevance before applying the result limit, using recency only as a
  tie-breaker. Principal-bound contexts still recall the broad profile,
  but `where` queries prefer `location` cards, food-safety queries prefer
  `allergy` cards, and preference queries prefer preference cards.
- `CardSelection::ForPrincipal` now expands one hop through selected
  relationship-card values, so a principal card such as
  `alice/manager = Bob` can also surface Bob's own cards for follow-up
  manager/reporting questions.
- `MemoryCardContext` now renders common structured slots as compact
  natural-language facts (`alice lives in Berlin`, `bob's manager =
  Carol`, `alice is allergic to peanuts`) so smaller local models use
  card context more reliably.

### Added

- Structured-memory pass-through on `MemvidStore`: `memory_card_count`,
  `entity_memories`, `current_memory`, `entity_preferences`,
  `aggregate_memory_slot`, `memory_timeline`, `put_memory_card`. Surfaces
  memvid's `MemoryCard` track (Subject-Predicate-Object cards extracted
  from frames when `PutOptions::extract_triplets` is on) without
  requiring callers to take a direct `memvid-core` dependency.
- Re-exports: `MemoryCard`, `MemoryCardId`, `MemoryKind`, `Polarity`,
  `VersionRelation` from `memvid-core`.
- `MemoryConfig` gains `auto_tag`, `extract_dates`, `extract_triplets`
  fields (all default `true`, matching `memvid-core`'s `PutOptions`
  defaults). Lets the persistence hook opt out of memvid's automatic
  tagging / date extraction / SPO-triplet extraction without rebuilding
  `PutOptions` by hand.
- `chatbot_with_memory_ollama` example: new REPL commands
  `/entity <name>`, `/prefs <name>`, `/slot <entity> <slot>`, and an
  augmented `/stats` that also reports `memory_cards`.

## [0.1.5](https://github.com/ForeverAngry/rig-memvid/compare/v0.1.4...v0.1.5) - 2026-05-12

### Added

- *(features)* Restore memvid-core simd default ([#7](https://github.com/ForeverAngry/rig-memvid/pull/7))

### Documentation

- Remove retired repo references

## [0.1.4](https://github.com/ForeverAngry/rig-memvid/compare/v0.1.3...v0.1.4) - 2026-05-06

### CI

- Serialize windows test jobs

## [0.1.3](https://github.com/ForeverAngry/rig-memvid/compare/v0.1.2...v0.1.3) - 2026-05-06

### Added

- Add in-memory lexical store

### Fixed

- Skip duplicate windows api_embed reopen test

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
