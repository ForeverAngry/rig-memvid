# rig-memvid Roadmap

This roadmap is the crate-local operating plan for `rig-memvid`. The cross-crate coordination summary lives in the [rig-ecosystem docs](https://github.com/ForeverAngry/rig-ecosystem).

## Role

`rig-memvid` adapts Memvid `.mv2` archives into Rig retrieval and prompt persistence flows. It is the durable memory adapter in the companion stack: episodic frames, structured cards, context candidates, and local-model smoke examples belong here when they depend on Memvid or Rig prompt/vector-store traits.

## Landed

- `MemvidStore` implementing Rig vector-store retrieval and insertion over a shared `Arc<Mutex<Memvid>>`.
- `MemvidPersistHook` for writing prompt turns into the same archive used for retrieval.
- `MemvidFilter`, lifecycle helpers, read-only open, and lex/vector feature wiring.
- `InMemoryStore<E>` deterministic lexical fallback for no-disk tests and offline modes.
- Structured-memory pass-through: memory cards, current memory, preferences, aggregate slots, timelines, and hand-written card insertion.
- `MemoryCardContext` and `CardSelection` for surfacing structured cards as Rig dynamic context.
- Principal-aware persistence and supplemental profile cards for common profile and relationship facts.
- Logic Mesh graph-track pass-through for entity and relationship traversal.
- Ollama and MLX examples that exercise live local-model memory behavior.
- `DemotionHook` integration tracking upstream `rig-memory`: long-tail conversation turns can be demoted into `MemvidStore` as an optional adapter without making `rig-memvid` depend on `rig-memory` for default builds.
- `MemoryCandidate` / `MemoryContextPack` over card-backed `ContextItem`s
  with deterministic supersession by `version_key` + `effective_timestamp`
  ([src/projection.rs](src/projection.rs)), plus `version_key`,
  `effective_timestamp`, `event_date`, and `document_date` provenance on
  card-derived items.
- Shared context-projection provenance keys for retrieval hits, in-memory
   hits, memory cards, and card docs: `source_uri`, `principal`,
   `recorded_at_millis`, `effective_at_millis`, `confidence`,
   `source_frame_id`, `version_key`, and `projection_state` where available.

## Prototype Grade

- Structured cards handle useful profile and relationship facts. Supersession by `version_key` collapses competing versions of the same fact today; stale/conflict surfacing across frames, archive tiers, and unresolved principal-bound recall policy still need hardening.
- Live local-model examples exist, but they must pin intended models and fail loudly on fallback or drift.
- Logic Mesh traversal is exposed, but graph-backed context planning and eval fixtures are still early.

## Next Work

1. Extend `MemoryCandidate` / `MemoryContextPack` beyond cards to frames,
   summaries, graph expansions, and compaction outputs, and surface the
   superseded list to callers/eval harnesses.
2. Add explicit scope and retention-tier provenance once the underlying memvid
   source records expose those concepts, complementing the principal,
   confidence, timestamp, and version fields already projected today.
3. Add stale/conflict surfacing so the host can explain a supersession
   decision (or an unresolved conflict) end-to-end.
4. Pin live Ollama examples to explicit models and fail loudly when the requested model is unavailable.
5. Add eval fixtures that assert selected, omitted, compacted, and used memory. Drives an explicit dev-dependency on [`rig-retrieval-evals`](https://crates.io/crates/rig-retrieval-evals) once its retriever-evals harness lands.
6. Track upstream Rig compaction outputs and wire them into memory candidates once that surface settles.

## Maturity Bar

- The packer explains every omission and can be replayed from fixtures.
- Tests cover ranking, budget, reserve space, separators, relationship expansion, and stale/conflict behavior.
- Live smokes use the intended local model and complement deterministic tests.
- No public API requires callers to depend directly on `memvid-core` for common memory flows.

## Non-Goals

- Do not become a tool router or MCP bridge.
- Do not add a runtime dependency to library dependencies.
- Do not upstream `MemoryGraph`-style traits to Rig until a second backend validates the shape.
- Do not implement encryption-at-rest or ACL-style access policy in the crate. Callers needing those guarantees should layer them at the filesystem (LUKS, FileVault, S3 SSE) or process (Linux capabilities, sandbox) boundary; `MemvidStore` deliberately treats the archive as plaintext that the host environment is expected to protect.
