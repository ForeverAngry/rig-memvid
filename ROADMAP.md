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

## Prototype Grade

- Memory candidates/context packs are emerging, but they are not fully collapsed onto `rig-compose` `ContextItem` / `ContextPack` yet.
- Structured cards handle useful profile and relationship facts, but stale/conflict handling, supersession policy, archive tiers, and principal-bound recall policy need hardening.
- Live local-model examples exist, but they must pin intended models and fail loudly on fallback or drift.
- Logic Mesh traversal is exposed, but graph-backed context planning and eval fixtures are still early.

## Next Work

1. Land and stabilize `MemoryCandidate` / `MemoryContextPack` over frames, cards, summaries, graph expansions, and compaction outputs.
2. Project memory candidates into `rig-compose` `ContextItem` with source kind, principal, timestamp, source URI, scope, confidence, supersession state, and retention tier.
3. Add stale/conflict handling so old facts do not silently compete with newer facts as equal context.
4. Pin live Ollama examples to explicit models and fail loudly when the requested model is unavailable.
5. Add eval fixtures that assert selected, omitted, compacted, and used memory.
6. Track upstream Rig compaction outputs and wire them into memory candidates once that surface settles.
7. Revisit upstream `rig-memory` after PR 1756 is released: evaluate a
   `DemotionHook` / long-tail memory adapter that writes demoted conversation
   turns into `MemvidStore`, and keep the integration optional so
   `rig-memvid` can continue to depend only on `rig-core` for default builds.

## Maturity Bar

- The packer explains every omission and can be replayed from fixtures.
- Tests cover ranking, budget, reserve space, separators, relationship expansion, and stale/conflict behavior.
- Live smokes use the intended local model and complement deterministic tests.
- No public API requires callers to depend directly on `memvid-core` for common memory flows.

## Non-Goals

- Do not become a tool router or MCP bridge.
- Do not add a runtime dependency to library dependencies.
- Do not upstream `MemoryGraph`-style traits to Rig until a second backend validates the shape.
