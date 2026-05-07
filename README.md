# rig-memvid

Memvid-backed persistent memory and lexical store for Rig agents.

[![crates.io](https://img.shields.io/crates/v/rig-memvid.svg)](https://crates.io/crates/rig-memvid)
[![docs.rs](https://img.shields.io/docsrs/rig-memvid)](https://docs.rs/rig-memvid)
[![license](https://img.shields.io/crates/l/rig-memvid.svg)](LICENSE)
[![rig-core](https://img.shields.io/badge/rig--core-0.36.0-blue)](https://crates.io/crates/rig-core)
[![memvid-core](https://img.shields.io/badge/memvid--core-2.0.139-blue)](https://crates.io/crates/memvid-core)

## Overview

`rig-memvid` exposes Memvid's single-file `.mv2` memory format to Rig agents. It provides a persistent `MemvidStore` that implements Rig vector-store traits, a `MemvidPersistHook` that writes prompt turns into the same archive, and an `InMemoryStore<E>` fallback for deterministic no-disk lexical retrieval in tests and offline modes.

The intended production pattern is: write user and assistant turns through `MemvidPersistHook`, then recall from the same `MemvidStore` through Rig dynamic context or direct vector-store queries.

## Why It Exists

Rig already defines provider-agnostic retrieval and prompt-hook traits. Memvid provides a crash-safe `.mv2` archive with lexical, vector, ACL, temporal, and encryption capabilities. `rig-memvid` fills the adapter gap by implementing Rig's `VectorStoreIndex`, `InsertDocuments`, and `PromptHook` flows over Memvid without making callers depend directly on `memvid-core` APIs for common use.

## Status

- Crate version: `0.1.2`.
- Rust edition: 2024.
- MSRV: 1.88.
- `rig-core` dependency: `0.36.0` with default features disabled.
- `memvid-core` dependency: `2.0.139` with default features disabled.
- Runtime stance: runtime-agnostic library; `tokio` is only a dev-dependency for tests and examples.
- Platform stance: not supported on `wasm` targets because `memvid-core` requires synchronous file I/O and OS-level file locking.
- Current Unreleased work adds `InMemoryStore<E>` and Unicode-aware lexical normalization for that offline store.

## Feature Flags

| Feature | Default | Enables | Checked by `just check` |
| --- | --- | --- | --- |
| `lex` | yes | Memvid lexical search via `memvid-core/lex`. | default clippy and tests; also in `lex,vec` and `lex,api_embed` clippy combos |
| `vec` | no | Memvid local vector search via `memvid-core/vec`. | clippy with `--no-default-features --features "lex,vec"`; tests with the same combo |
| `api_embed` | no | Remote embedding provider support via `memvid-core/api_embed`. | clippy with `--no-default-features --features "lex,api_embed"` |
| `temporal` | no | Temporal track support via `memvid-core/temporal_track`. | not exercised by `just check` |
| `encryption` | no | At-rest encryption via `memvid-core/encryption`. | not exercised by `just check` |

## Key Types

- [src/store.rs](src/store.rs): `MemvidStore`, the cloneable `Arc<Mutex<Memvid>>` wrapper implementing Rig retrieval and insertion traits.
- [src/store.rs](src/store.rs): `MemvidStoreBuilder`, with file lifecycle methods, lexical enablement, snippet sizing, ACL context, read-only open, and vector embedder configuration when `vec` is enabled.
- [src/store.rs](src/store.rs): `MemvidFilter`, a Rig `SearchFilter` adapter for `uri`, `scope`, `as_of_frame`, and `as_of_ts` predicates.
- [src/hook.rs](src/hook.rs): `MemvidPersistHook<M>`, a Rig `PromptHook` implementation that writes user prompts and assistant responses into `MemvidStore`.
- [src/hook.rs](src/hook.rs): `MemoryConfig`, `WritePolicy`, and `WriteTransform`, which control what gets persisted, commit cadence, default tags, and scope URI.
- [src/inmem.rs](src/inmem.rs): `Episode`, `InMemoryStore<E>`, `InMemoryHit<E>`, and `InMemoryError`, the no-disk deterministic lexical retrieval surface.
- [src/error.rs](src/error.rs): `MemvidError`, the typed error surface for store, filter, lifecycle, and memvid failures.

The crate re-exports `memvid_core` so callers can construct `PutOptions`, `AclContext`, and `SearchRequest` without adding a direct dependency.

## Integration With Rig

`rig-memvid` pins `rig-core = 0.36.0` in [Cargo.toml](Cargo.toml). `MemvidStore` plugs into Rig's vector-store flow, including `VectorStoreIndex` and `InsertDocuments`. `MemvidPersistHook<M>` plugs into Rig's prompt lifecycle via `PromptHook<M>` for any `CompletionModel`.

It is community-maintained and not part of the upstream `rig` repository.

## Usage

Persistent store behavior is covered by [tests/smoke.rs](tests/smoke.rs) and [tests/integration.rs](tests/integration.rs). The examples [examples/chatbot_with_memory.rs](examples/chatbot_with_memory.rs), [examples/chatbot_with_memory_ollama.rs](examples/chatbot_with_memory_ollama.rs), and [examples/inspect_memory.rs](examples/inspect_memory.rs) show end-to-end archive usage.

```rust,no_run
use memvid_core::PutOptions;
use rig::vector_store::{
    request::VectorSearchRequestBuilder, VectorSearchRequest, VectorStoreIndex,
};
use rig_memvid::{MemvidFilter, MemvidStore};

# async fn run() -> Result<(), Box<dyn std::error::Error>> {
let store = MemvidStore::builder()
    .path("./agent_memory.mv2")
    .enable_lex()
    .open_or_create()?;

store.put_text(
    "The Tower of London was founded by William the Conqueror in 1066.",
    PutOptions::default(),
)?;

let request: VectorSearchRequest<MemvidFilter> =
    VectorSearchRequestBuilder::<MemvidFilter>::default()
        .query("Tower of London")
        .samples(5)
        .build();

let hits: Vec<(f64, String, serde_json::Value)> = store.top_n(request).await?;
assert!(!hits.is_empty());
# Ok(()) }
```

For no-disk tests or offline modes, [src/inmem.rs](src/inmem.rs) includes unit tests for append, lookup, deterministic ranking, zero-score filtering, and Unicode normalization.

```rust,no_run
use rig_memvid::{Episode, InMemoryStore};

#[derive(Clone)]
struct Finding {
    summary: String,
}

impl Episode for Finding {
    fn summary(&self) -> &str {
        &self.summary
    }
}

# async fn run() -> Result<(), Box<dyn std::error::Error>> {
let store = InMemoryStore::<Finding>::new();
store
    .append(Finding {
        summary: "ПОЛЬЗОВАТЕЛЬ logged in".into(),
    })
    .await?;

let hits = store.retrieve_similar("пользователь", 5).await?;
assert_eq!(hits.len(), 1);
# Ok(()) }
```

## Validation

Canonical validation is `just check`.

That recipe runs formatter checks, clippy for default features plus `--no-default-features --features "lex,vec"` and `--no-default-features --features "lex,api_embed"`, then tests for default features, no default features, and `--no-default-features --features "lex,vec"`.

Examples must also continue to build with `cargo build --examples`.

## Gotchas

- `MemvidStore` uses `std::sync::Mutex`, not `tokio::sync::Mutex`, to remain runtime-agnostic. Guards are always dropped before `.await` points.
- Reads cannot run in parallel through one `MemvidStore` handle because the underlying `Memvid` API takes `&mut self`. Open separate read-only handles for high-concurrency read workloads.
- `MemvidStore::search` is the raw memvid path. Do not call it from inside a `WriteTransform`; hook writes already go through the same store and a re-entrant call can deadlock.
- `MemvidFilter::gt`, `lt`, and `or` are rejected because they do not map onto memvid's query model.
- The `vec` path only honors `MemvidFilter::scope`; `uri`, `as_of_frame`, and `as_of_ts` are unsupported on vector search.
- `InMemoryStore` is deterministic and dependency-free, but it is lexical token overlap only, not semantic vector retrieval.
- `rig-memvid` intentionally fails to compile on `wasm` targets with a clear message.

## Ecosystem

These companion crates are maintained as separate repositories. Together they form a small stack around the upstream Rig project: `rig-compose` provides the kernel surface, `rig-resources` contributes reusable skills and tools, `rig-mcp` moves tools across MCP, and `rig-memvid` connects Rig agents to persistent `.mv2` memory.

```mermaid
flowchart TD
    rig["rig / rig-core"]
    compose["rig-compose 0.1.x"]
    resources["rig-resources 0.1.x"]
    mcp["rig-mcp 0.1.x"]
    memvid["rig-memvid 0.1.x"]

    compose -. "Rig-shaped kernel; no direct rig-core dep" .-> rig
    resources -- "rig-compose = 0.2; features: security, graph, full" --> compose
    mcp -- "rig-compose = 0.2; rmcp stdio bridge" --> compose
    memvid -- "rig-core = 0.36.0; features: lex, vec, api_embed, temporal, encryption" --> rig
```

Pinned Rig-facing dependencies from the current manifests:

| Crate | Direct Rig-facing dependency | Notes |
| --- | --- | --- |
| `rig-compose` | none | Defines a Rig-shaped kernel surface without depending on `rig-core`. |
| `rig-resources` | `rig-compose = 0.2` | Provides reusable skills, resource tools, and security helpers. |
| `rig-mcp` | `rig-compose = 0.2` | Bridges `rig-compose` tools over MCP stdio and loopback transports. |
| `rig-memvid` | `rig-core = 0.36.0` | Implements Rig vector-store and prompt-hook flows over Memvid. |

The concrete multi-crate workflow tested today is the MCP loopback path: a `rig_compose::ToolRegistry` is exposed through `rig_mcp::LoopbackTransport`, remote schemas are wrapped as `rig_mcp::McpTool`, and the wrapped tools are registered back into another `ToolRegistry`. That proves a local `rig-compose` tool and an MCP-adapted tool are indistinguishable to callers. The backing test is `mcp_tool_indistinguishable_from_local` in [rig-mcp/src/transport.rs](https://github.com/ForeverAngry/rig-mcp/blob/main/src/transport.rs).

## License

[MIT](LICENSE)
