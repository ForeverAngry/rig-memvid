# rig-memvid — Copilot Instructions

`rig-memvid` bridges the [Rig](https://crates.io/crates/rig-core) agent framework
with the [Memvid](https://crates.io/crates/memvid-core) `.mv2` AI-memory format.
It exposes two composable primitives that share an `Arc<Mutex<Memvid>>`:

- `MemvidStore` — implements Rig's `VectorStoreIndex` (RAG retrieval).
- `MemvidPersistHook` — implements Rig's `PromptHook` (writes turns to memory).

## Project conventions

- **Edition**: Rust 2024, MSRV `1.88`.
- **Errors**: define typed errors in [src/error.rs](src/error.rs) using `thiserror`.
  Return `Result<_, Error>`; do not introduce ad-hoc `Box<dyn Error>`.
- **Async**: public APIs are `async` and runtime-agnostic. Tests/examples use `tokio`.
- **Locking**: never `.await` while holding a `Mutex` guard
  (`clippy::await_holding_lock` is `deny`). Scope the guard, drop it, then await.
- **Logging**: use `tracing` (no `println!`/`dbg!` in library code; `dbg_macro` is forbidden).
- **No panics in library code**. The following clippy lints are `deny`/`forbid` and
  must stay clean:
  `unwrap_used`, `expect_used`, `panic`, `panic_in_result_fn`, `indexing_slicing`,
  `unreachable`, `todo`, `unimplemented`, `dbg_macro`.
  Prefer `?`, `ok_or(Error::...)`, `get(..)`, pattern matching.
  `unwrap`/`expect` are acceptable in `tests/`, `examples/`, and `#[cfg(test)]` blocks.
- **Public API**: keep `pub` surface minimal and documented with `///` rustdoc
  including a runnable (`no_run` is fine) example for new types/functions.

## Feature flags

Default = `lex`. Other flags forward to `memvid-core`:
`vec`, `api_embed`, `temporal`, `encryption`. When adding code that depends on
optional capabilities, gate it with `#[cfg(feature = "...")]` and ensure it
still compiles under the CI feature combos:

- `--all-features`
- `--no-default-features --features "lex,vec"`
- `--no-default-features --features "lex,api_embed"`

## Validation

Before considering a change done, run the same gates as CI (see [justfile](justfile)):

```sh
just check        # fmt --check + clippy (multiple feature sets) + tests
# or individually:
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --all-features
```

Integration tests live in [tests/](tests/); keep new behavioural tests there.
Examples in [examples/](examples/) must continue to build (`cargo build --examples`).

## When extending the crate

- New store-side functionality → extend [src/store.rs](src/store.rs); preserve the
  `VectorStoreIndex` contract and the shared `Arc<Mutex<Memvid>>`.
- New persistence behaviour → extend [src/hook.rs](src/hook.rs); honour
  `MemoryConfig` (`WritePolicy`, `commit_each_turn`, `default_tags`).
- Re-export new public items from [src/lib.rs](src/lib.rs).
- Update [README.md](README.md) and [CHANGELOG.md](CHANGELOG.md) for user-visible changes.

## Out of scope

- Do not vendor or fork `rig-core` / `memvid-core` APIs — depend on the
  published versions pinned in [Cargo.toml](Cargo.toml).
- Do not add a runtime dependency (e.g. `tokio`) to `[dependencies]`; keep
  the library runtime-agnostic.
