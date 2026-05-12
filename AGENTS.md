# AGENTS.md

Guidance for AI coding agents working in `rig-memvid`. Mirrors
[.github/copilot-instructions.md](.github/copilot-instructions.md) for tools
that read `AGENTS.md` (Codex, Cursor, etc.).

## Project

`rig-memvid` exposes the Memvid `.mv2` AI-memory format to Rig agents via two
primitives sharing an `Arc<Mutex<Memvid>>`:

- `MemvidStore` — Rig `VectorStoreIndex` for retrieval ([src/store.rs](src/store.rs)).
- `MemvidPersistHook` — Rig `PromptHook` for writes ([src/hook.rs](src/hook.rs)).

## Rules

- Rust 2024, MSRV 1.89. Library is runtime-agnostic; do not add `tokio` to
  `[dependencies]`.
- Errors: typed `thiserror` enum in [src/error.rs](src/error.rs); return
  `Result<_, Error>`.
- Never `.await` while holding a `Mutex` guard. Scope-drop the guard first.
- No `unwrap`, `expect`, `panic!`, `todo!`, `unimplemented!`, `dbg!`,
  indexing/slicing, or `unreachable!` in library code — these are clippy
  `deny`/`forbid`. Use `?`, `ok_or(Error::...)`, `get(..)`, match.
  Allowed in `tests/`, `examples/`, `#[cfg(test)]`.
- Use `tracing` for logs; no `println!` in library code.
- Document new `pub` items with `///` rustdoc and a `no_run` example.
- Re-export new public items from [src/lib.rs](src/lib.rs).

## Features

Default `lex`. Optional: `vec`, `api_embed`, `temporal`, `encryption`. Gate
optional code with `#[cfg(feature = "...")]`. Code must build under all CI
combos:

- `--all-features`
- `--no-default-features --features "lex,vec"`
- `--no-default-features --features "lex,api_embed"`

## Validation

```sh
just check
# = cargo fmt --all -- --check
#   cargo clippy --all-targets -- -D warnings  (× feature combos)
#   cargo test --all-features
```

Integration tests: [tests/](tests/). Examples must keep building:
`cargo build --examples`.

## Scope

Do not fork or vendor `rig-core` / `memvid-core` — depend on the versions in
[Cargo.toml](Cargo.toml). Update [README.md](README.md) and
[CHANGELOG.md](CHANGELOG.md) for user-visible changes.
