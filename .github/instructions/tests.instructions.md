---
applyTo: "tests/**/*.rs"
description: "Conventions for integration tests in rig-memvid (tokio, tempfile, allowed unwraps)."
---

# Test conventions

- Use `#[tokio::test(flavor = "multi_thread")]` for async tests.
- Create `.mv2` archives in a `tempfile::TempDir` so tests are hermetic.
- `unwrap` / `expect` are allowed here (clippy lints are relaxed in tests).
- Prefer `anyhow::Result<()>` as the test return type for `?` ergonomics.
- Cover both success paths and `Error` variants from
  [src/error.rs](../../src/error.rs).
- Gate feature-specific tests with `#[cfg(feature = "...")]` so they are
  skipped under the default `lex`-only build.
- Do not depend on the network. The `api_embed` paths must be stubbed or
  skipped unless an env var (e.g. `OPENAI_API_KEY`) is set — in which case
  guard the test with `if std::env::var("...").is_err() { return Ok(()); }`.
