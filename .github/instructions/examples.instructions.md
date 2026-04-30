---
applyTo: "examples/**/*.rs"
description: "Conventions for runnable examples in rig-memvid (tokio main, env-driven config, no secrets)."
---

# Example conventions

- Each example must compile under `cargo build --examples` with default
  features. Feature-specific examples must be gated with
  `#[cfg(feature = "...")]` plus a `compile_error!`-free fallback `main`.
- Use `#[tokio::main]` and return `anyhow::Result<()>`.
- Read configuration from environment variables with sensible defaults
  (see [examples/chatbot_with_memory_ollama.rs](../../examples/chatbot_with_memory_ollama.rs)
  for `OLLAMA_MODEL` / `OLLAMA_API_BASE_URL`). Never hard-code secrets.
- Persist memory to a path configurable via env or CLI arg; default to a
  file in the workspace root (e.g. `chatbot_memory.mv2`).
- Use `tracing_subscriber::fmt().with_env_filter(...)` for logging output
  rather than `println!` for diagnostics; `println!` is fine for the
  user-facing chat transcript.
- Keep examples small and self-contained; shared helpers belong in `src/`,
  not duplicated across examples.
