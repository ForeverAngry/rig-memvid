# rig-memvid task runner.
#
# Install just: https://github.com/casey/just
#   brew install just
#
# Run `just` with no args to see the recipe list.

# Path to the memvid archive used by the Ollama chatbot example.
MEMORY := "chatbot_memory_ollama.mv2"

# Default Ollama model + endpoint (override on the command line, e.g.
#   `just agent OLLAMA_MODEL=llama3.2`).
OLLAMA_MODEL    := env_var_or_default("OLLAMA_MODEL", "qwen2.5-coder:3b-instruct")
OLLAMA_API_BASE := env_var_or_default("OLLAMA_API_BASE_URL", "http://localhost:11434")

# Show all recipes.
default:
    @just --list

# Run the local-Ollama chatbot end-to-end (writes into {{MEMORY}}).
agent:
    OLLAMA_MODEL="{{OLLAMA_MODEL}}" \
    OLLAMA_API_BASE_URL="{{OLLAMA_API_BASE}}" \
    cargo run --example chatbot_with_memory_ollama

# Pull the configured Ollama model so `just agent` can use it.
pull:
    ollama pull "{{OLLAMA_MODEL}}"

# List models available on the local Ollama daemon.
models:
    @ollama list

# Pretty-print every frame stored in a memvid archive (default {{MEMORY}}).
inspect FILE=MEMORY:
    cargo run --quiet --example inspect_memory -- "{{FILE}}"

# End-to-end demo: run the agent then visualise what landed in memory.
demo: agent inspect

# Wipe the local memory archive so the next `just agent` run starts fresh.
reset:
    rm -f "{{MEMORY}}"
    @echo "removed {{MEMORY}}"

# Build all targets with default features.
build:
    cargo build --all-targets

# Run formatter check + clippy + tests across the same feature combos as CI.
check: fmt clippy test

fmt:
    cargo fmt --all -- --check

clippy:
    cargo clippy --all-targets -- -D warnings
    cargo clippy --no-default-features --features "lex,vec" --all-targets -- -D warnings
    cargo clippy --no-default-features --features "lex,api_embed" --all-targets -- -D warnings

test:
    cargo test --all-targets
    cargo test --no-default-features --all-targets
    cargo test --no-default-features --features "lex,vec" --all-targets

# Run the (ignored) semantic search test. Downloads BGE-small on first run.
test-semantic:
    cargo test --no-default-features --features "lex,vec" -- --ignored vec_semantic_search

# Validate the package as it would be uploaded to crates.io.
publish-dry-run:
    cargo publish --dry-run

# Publish to crates.io. Requires `cargo login` or CARGO_REGISTRY_TOKEN.
publish:
    cargo publish

# Preview what release-plz would bump/changelog without changing anything.
# Install: `cargo install release-plz` (or `brew install release-plz`).
release-preview:
    release-plz update --dry-run

# Open a release PR locally (writes to a branch). Same thing CI does on push.
release-pr:
    release-plz release-pr

# Inspect the next semver bump release-plz would compute from current commits.
next-version:
    @release-plz update --dry-run 2>&1 | grep -E "(bumping|no changes)" || true
