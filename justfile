# rig-memvid task runner.
#
# Install just: https://github.com/casey/just
#   brew install just
#
# Run `just` with no args to see the recipe list.

# Path to the memvid archive used by the Ollama chatbot example.
MEMORY := "chatbot_memory_ollama.mv2"

# Default Ollama model + endpoint (override on the command line, e.g.
#   `just agent OLLAMA_MODEL=qwen3.5:9b`).
OLLAMA_MODEL    := env_var_or_default("OLLAMA_MODEL", "qwen3.5:9b")
OLLAMA_API_BASE := env_var_or_default("OLLAMA_API_BASE_URL", "http://localhost:11434")

# Show all recipes.
default:
    @just --list

# --- Ollama chatbot helpers ---------------------------------------------------

# Run the local-Ollama chatbot end-to-end (writes into {{MEMORY}}).
agent:
    OLLAMA_MODEL="{{OLLAMA_MODEL}}" \
    OLLAMA_API_BASE_URL="{{OLLAMA_API_BASE}}" \
    cargo run --example chatbot_with_memory_ollama

# Run the local-Ollama chatbot with an explicit model name.
agent-with-model MODEL:
    OLLAMA_MODEL="{{MODEL}}" \
    OLLAMA_API_BASE_URL="{{OLLAMA_API_BASE}}" \
    cargo run --example chatbot_with_memory_ollama

# Pick a model from `ollama list`, then run the local-Ollama chatbot.
agent-select:
    #!/usr/bin/env bash
    set -euo pipefail

    models=($(ollama list | awk 'NR > 1 && NF { print $1 }'))
    if [ "${#models[@]}" -eq 0 ]; then
        echo "No Ollama models found. Run 'just pull' or 'ollama pull <model>' first." >&2
        exit 1
    fi

    echo "Select an Ollama model:"
    select model in "${models[@]}"; do
        if [ -n "${model:-}" ]; then
            OLLAMA_MODEL="${model}" \
            OLLAMA_API_BASE_URL="{{OLLAMA_API_BASE}}" \
            cargo run --example chatbot_with_memory_ollama
            break
        fi
        echo "Invalid selection." >&2
    done

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

# --- Standard recipes ---------------------------------------------------------

# Build all targets with default features.
build:
    cargo build --all-targets

# Format check + clippy + tests + msrv + doc + examples.
check: fmt clippy test msrv doc examples

# Verify code is formatted (does not mutate).
fmt:
    cargo fmt --all -- --check

# Clippy across CI feature combos.
clippy:
    cargo clippy --all-targets -- -D warnings
    cargo clippy --no-default-features --features "lex,vec" --all-targets -- -D warnings
    cargo clippy --no-default-features --features "lex,api_embed" --all-targets -- -D warnings
    cargo clippy --no-default-features --features "lex,vec,api_embed" --all-targets -- -D warnings
    cargo clippy --no-default-features --features "lex,temporal" --all-targets -- -D warnings
    cargo clippy --no-default-features --features "lex,encryption" --all-targets -- -D warnings
    cargo clippy --no-default-features --features "lex,compaction" --all-targets -- -D warnings
    cargo clippy --no-default-features --features "lex,context-projection" --all-targets -- -D warnings
    cargo clippy --no-default-features --features "lex,observe" --all-targets -- -D warnings
    cargo clippy --all-features --all-targets -- -D warnings

# Tests across CI feature combos.
test:
    cargo test --all-targets
    cargo test --no-default-features --all-targets
    cargo test --no-default-features --features "lex,vec" --all-targets
    cargo test --no-default-features --features "lex,api_embed" --all-targets
    cargo test --no-default-features --features "lex,compaction" --all-targets
    cargo test --no-default-features --features "lex,context-projection" --all-targets
    cargo test --no-default-features --features "lex,observe" --all-targets
    cargo test --all-features --all-targets

# MSRV gate (Rust 1.89).
msrv:
    cargo +1.89 build --all-targets

# Rustdoc with strict warnings.
doc:
    RUSTDOCFLAGS="-D warnings -D rustdoc::broken_intra_doc_links" cargo doc --all-features --no-deps

# Build every example with all features.
examples:
    cargo build --examples --all-features

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
release-preview:
    release-plz update --dry-run

# Open a release PR locally (writes to a branch). Same thing CI does on push.
release-pr:
    release-plz release-pr

# Inspect the next semver bump release-plz would compute from current commits.
next-version:
    @release-plz update --dry-run 2>&1 | grep -E "(bumping|no changes|next version)" || true

# Run all checks needed for a PR / commit to main locally.
pr-ready: check test-semantic publish-dry-run

# Install a git pre-push hook that runs `just pr-ready`.
install-hooks:
    #!/usr/bin/env bash
    echo '#!/usr/bin/env bash' > .git/hooks/pre-push
    echo 'set -e' >> .git/hooks/pre-push
    echo 'echo "Running just pr-ready..."' >> .git/hooks/pre-push
    echo 'just pr-ready' >> .git/hooks/pre-push
    chmod +x .git/hooks/pre-push
    echo "pre-push hook installed."
