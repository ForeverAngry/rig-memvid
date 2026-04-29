<div align="center">

# rig-memvid

**Persistent memory & vector store for [Rig](https://github.com/0xPlaygrounds/rig) agents,
backed by [Memvid](https://crates.io/crates/memvid-core).**

[![crates.io](https://img.shields.io/crates/v/rig-memvid.svg)](https://crates.io/crates/rig-memvid)
[![docs.rs](https://img.shields.io/docsrs/rig-memvid)](https://docs.rs/rig-memvid)
[![license](https://img.shields.io/crates/l/rig-memvid.svg)](./LICENSE)
[![rig-core](https://img.shields.io/badge/rig--core-0.36-blue)](https://crates.io/crates/rig-core)
[![memvid-core](https://img.shields.io/badge/memvid--core-2.0-blue)](https://crates.io/crates/memvid-core)

</div>

---

`memvid-core` is a crash-safe, deterministic, single-file (`.mv2`) AI memory format.
This crate exposes that format to Rig through two composable primitives:

- **`MemvidStore`** — a `VectorStoreIndex` you register with an agent for RAG.
- **`MemvidPersistHook`** — a `PromptHook` that appends every prompt and response
  to the memvid file as the agent runs.

Persist with the hook, recall with the store. They share an `Arc<Mutex<Memvid>>`,
so writes are searchable on the next turn.

## Contents

- [Features](#features)
- [Quickstart](#quickstart)
- [Vector search](#vector-search-vec)
- [Filters](#filters)
- [Compatibility](#compatibility)
- [Caveats](#caveats)
- [License](#license)

## Features

| Feature      | Default | Description                                                                  |
| ------------ | :-----: | ---------------------------------------------------------------------------- |
| `lex`        |   ✅    | BM25 / Tantivy lexical search.                                               |
| `vec`        |   —     | HNSW vector search via `memvid-core/vec` (ONNX Runtime + BGE / Nomic / GTE). |
| `api_embed`  |   —     | Remote embedding providers (OpenAI, etc.).                                   |
| `temporal`   |   —     | Temporal track / point-in-time queries.                                      |
| `encryption` |   —     | At-rest encryption of the `.mv2` file.                                       |

## Quickstart

Add the crate:

```toml
[dependencies]
rig-memvid = "0.1"
```

Wire a store and a persistence hook into your Rig agent:

```rust,no_run
use rig::providers::openai;
use rig_memvid::{MemvidStore, MemvidPersistHook, MemoryConfig, WritePolicy};

# async fn run() -> anyhow::Result<()> {
let store = MemvidStore::builder()
    .path("./agent_memory.mv2")
    .open_or_create()?;

let openai = openai::Client::from_env()?;

let hook = MemvidPersistHook::new(
    store.clone(),
    MemoryConfig {
        policy: WritePolicy::Raw,
        commit_each_turn: true,
        default_tags: vec!["chat".into()],
    },
);

let agent = openai
    .agent(openai::GPT_4O)
    .preamble("You are a helpful assistant with persistent memory.")
    .dynamic_context(4, store)
    .build();

let response = agent
    .prompt("What did we discuss yesterday?")
    .with_hook(hook)
    .await?;

println!("{response}");
# Ok(()) }
```

> **See also:** [`examples/chatbot_with_memory.rs`](examples/chatbot_with_memory.rs)
> (OpenAI) and [`examples/chatbot_with_memory_ollama.rs`](examples/chatbot_with_memory_ollama.rs)
> (local Ollama).

## Vector search (`vec`)

By default, `MemvidStore` uses BM25 lexical search. Enable the `vec` feature
to switch reads and writes through memvid's HNSW vector index:

```toml
[dependencies]
rig-memvid = { version = "0.1", features = ["lex", "vec"] }
```

```rust,ignore
let store = MemvidStore::builder()
    .path("./agent_memory.mv2")
    .with_default_embedder()? // BGE-small, 384-dim
    .open_or_create()?;
```

When an embedder is attached:

- `put_text` embeds the payload and stores it via `put_with_embedding_and_options`.
- `top_n` / `top_n_ids` embed the query and run `vec_search_with_embedding`.
- Only `MemvidFilter::scope` is honoured on the vec path. `uri`, `as_of_frame`,
  and `as_of_ts` return `MemvidError::UnsupportedFilter` — drop down to
  `MemvidStore::search(SearchRequest)` for full filter control.

> ⚠️ **Dependency conflict:** `vec` pins `ort = =2.0.0-rc.10`, which collides
> with `rig-fastembed`'s `=2.0.0-rc.9`. If you depend on `rig-fastembed`,
> stay lex-only until the pins converge upstream.

<details>
<summary><strong>Bootstrapping the local embedder model</strong></summary>

ONNX models and tokenizers load from `$XDG_CACHE_HOME/memvid/text-models/`
(or the platform equivalent). Memvid does not auto-download — fetch them
manually:

```bash
mkdir -p ~/Library/Caches/memvid/text-models   # macOS
curl -L https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/onnx/model.onnx \
  -o ~/Library/Caches/memvid/text-models/bge-small-en-v1.5.onnx
curl -L https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/tokenizer.json \
  -o ~/Library/Caches/memvid/text-models/bge-small-en-v1.5_tokenizer.json
```

> Note the underscore in `bge-small-en-v1.5_tokenizer.json` — memvid expects
> that exact filename.

</details>

## Filters

`MemvidFilter` implements `SearchFilter` and accepts the following keys via
`eq(...)`:

| Key           | Type     | Meaning                                       |
| ------------- | -------- | --------------------------------------------- |
| `uri`         | `String` | Restrict to frames whose URI matches a prefix |
| `scope`       | `String` | Restrict to a logical scope                   |
| `as_of_frame` | `u64`    | Point-in-time view by frame id                |
| `as_of_ts`    | `i64`    | Point-in-time view by unix-millis timestamp   |

`gt` / `lt` / `or` are not supported by memvid's query model and produce
`MemvidError::UnsupportedFilter` at query time.

## Compatibility

| `rig-memvid` | `rig-core` | `memvid-core` |
| ------------ | ---------- | ------------- |
| `0.1`        | `0.36`     | `2.0`         |

This crate is community-maintained and not affiliated with the `rig` project.

## Caveats

- **No WASM.** `memvid-core` depends on `tantivy`, `mmap`, and optionally
  `onnxruntime`; this crate will not build for `wasm32-*` targets.
- **BM25 tokenization is strict.** Lexical queries with stopwords or
  full-sentence phrasing may return zero hits — prefer keyword-style
  queries on the lex path, or enable `vec` for semantic retrieval.

## License

[MIT](./LICENSE)
