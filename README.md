# rig-memvid

[Memvid](https://crates.io/crates/memvid-core)-backed persistent memory and
vector store for [Rig](https://github.com/0xPlaygrounds/rig) agents.

`memvid-core` is a crash-safe, deterministic, single-file (`.mv2`) AI memory
format. This crate exposes that format to Rig in two ways:

1. `MemvidStore` — a `VectorStoreIndex` you can register with an agent for RAG.
2. `MemvidPersistHook` — a `PromptHook` that automatically appends every user
   prompt and assistant response into a memvid file as the agent runs.

The two compose: persist with the hook, recall with the store. They share an
`Arc<Mutex<Memvid>>` so writes are immediately searchable.

## Features

| Feature      | Default | Description                                                                                                            |
| ------------ | ------- | ---------------------------------------------------------------------------------------------------------------------- |
| `lex`        | yes     | BM25 / Tantivy lexical search.                                                                                         |
| `vec`        | no      | HNSW vector search via `memvid-core/vec` (ONNX Runtime + bundled BGE-small / BGE-base / Nomic / GTE-large).            |
| `api_embed`  | no      | Remote embedding providers (OpenAI etc.) via `memvid-core/api_embed`.                                                  |
| `temporal`   | no      | Temporal track / point-in-time queries.                                                                                |
| `encryption` | no      | At-rest encryption of the `.mv2` file.                                                                                 |

### Vector search (`vec`)

Enabling `vec` pulls in `ort = =2.0.0-rc.10`, which conflicts with
`rig-fastembed`'s `=2.0.0-rc.9`. Workspaces that depend on `rig-fastembed`
must stay on the default lex-only configuration until those pins converge
upstream. Workspaces that don't can opt in:

```toml
[dependencies]
rig-memvid = { version = "0.1", features = ["lex", "vec"] }
```

Attach a local embedder on the builder; reads and writes through
[`MemvidStore`] are then routed through memvid's HNSW index instead of
BM25:

```rust,ignore
let store = MemvidStore::builder()
    .path("./agent_memory.mv2")
    .with_default_embedder()? // BGE-small, 384-dim
    .open_or_create()?;
```

The ONNX model and tokenizer are loaded from
`$XDG_CACHE_HOME/memvid/text-models/` (or the platform equivalent). They
must already be on disk; memvid does not auto-download. Fetch them with:

```bash
mkdir -p ~/Library/Caches/memvid/text-models   # macOS
curl -L https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/onnx/model.onnx \
  -o ~/Library/Caches/memvid/text-models/bge-small-en-v1.5.onnx
curl -L https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/tokenizer.json \
  -o ~/Library/Caches/memvid/text-models/bge-small-en-v1.5_tokenizer.json
```

When an embedder is attached:

- `MemvidStore::put_text` embeds the payload and stores it via
  `put_with_embedding_and_options`.
- `MemvidStore::top_n` / `top_n_ids` embed the query and run
  `vec_search_with_embedding`.
- `MemvidFilter::scope` is honoured; `uri`, `as_of_frame`, and `as_of_ts`
  return `MemvidError::UnsupportedFilter` (memvid's vector path doesn't
  support those). Drop down to `MemvidStore::search(SearchRequest)` for
  full filter control.

## Compatibility

| `rig-memvid` | `rig-core` | `memvid-core` |
| ------------ | ---------- | ------------- |
| `0.1`        | `0.36`     | `2.0`         |

This crate is community-maintained and not affiliated with the `rig` project.

## WASM

`memvid-core` depends on `tantivy`, `mmap`, and (optionally) `onnxruntime`.
This crate is **not** WASM-compatible and will not build for `wasm32-*`
targets.

## Quickstart

```rust,no_run
use rig::providers::openai;
use rig_memvid::{MemvidStore, MemvidPersistHook, MemoryConfig, WritePolicy};

# async fn run() -> anyhow::Result<()> {
let store = MemvidStore::builder()
    .path("./agent_memory.mv2")
    .open_or_create()?;

let openai = openai::Client::from_env();

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

## Filter

`MemvidFilter` implements `SearchFilter` and accepts these keys via `eq(...)`:

- `uri` (`String`) — restrict to frames whose URI matches the given prefix
- `scope` (`String`) — restrict to a logical scope
- `as_of_frame` (`u64`) — point-in-time view by frame id
- `as_of_ts` (`i64`) — point-in-time view by unix-millis timestamp

`gt`/`lt`/`or` are not supported by memvid's query model and will produce a
`MemvidError::UnsupportedFilter` at query time.

## License

MIT
