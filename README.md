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

| Feature      | Default | Description                                              |
| ------------ | ------- | -------------------------------------------------------- |
| `lex`        | yes     | BM25 / Tantivy lexical search.                           |
| `temporal`   | no      | Temporal track / point-in-time queries.                  |
| `encryption` | no      | At-rest encryption of the `.mv2` file.                   |

v0.1 is lexical-only. Vector backends (`memvid-core/vec`, `memvid-core/api-embed`)
pin `ort = =2.0.0-rc.10`, which conflicts with `rig-fastembed`'s `=2.0.0-rc.9`
pin and breaks dependency resolution for any workspace that uses both. Vector
support will be re-enabled once those pins converge.

## Compatibility

| `rig-memvid` | `rig-core` | `memvid-core` |
| ------------ | ---------- | ------------- |
| `0.1`        | `0.35`     | `2.0`         |

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
