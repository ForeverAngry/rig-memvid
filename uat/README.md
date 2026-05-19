# rig-memvid UAT

User-acceptance tests for the **compaction integration** surface
(`MemvidDemotionHook`, `MemvidStoringCompactor`,
`rig_memory::CompactingMemory`). These are **black-box behavioural
specs** to be exercised against a chat agent backed by `rig-memvid` —
not Rust unit tests. They are designed to be runnable by an unbiased
operator (human or LLM in a chat surface) who has no access to the
crate's internals.

## Layout

```text
uat/
├── README.md                                    # this file
├── expectations/
│   └── <test_name>/
│       └── v<n>.md                              # spec, never overwritten
└── results/
    └── <test_name>/
        └── v<n>/
            └── <iso8601-timestamp>__<runner>.md # captured transcript
```

- `expectations/<test_name>/v<n>.md` — pinned specification. Once a
  version is published, never edit it; bump to `v(n+1)` instead.
- `results/<test_name>/v<n>/<timestamp>__<runner>.md` — one file per
  run, capturing the chat transcript and pass/fail verdict against the
  spec's acceptance criteria.

## How to run a spec

1. Pick a spec under `expectations/<test_name>/v<n>.md`.
2. Set up an agent that:
   - Uses `rig-memvid` 0.1.4+ with the `compaction` feature enabled.
   - Wires `MemvidDemotionHook` and/or `MemvidStoringCompactor` into a
     `rig_memory::CompactingMemory` with a small token budget (so
     demotion fires within the test budget — typically 4-8k tokens).
   - Persists `.mv2` to a path the operator can clear between runs.
3. Follow the spec **verbatim** — same prompts, same order, same
   restart points. Do not paraphrase user messages.
4. Capture the transcript (every user prompt + every assistant reply +
   any out-of-band action like "kill the process and restart") to a
   new file under `results/<test_name>/v<n>/`.
5. At the bottom of the result file, record:
   - **Verdict**: `PASS`, `FAIL`, or `INCONCLUSIVE`.
   - **Evidence**: short bullet list mapping each acceptance criterion
     to the transcript line(s) that satisfy or violate it.
   - **Environment**: model name, OS, `cargo --version`, archive path,
     and any non-default `MemoryConfig` knobs.

## Result file naming

`<iso8601-utc>__<runner-id>.md`

Examples:

- `2025-01-15T18-42-03Z__bc.md`
- `2025-01-15T19-08-11Z__claude-opus-4.7.md`

Use `:` → `-` substitution in the timestamp so the path stays portable.

## Independence rule

Spec authors **must not** reference internal types, feature flags,
function names, or implementation details. The acceptance criteria are
observable from the chat surface alone: did the assistant recall the
fact? Did duplicates appear in subsequent retrieval? Did the right
scope answer? Anything else is out of scope for UAT.

## Spec catalogue (v1)

| Test name                         | What it pins                       |
| --------------------------------- | ---------------------------------- |
| `demotion_persistence_basic`      | Eviction-aware long-horizon recall |
| `demotion_idempotency_restart`    | No duplicate cards after restart   |
| `compaction_summary_searchable`   | Compaction summaries are usable    |
| `scope_isolation`                 | Cross-scope leakage is impossible  |
| `principal_binding_consistency`   | Preferences stick to the speaker   |
