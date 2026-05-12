//! Micro-bench for SIMD vs scalar vector retrieval in `MemvidStore`.
//!
//! Builds (or reuses) a dedicated `bench_vec_search.mv2` fixture
//! populated with `BENCH_CORPUS_SIZE` synthetic chunks so the
//! similarity-scoring loop in `memvid-core` dominates the per-query
//! cost, then runs timed `top_n` calls against it.
//!
//! Run with SIMD on (default):
//!   cargo run --release --features vec --example bench_vec_search
//!
//! Run with SIMD off (requires un-chaining `simd` from `vec` in
//! Cargo.toml first):
//!   cargo run --release --no-default-features --features vec \
//!       --example bench_vec_search
//!
//! Environment overrides:
//!   BENCH_CORPUS_SIZE  number of chunks to seed (default 5000)
//!   BENCH_ITERATIONS   number of timed queries (default 500)
//!   BENCH_SAMPLES      top-k per query (default 16)
//!   BENCH_REGEN=1      force regeneration of the fixture

#[cfg(not(feature = "vec"))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("bench_vec_search requires --features vec")
}

#[cfg(feature = "vec")]
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    use std::path::PathBuf;
    use std::time::Instant;

    use memvid_core::{PutOptions, SearchHit};
    use rig::vector_store::VectorStoreIndex;
    use rig::vector_store::request::VectorSearchRequestBuilder;
    use rig_memvid::{MemvidFilter, MemvidStore};

    fn env_usize(key: &str, default: usize) -> usize {
        std::env::var(key)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(default)
    }

    let simd_status = if cfg!(feature = "simd") {
        "ON  (rig-memvid `simd` feature enabled)"
    } else {
        "OFF (rig-memvid `simd` feature disabled)"
    };
    println!("SIMD feature: {simd_status}");

    let corpus_size = env_usize("BENCH_CORPUS_SIZE", 5000);
    let iterations = env_usize("BENCH_ITERATIONS", 500);
    let samples = env_usize("BENCH_SAMPLES", 16) as u64;
    let regen = std::env::var("BENCH_REGEN").is_ok_and(|v| v != "0" && !v.is_empty());

    let path = PathBuf::from(
        std::env::var("MEMVID_PATH").unwrap_or_else(|_| "bench_vec_search.mv2".to_string()),
    );

    if regen && path.exists() {
        std::fs::remove_file(&path)?;
    }
    let need_seed = !path.exists();

    let store = MemvidStore::builder()
        .path(&path)
        .enable_lex()
        .with_default_embedder()?
        .open_or_create()?;

    if need_seed {
        println!("Seeding {corpus_size} chunks into {} ...", path.display());
        let topics = [
            "coffee",
            "espresso",
            "tea",
            "berlin",
            "lisbon",
            "rust",
            "python",
            "neural networks",
            "vector search",
            "tokio runtime",
            "memvid",
            "agents",
            "embeddings",
            "compaction",
            "tantivy",
            "BM25",
            "transformers",
            "PostgreSQL",
            "indexing",
            "kubernetes",
            "docker",
            "linux kernel",
            "macOS",
            "WebAssembly",
            "compilers",
            "borrow checker",
            "async/await",
            "garbage collection",
            "SIMD",
            "cache locality",
        ];
        let templates = [
            "The user prefers {} over the alternative when working on long projects.",
            "Today we discussed {} and how it interacts with downstream systems.",
            "A common gotcha with {} is the implicit cost of repeated allocation.",
            "When evaluating {}, benchmark first under realistic load.",
            "Notes on {}: keep the hot path tight and avoid unnecessary cloning.",
            "Reminder: {} should be measured, not assumed.",
            "Compared two implementations of {} and the second won by 15%.",
            "On {}, the conventional wisdom is wrong about half the time.",
        ];
        let start = Instant::now();
        for i in 0..corpus_size {
            let topic = topics.get(i % topics.len()).copied().unwrap_or("topic");
            let template = templates
                .get((i / topics.len()) % templates.len())
                .copied()
                .unwrap_or("Note about {}.");
            let text = format!("[chunk {i}] {}", template.replace("{}", topic));
            store.put_text_uncommitted(&text, PutOptions::default())?;
        }
        store.commit()?;
        println!("Seed complete in {:.2?}", start.elapsed());
    } else {
        store.commit()?;
        println!("Reusing existing fixture at {}", path.display());
    }

    const QUERIES: &[&str] = &[
        "what does the user prefer for coffee",
        "tell me about vector search performance",
        "discussion of async runtimes",
        "notes on compilers and the borrow checker",
        "wisdom about benchmarking",
        "how do we keep the hot path tight",
        "stories about kubernetes deployments",
        "anything about WebAssembly",
    ];

    // Warmup.
    for q in QUERIES.iter().take(2) {
        let req = VectorSearchRequestBuilder::<MemvidFilter>::default()
            .query(*q)
            .samples(samples)
            .build();
        let _: Vec<(f64, String, SearchHit)> = store.top_n(req).await?;
    }

    let mut total_hits = 0usize;
    let start = Instant::now();
    for i in 0..iterations {
        let q = QUERIES.get(i % QUERIES.len()).copied().unwrap_or("");
        let req = VectorSearchRequestBuilder::<MemvidFilter>::default()
            .query(q)
            .samples(samples)
            .build();
        let hits: Vec<(f64, String, SearchHit)> = store.top_n(req).await?;
        total_hits += hits.len();
    }
    let elapsed = start.elapsed();

    let per_query_us = elapsed.as_micros() as f64 / iterations as f64;
    println!(
        "corpus={corpus_size}, iterations={iterations}, samples={samples} \
         => total {:.3?}, {per_query_us:.1} µs/query, {total_hits} total hits",
        elapsed,
    );
    Ok(())
}
