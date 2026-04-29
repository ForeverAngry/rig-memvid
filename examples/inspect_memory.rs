//! Pretty-print the contents of a `.mv2` memvid archive.
//!
//! Usage:
//!
//! ```bash
//! cargo run --example inspect_memory -- path/to/memory.mv2
//! ```
//!
//! Defaults to `chatbot_memory_ollama.mv2` (the file produced by the
//! `chatbot_with_memory_ollama` example) when no argument is given.

use std::path::PathBuf;

use anyhow::{Context, Result};
use memvid_core::Memvid;

fn main() -> Result<()> {
    let path: PathBuf = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("chatbot_memory_ollama.mv2"));

    if !path.exists() {
        anyhow::bail!(
            "memvid file not found: {}\n\
             Tip: run `cargo run --example chatbot_with_memory_ollama` first.",
            path.display()
        );
    }

    let mut mv = Memvid::open_read_only(&path)
        .with_context(|| format!("failed to open {}", path.display()))?;
    let count = mv.frame_count();

    let bar = "─".repeat(72);
    println!("{bar}");
    println!(" memvid archive : {}", path.display());
    println!(" frames         : {count}");
    println!("{bar}");

    if count == 0 {
        println!("(empty)");
        return Ok(());
    }

    for id in 0..count as u64 {
        let text = match mv.frame_text_by_id(id) {
            Ok(t) => t,
            Err(e) => {
                println!("\nframe #{id}: <unreadable: {e}>");
                continue;
            }
        };
        println!("\n┌─ frame #{id} ({} chars)", text.len());
        for line in text.lines() {
            println!("│ {line}");
        }
        println!("└{}", "─".repeat(60));
    }

    println!("\n{bar}");
    Ok(())
}
