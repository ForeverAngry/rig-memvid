#![allow(clippy::unwrap_used, clippy::panic, clippy::panic_in_result_fn)]

use anyhow::Result;
use rig_memory_policy::PolicyError;
use rig_memvid::inmem as module_path;

#[derive(Clone, Debug)]
struct Finding {
    summary: &'static str,
}

impl module_path::Episode for Finding {
    fn summary(&self) -> &str {
        self.summary
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn module_path_reexports_policy_inmem_store() -> Result<()> {
    let store = module_path::InMemoryStore::<Finding>::new();
    let key = store
        .append(Finding {
            summary: "scheduled maintenance window",
        })
        .await?;

    let hit = store.retrieve_similar("maintenance", 1).await?.remove(0);

    assert_eq!(hit.key, key);
    assert_eq!(hit.episode.summary, "scheduled maintenance window");
    assert_eq!(hit.score, 1.0);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn top_level_reexports_keep_historic_import_path() -> Result<()> {
    use rig_memvid::{Episode, InMemoryHit, InMemoryStore};

    fn assert_hit_shape<E: Episode>(hit: &InMemoryHit<E>) -> (&str, f32) {
        (hit.episode.summary(), hit.score)
    }

    let store = InMemoryStore::<Finding>::new();
    store
        .append(Finding {
            summary: "incident response notes",
        })
        .await?;

    let hits = store.retrieve_similar("incident", 5).await?;
    let hit = hits.first().unwrap();
    let (summary, score) = assert_hit_shape(hit);

    assert_eq!(summary, "incident response notes");
    assert_eq!(score, 1.0);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn in_memory_error_aliases_policy_error() -> Result<()> {
    fn accept_policy_error(error: rig_memvid::InMemoryError) -> PolicyError {
        error
    }

    let store = rig_memvid::InMemoryStore::<Finding>::new();
    let error = store.get("missing").await.unwrap_err();
    let policy_error = accept_policy_error(error);

    assert!(matches!(policy_error, PolicyError::NotFound(key) if key == "missing"));
    Ok(())
}
