//! Shared write path for the compaction integration.
//!
//! Both [`crate::MemvidDemotionHook`] and [`crate::MemvidStoringCompactor`]
//! land single rendered chunks of text into a [`MemvidStore`] with the
//! same metadata schema, the same content-hash dedup gate, and the same
//! error-propagation contract. This module factors that path out so the
//! two surfaces stay in lockstep.
//!
//! Gated on the `compaction` feature.

use memvid_core::PutOptions;
use rig::memory::MemoryError;

use crate::dedup::{DedupSet, compute_key, hex_encode_key};
use crate::hook::MemoryConfig;
use crate::metadata::{FrameKind, MemvidFrameMetadata};
use crate::store::MemvidStore;

/// Write a single rendered frame into `store`, gated by `dedup`.
///
/// Returns `Ok(true)` if the frame was newly written, `Ok(false)` if a
/// content-identical frame was already seen in this process (dedup hit
/// — caller can treat as a no-op success), and `Err(MemoryError)` if
/// any of the underlying operations failed.
///
/// The frame carries the metadata schema documented on
/// [`crate::MemvidDemotionHook`] / [`crate::MemvidStoringCompactor`]:
/// `kind`, `conversation_id`, `chat_role`, `dedup_key`, and `scope`
/// (when configured).
pub(crate) fn write_frame(
    store: &MemvidStore,
    config: &MemoryConfig,
    dedup: &DedupSet,
    kind: FrameKind,
    conversation_id: &str,
    chat_role: &str,
    text: &str,
) -> Result<bool, MemoryError> {
    if text.is_empty() {
        return Ok(false);
    }
    let scope = config.scope.as_deref();
    let key = compute_key(kind.as_str(), conversation_id, chat_role, scope, text);

    if dedup
        .contains(&key)
        .map_err(|err| MemoryError::backend(Box::new(err)))?
    {
        tracing::debug!(
            target: "rig_memvid::frame_writer",
            conversation_id,
            role = chat_role,
            kind = kind.as_str(),
            "skipping duplicate frame",
        );
        return Ok(false);
    }

    let key_hex = hex_encode_key(&key);
    let opts = build_put_options(config, kind, conversation_id, chat_role, &key_hex);
    if let Err(err) = store.put_text_uncommitted(text, opts) {
        tracing::warn!(
            target: "rig_memvid::frame_writer",
            error = %err,
            conversation_id,
            role = chat_role,
            kind = kind.as_str(),
            "failed to persist frame into memvid",
        );
        return Err(MemoryError::backend(Box::new(err)));
    }
    dedup
        .insert(key)
        .map_err(|err| MemoryError::backend(Box::new(err)))?;
    Ok(true)
}

/// Commit the underlying archive if [`MemoryConfig::commit_each_turn`]
/// is enabled. No-op otherwise.
pub(crate) fn commit_if_each_turn(
    store: &MemvidStore,
    config: &MemoryConfig,
) -> Result<(), MemoryError> {
    if config.commit_each_turn {
        store
            .commit()
            .map_err(|err| MemoryError::backend(Box::new(err)))?;
    }
    Ok(())
}

fn build_put_options(
    config: &MemoryConfig,
    kind: FrameKind,
    conversation_id: &str,
    chat_role: &str,
    dedup_key_hex: &str,
) -> PutOptions {
    let mut opts = PutOptions {
        tags: config.default_tags.clone(),
        auto_tag: config.auto_tag,
        extract_dates: config.extract_dates,
        extract_triplets: config.extract_triplets,
        ..PutOptions::default()
    };
    if let Some(scope) = config.scope.as_deref() {
        opts.uri = Some(scope.to_string());
    }
    let metadata = MemvidFrameMetadata {
        schema_version: 1,
        kind,
        conversation_id: conversation_id.to_string(),
        chat_role: chat_role.to_string(),
        dedup_key: dedup_key_hex.to_string(),
        scope: config.scope.clone(),
    };
    opts.extra_metadata = metadata.into_map();
    opts
}
