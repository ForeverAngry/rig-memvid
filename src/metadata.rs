//! Typed metadata envelope written into
//! [`memvid_core::SearchHitMetadata::extra_metadata`] for every frame
//! produced by `rig-memvid` lifecycle hooks.
//!
//! Implementation lives in [`rig_memory_policy::metadata`]; this module
//! re-exports those items, with [`MemvidFrameMetadata`] preserved as a
//! backward-compatible type alias for the published name
//! (see `rig-memvid#28`).
// re-exported from rig-memory-policy for backward compat

pub use rig_memory_policy::metadata::{FrameKind, FrameMetadata};

/// Backward-compatible alias for [`FrameMetadata`].
///
/// Existing downstream code that named this type as `MemvidFrameMetadata`
/// continues to compile against the new policy crate without changes.
pub type MemvidFrameMetadata = FrameMetadata;
