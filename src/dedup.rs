//! In-process content-hash dedup shim.
//!
//! All implementation lives in [`rig_memory_policy::dedup`]; this module
//! re-exports the items historically consumed via `crate::dedup` so the
//! hook, compactor, and frame-writer call sites keep their existing import
//! paths after the policy extraction (see `rig-memvid#28`).
// re-exported from rig-memory-policy for backward compat

pub(crate) use rig_memory_policy::dedup::{DedupSet, compute_key, hex_encode_key};
