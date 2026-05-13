//! Typed metadata envelope for frames written by Rig compaction hooks.
//!
//! Gated on the `compaction` feature.

use serde::{Deserialize, Serialize};

/// The type of frame produced by a memory lifecycle hook.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum FrameKind {
    /// A single user/assistant interaction evicted from the working set.
    DemotedMessage,
    /// A rolled-up summary of multiple evicted interactions.
    CompactionSummary,
}

/// The typed schema written into [`memvid_core::SearchHitMetadata::extra_metadata`]
/// for every frame produced by `rig-memvid` lifecycle hooks.
///
/// Downstream tools (evals, memory inspectors, RAG pipelines) can decode
/// the `extra_metadata` map into this struct to reason about the origin
/// of the retrieved frame.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemvidFrameMetadata {
    /// The schema version of this envelope. Currently `1`.
    pub schema_version: u8,
    /// What kind of memory lifecycle produced this frame.
    pub kind: FrameKind,
    /// The conversation ID from the Rig `CompactingMemory` context.
    pub conversation_id: String,
    /// The logical chat role (`system`, `user`, `assistant`).
    pub chat_role: String,
    /// The deterministic content-hash dedup key.
    pub dedup_key: String,
    /// The isolation scope under which this frame was written, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
}

impl MemvidFrameMetadata {
    /// Attempt to parse the metadata envelope from a string-to-string dictionary.
    pub fn try_from_map(
        map: &std::collections::BTreeMap<String, String>,
    ) -> Result<Self, serde_json::Error> {
        let mut obj = serde_json::Map::new();
        #[allow(clippy::collapsible_if)]
        for (k, v) in map {
            if let Ok(num) = v.parse::<u8>() {
                if k == "schema_version" {
                    obj.insert(k.clone(), serde_json::Value::Number(num.into()));
                    continue;
                }
            }
            obj.insert(k.clone(), serde_json::Value::String(v.clone()));
        }
        serde_json::from_value(serde_json::Value::Object(obj))
    }

    /// Convert the metadata envelope into a string-to-string dictionary.
    pub fn into_map(self) -> std::collections::BTreeMap<String, String> {
        let mut map = std::collections::BTreeMap::new();
        map.insert(
            "schema_version".to_string(),
            self.schema_version.to_string(),
        );
        map.insert("kind".to_string(), self.kind.as_str().to_string());
        map.insert("conversation_id".to_string(), self.conversation_id);
        map.insert("chat_role".to_string(), self.chat_role);
        map.insert("dedup_key".to_string(), self.dedup_key);
        if let Some(scope) = self.scope {
            map.insert("scope".to_string(), scope);
        }
        map
    }
}

impl FrameKind {
    /// Provide the snake_case string representation.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::DemotedMessage => "demoted_message",
            Self::CompactionSummary => "compaction_summary",
        }
    }
}
