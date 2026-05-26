//! Shared example-only harness record.
//!
//! Kept under `examples/` while the ecosystem proves the shape across more
//! than one backend. Do not promote this to a library module until the schema
//! has at least three real producers.

use rig_compose::{ToolInvocation, ToolInvocationResult};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Serializable tool invocation mirror.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HarnessToolInvocation {
    /// Tool name selected by the model output normalizer.
    pub name: String,
    /// JSON arguments sent to the tool.
    pub args: Value,
}

impl From<&ToolInvocation> for HarnessToolInvocation {
    fn from(value: &ToolInvocation) -> Self {
        Self {
            name: value.name.clone(),
            args: value.args.clone(),
        }
    }
}

/// Serializable tool dispatch result mirror.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HarnessToolResult {
    /// Invocation that produced this result.
    pub invocation: HarnessToolInvocation,
    /// JSON result returned by the tool.
    pub output: Value,
}

impl From<&ToolInvocationResult> for HarnessToolResult {
    fn from(value: &ToolInvocationResult) -> Self {
        Self {
            invocation: HarnessToolInvocation::from(&value.invocation),
            output: value.output.clone(),
        }
    }
}

/// Serializable one-turn tool-loop harness record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HarnessRun {
    /// Schema version for the example harness record.
    pub schema_version: u32,
    /// Producer crate or example id.
    pub producer: String,
    /// User task given to the first model turn.
    pub task: String,
    /// Raw first model output before normalization.
    pub first_model_output: String,
    /// Normalized tool invocations.
    pub invocations: Vec<HarnessToolInvocation>,
    /// Tool dispatch results.
    pub dispatch_results: Vec<HarnessToolResult>,
    /// Final answer after tool results are folded back in.
    pub final_answer: String,
    /// Assertions that passed for this run.
    pub passed_assertions: Vec<String>,
}

impl HarnessRun {
    /// Current schema version.
    pub const SCHEMA_VERSION: u32 = 1;

    /// Build a harness run from native `rig-compose` invocation/result types.
    #[allow(dead_code)]
    pub fn from_native(
        producer: impl Into<String>,
        task: impl Into<String>,
        first_model_output: impl Into<String>,
        invocations: &[ToolInvocation],
        dispatch_results: &[ToolInvocationResult],
        final_answer: impl Into<String>,
        passed_assertions: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self {
            schema_version: Self::SCHEMA_VERSION,
            producer: producer.into(),
            task: task.into(),
            first_model_output: first_model_output.into(),
            invocations: invocations
                .iter()
                .map(HarnessToolInvocation::from)
                .collect(),
            dispatch_results: dispatch_results
                .iter()
                .map(HarnessToolResult::from)
                .collect(),
            final_answer: final_answer.into(),
            passed_assertions: passed_assertions.into_iter().map(Into::into).collect(),
        }
    }
}
