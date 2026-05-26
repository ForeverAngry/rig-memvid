#[path = "common/harness_record.rs"]
mod harness_record;

use harness_record::HarnessRun;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let run = HarnessRun {
        schema_version: HarnessRun::SCHEMA_VERSION,
        producer: "rig-memvid/examples/harness_record".to_string(),
        task: "Project a memory-backed retrieval step into the shared harness record.".to_string(),
        first_model_output: "Memvid retrieval supplied context before final answer.".to_string(),
        invocations: Vec::new(),
        dispatch_results: Vec::new(),
        final_answer: "Harness record shape is shared with rig-compose.".to_string(),
        passed_assertions: vec!["schema-shared".to_string()],
    };

    println!("{}", serde_json::to_string_pretty(&run)?);
    Ok(())
}
