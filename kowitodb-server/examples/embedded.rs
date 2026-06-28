//! Use KowitoDB as an **embedded library** — no gRPC server, all in-process.
//! Build an engine, ingest a few facts, and run `ai.ask()` across every index.
//!
//! Run:  cargo run -p kowitodb-server --example embedded

use kowitodb_core::KnowledgeObject;
use kowitodb_server::KowitoDBEngine;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // In-memory engine (no disk). To persist across restarts, use
    // `KowitoDBEngine::open(storage_path, index_path).await?` instead.
    let engine = KowitoDBEngine::new_in_memory()?;

    let facts = [
        (
            "Acme Corp renewed their enterprise license in March 2024 after a $15M Series A.",
            vec!["acme", "renewal", "series a"],
            "Acme Corp",
        ),
        (
            "Globex shipped their v2 platform and onboarded three enterprise customers.",
            vec!["globex", "launch"],
            "Globex",
        ),
    ];
    for (text, keywords, company) in facts {
        let obj = KnowledgeObject::new(text)
            .with_keywords(keywords.into_iter().map(String::from).collect())
            .with_metadata("company", company);
        engine.insert(obj).await?;
    }

    // One call fans out across vector + full-text + graph + metadata + time,
    // reranks, and assembles the answer.
    let resp = engine
        .ask("Which companies had enterprise activity?", 5)
        .await?;

    println!("detected intent: {}", resp.detected_intent);
    for r in &resp.results {
        println!(
            "  [{:.2}] ({}) {}",
            r.relevance_score, r.retrieval_source, r.content
        );
    }
    Ok(())
}
