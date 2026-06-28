//! Agent memory — the differentiated "agent-memory OS" thesis. Conversation
//! turns are recorded AND promoted to searchable knowledge automatically, so
//! past conversation is retrievable later via `ai.ask()` *alongside* ingested
//! documents — in one engine, no glue. All in-process, no server.
//!
//! Run:  cargo run -p kowitodb-server --example agent_memory

use kowitodb_core::KnowledgeObject;
use kowitodb_server::KowitoDBEngine;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let engine = KowitoDBEngine::new_in_memory()?;

    // 1. Background knowledge (documents you ingest).
    engine
        .insert(KnowledgeObject::new(
            "Acme Corp is on the Enterprise plan, renewed in March 2024.",
        ))
        .await?;

    // 2. A conversation. Each turn is recorded in the session AND promoted into
    //    a searchable knowledge object (Mem0-style), linked into the graph.
    let session = "user-42";
    for (role, text) in [
        (
            "user",
            "By the way, I prefer dark mode and concise answers.",
        ),
        ("assistant", "Noted — dark mode and concise it is."),
        ("user", "Remind me which plan Acme is on?"),
    ] {
        engine
            .remember_turn(session, role, text.to_string())
            .await?;
    }

    // 3. Later: a single ai.ask() spans BOTH the conversation memory and the
    //    documents — no separate memory store, no separate retrieval pass.
    for q in ["What are the user's preferences?", "What plan is Acme on?"] {
        println!("❯ ai.ask(\"{q}\")");
        let resp = engine.ask(q, 3).await?;
        for r in &resp.results {
            println!("  [{:.2}] {}", r.relevance_score, r.content);
        }
        println!();
    }
    Ok(())
}
