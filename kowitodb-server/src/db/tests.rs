//! `KowitoDBEngine` integration tests (split from the former db.rs god-file).

use super::*;

#[tokio::test]
async fn test_insert_and_ask_end_to_end() {
    let engine = KowitoDBEngine::new_in_memory().unwrap();

    // Insert enterprise customer knowledge
    let acme = KnowledgeObject::new(
            "Acme Corp renewed their enterprise license in March 2024 after raising Series A funding of $15M."
        )
        .with_keywords(vec!["acme".into(), "renewal".into(), "series a".into(), "enterprise".into()])
        .with_metadata("company", "Acme Corp")
        .with_metadata("stage", "series_a")
        .with_metadata("renewed", "true")
        .with_importance(0.9);

    let globex = KnowledgeObject::new(
            "Globex Inc. received Series B funding of $30M in January 2024 and upgraded to enterprise tier."
        )
        .with_keywords(vec!["globex".into(), "series b".into(), "enterprise".into(), "funding".into()])
        .with_metadata("company", "Globex Inc.")
        .with_metadata("stage", "series_b")
        .with_metadata("renewed", "true");

    let initech = KnowledgeObject::new(
        "Initech went through Series A in 2023 but churned in December 2024 due to budget cuts.",
    )
    .with_keywords(vec!["initech".into(), "series a".into(), "churn".into()])
    .with_metadata("company", "Initech")
    .with_metadata("stage", "series_a")
    .with_metadata("renewed", "false");

    // Insert all
    engine.insert(acme).await.unwrap();
    engine.insert(globex).await.unwrap();
    engine.insert(initech).await.unwrap();

    // Ask a natural language question
    let response = engine
        .ask("Which enterprise customers renewed after Series A?", 5)
        .await
        .unwrap();

    // Verify we got results
    assert!(!response.results.is_empty(), "Should have results");
    println!(
        "Intent: {}, Results: {}",
        response.detected_intent,
        response.results.len()
    );

    // Results should contain real content, not placeholders
    for r in &response.results {
        assert!(
            !r.content.starts_with('<'),
            "Content should be real, got: {}",
            r.content
        );
        assert!(r.content.len() > 10, "Content too short: {}", r.content);
    }

    // Plan should be explained
    assert!(!response.plan_explanation.is_empty());
    println!("Plan:\n{}", response.plan_explanation);
}

#[tokio::test]
async fn test_insert_get_delete_roundtrip() {
    let engine = KowitoDBEngine::new_in_memory().unwrap();

    let obj = KnowledgeObject::new("Test content for roundtrip")
        .with_keywords(vec!["test".into()])
        .with_metadata("key", "value");

    let id = engine.insert(obj).await.unwrap();

    // Get it back
    let retrieved = engine.get(id).await.unwrap().expect("Object should exist");
    assert_eq!(retrieved.content, "Test content for roundtrip");
    assert_eq!(retrieved.keywords, vec!["test"]);

    // Delete
    let existed = engine.delete(id).await.unwrap();
    assert!(existed);

    // Should be gone
    let gone = engine.get(id).await.unwrap();
    assert!(gone.is_none());
}

#[tokio::test]
async fn test_graph_traversal_via_insert() {
    let engine = KowitoDBEngine::new_in_memory().unwrap();

    let openai = KnowledgeObject::new("OpenAI is an AI research lab").with_keywords(vec![
        "openai".into(),
        "ai".into(),
        "research".into(),
    ]);

    let ms = KnowledgeObject::new("Microsoft invested $10B in OpenAI")
        .with_keywords(vec![
            "microsoft".into(),
            "investment".into(),
            "openai".into(),
        ])
        .with_relationship("invested_in", openai.id);

    engine.insert(openai).await.unwrap();
    engine.insert(ms).await.unwrap();

    // Ask about companies connected to OpenAI
    let response = engine
        .ask("Which companies invested in OpenAI?", 5)
        .await
        .unwrap();

    println!(
        "Graph query results: {} (intent: {})",
        response.results.len(),
        response.detected_intent
    );
    // Should find Microsoft via graph traversal
    assert!(!response.results.is_empty());
}

#[tokio::test]
async fn test_reindex_rebuilds_in_memory_indexes_after_restart() {
    let base = std::env::temp_dir().join(format!("kowitodb-restart-{}", uuid::Uuid::new_v4()));
    let storage_path = base.join("storage");
    let index_path = base.join("index");
    std::fs::create_dir_all(&storage_path).unwrap();
    std::fs::create_dir_all(&index_path).unwrap();

    // First session: insert objects, then drop the engine (simulated shutdown).
    {
        let engine = KowitoDBEngine::open(&storage_path, &index_path)
            .await
            .unwrap();
        engine
            .insert(
                KnowledgeObject::new("Acme renewed their enterprise contract")
                    .with_keywords(vec!["acme".into(), "enterprise".into()])
                    .with_metadata("company", "Acme"),
            )
            .await
            .unwrap();
        engine
            .insert(
                KnowledgeObject::new("Globex churned last quarter")
                    .with_metadata("company", "Globex"),
            )
            .await
            .unwrap();
        assert_eq!(engine.stats().await.unwrap().vector_count, 2);
    }

    // Second session over the same paths. Without reindex the in-memory
    // indexes would be empty; open() must repopulate them from storage.
    let engine = KowitoDBEngine::open(&storage_path, &index_path)
        .await
        .unwrap();
    let stats = engine.stats().await.unwrap();
    assert_eq!(stats.total_objects, 2);
    assert_eq!(
        stats.vector_count, 2,
        "vector index must be rebuilt from persisted embeddings on restart"
    );

    // Metadata index rebuilt.
    assert_eq!(
        engine.metadata_index.query_exact("company", "Acme").len(),
        1
    );

    // End-to-end ask works after restart.
    let resp = engine.ask("enterprise contract", 5).await.unwrap();
    assert!(!resp.results.is_empty());

    let _ = std::fs::remove_dir_all(&base);
}

#[tokio::test]
async fn test_vector_index_checkpoint_and_reload() {
    let base = std::env::temp_dir().join(format!("kowitodb-ckpt-{}", uuid::Uuid::new_v4()));
    let storage_path = base.join("storage");
    let index_path = base.join("index");
    std::fs::create_dir_all(&storage_path).unwrap();
    std::fs::create_dir_all(&index_path).unwrap();

    {
        let engine = KowitoDBEngine::open(&storage_path, &index_path)
            .await
            .unwrap();
        for i in 0..3 {
            engine
                .insert(KnowledgeObject::new(format!("document {i}")))
                .await
                .unwrap();
        }
        assert_eq!(engine.stats().await.unwrap().vector_count, 3);
        engine.checkpoint().unwrap();
    }

    // The checkpoint wrote a snapshot.
    assert!(
        index_path.join("hnsw.bin").exists(),
        "checkpoint must write hnsw.bin"
    );

    // Reopen: the snapshot is loaded and vectors are intact + searchable.
    let engine = KowitoDBEngine::open(&storage_path, &index_path)
        .await
        .unwrap();
    assert_eq!(engine.stats().await.unwrap().vector_count, 3);
    assert!(!engine.ask("document", 5).await.unwrap().results.is_empty());

    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn test_retrieval_confidence() {
    // Empty → zero confidence (triggers corrective).
    assert_eq!(retrieval_confidence(&[], 5), 0.0);

    let mk = |sources: Vec<IndexSource>| RankedResult {
        id: uuid::Uuid::new_v4(),
        score: 1.0,
        sources,
        source_scores: HashMap::new(),
    };

    // One result, single source, 5 requested → low confidence.
    let sparse = vec![mk(vec![IndexSource::Vector])];
    assert!(retrieval_confidence(&sparse, 5) < CONFIDENCE_THRESHOLD);

    // Full coverage with cross-source agreement → high confidence.
    let strong: Vec<_> = (0..5)
        .map(|_| mk(vec![IndexSource::Vector, IndexSource::FullText]))
        .collect();
    assert!(retrieval_confidence(&strong, 5) > CONFIDENCE_THRESHOLD);
}

#[test]
fn test_contextualize_for_index() {
    let obj = KnowledgeObject::new("Quarterly results were strong.")
        .with_metadata("company", "Acme")
        .with_keywords(vec!["renewal".into()]);
    let text = contextualize_for_index(&obj);
    assert!(text.contains("company: Acme"));
    assert!(text.contains("Keywords: renewal"));
    assert!(text.contains("Quarterly results were strong."));
    // The object's stored content is never modified.
    assert_eq!(obj.content, "Quarterly results were strong.");
}

#[tokio::test]
async fn test_llm_contextualization_prepends_generated_context() {
    std::env::set_var("KOWITODB_LLM_CONTEXTUAL", "1");
    let mut engine = KowitoDBEngine::new_in_memory().unwrap();
    engine.llm_client = Some(std::sync::Arc::new(crate::llm::testing::MockLlm {
        response: "This is from Acme's Q3 earnings report.".into(),
    }));
    let obj = KnowledgeObject::new("Revenue grew 20%.");
    let text = engine.contextualize(&obj).await;
    assert!(
        text.contains("Acme's Q3 earnings report"),
        "LLM-generated context should be prepended to the indexed text"
    );
    assert!(text.contains("Revenue grew 20%."));
    std::env::remove_var("KOWITODB_LLM_CONTEXTUAL");
}

#[tokio::test]
async fn test_nl_to_sql_executes_against_store() {
    let mut engine = KowitoDBEngine::new_in_memory().unwrap();
    engine
        .insert(KnowledgeObject::new("Acme raised a Series A"))
        .await
        .unwrap();
    engine
        .insert(KnowledgeObject::new("Initech churned last quarter"))
        .await
        .unwrap();
    engine.llm_client = Some(std::sync::Arc::new(crate::llm::testing::MockLlm {
        response: "```sql\nSELECT content FROM knowledge;\n```".into(),
    }));
    let rows = engine
        .answer_with_sql("how many records are there?")
        .await
        .unwrap()
        .expect("LLM client present → Some rows");
    assert_eq!(rows.len(), 2, "NL→SQL query should run over the store");
}

#[tokio::test]
async fn test_memory_distillation_promotes_salient_fact() {
    let mut engine = KowitoDBEngine::new_in_memory().unwrap();
    engine.llm_client = Some(std::sync::Arc::new(crate::llm::testing::MockLlm {
        response: "The user prefers dark mode.".into(),
    }));
    engine
        .remember_turn(
            "s1",
            "user",
            "hey so, um, I really like dark mode I guess".into(),
        )
        .await
        .unwrap();
    // The promoted memory is the distilled fact, not the raw rambling turn.
    let resp = engine.ask("dark mode preference", 5).await.unwrap();
    assert!(
        resp.results
            .iter()
            .any(|r| r.content == "The user prefers dark mode."),
        "distilled fact should be the searchable memory"
    );
}

#[tokio::test]
async fn test_late_interaction_maxsim_retrieval() {
    let engine = KowitoDBEngine::new_in_memory().unwrap();
    let basis = |i: usize| {
        let mut v = vec![0.0f32; 4];
        v[i] = 1.0;
        v
    };
    let a = engine
        .insert(KnowledgeObject::new("alpha doc"))
        .await
        .unwrap();
    let b = engine
        .insert(KnowledgeObject::new("beta doc"))
        .await
        .unwrap();
    // Token vectors (from a hypothetical ColBERT model): A covers e0/e1, B e2/e3.
    engine.index_token_vectors(a, vec![basis(0), basis(1)]);
    engine.index_token_vectors(b, vec![basis(2), basis(3)]);

    let res = engine.late_interaction_search(&[basis(0)], 2, None);
    assert_eq!(res[0].0, a, "MaxSim ranks the token-matching doc first");

    // Delete drops its token vectors from the index too.
    engine.delete(a).await.unwrap();
    let res = engine.late_interaction_search(&[basis(0)], 2, None);
    assert!(res.iter().all(|(id, _)| *id != a));
}

#[tokio::test]
async fn test_graphrag_community_summaries_and_global_query() {
    let mut engine = KowitoDBEngine::new_in_memory().unwrap();
    // Two entity clusters; the auto-graph links each cluster internally.
    for c in [
        "Acme launched Rocket",
        "Acme hired Director",
        "Acme raised Capital",
    ] {
        engine.insert(KnowledgeObject::new(c)).await.unwrap();
    }
    for c in ["Globex shipped Gadget", "Globex acquired Startup"] {
        engine.insert(KnowledgeObject::new(c)).await.unwrap();
    }

    engine.llm_client = Some(std::sync::Arc::new(crate::llm::testing::MockLlm {
        response: "Two companies, Acme and Globex, are active.".into(),
    }));

    // Two communities (Acme ×3, Globex ×2) are detected and summarized.
    let n = engine.build_community_summaries().await.unwrap();
    assert_eq!(n, 2, "two entity clusters → two community summaries");

    // Global map-reduce query produces a holistic answer.
    let answer = engine
        .global_query("Which companies are mentioned?")
        .await
        .unwrap()
        .expect("LLM present → Some answer");
    assert!(answer.contains("Acme") && answer.contains("Globex"));
}

#[tokio::test]
async fn test_graphrag_global_query_without_llm_is_none() {
    let engine = KowitoDBEngine::new_in_memory().unwrap();
    engine
        .insert(KnowledgeObject::new("Acme launched Rocket"))
        .await
        .unwrap();
    engine
        .insert(KnowledgeObject::new("Acme hired Director"))
        .await
        .unwrap();
    // No LLM client → global query falls back (None), no panic.
    assert!(engine.global_query("themes?").await.unwrap().is_none());
}

#[tokio::test]
async fn test_persistence_survives_restart() {
    let dir = std::env::temp_dir().join(format!("kowitodb-durab-{}", uuid::Uuid::new_v4()));
    let storage = dir.join("storage");
    let index = dir.join("index");

    // Session 1: ingest, checkpoint the vector index, then drop the engine
    // (the `{}` scope releases sled's file lock — a clean "shutdown").
    {
        let engine = KowitoDBEngine::open(&storage, &index).await.unwrap();
        for c in ["Acme renewed their enterprise license", "Globex shipped v2"] {
            engine.insert(KnowledgeObject::new(c)).await.unwrap();
        }
        engine.checkpoint().unwrap();
    }

    // Session 2: reopen from the same paths and verify the data recovered
    // and is searchable (the vector index rebuilds from storage on open).
    let engine = KowitoDBEngine::open(&storage, &index).await.unwrap();
    let stats = engine.stats().await.unwrap();
    assert_eq!(stats.total_objects, 2, "objects must survive a restart");
    let resp = engine.ask("Acme enterprise", 5).await.unwrap();
    assert!(
        resp.results.iter().any(|r| r.content.contains("Acme")),
        "recovered data must be searchable"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn test_auto_graph_links_co_mentions() {
    let engine = KowitoDBEngine::new_in_memory().unwrap();
    let a = engine
        .insert(KnowledgeObject::new(
            "Acme Corporation raised a Series A round.",
        ))
        .await
        .unwrap();
    let b = engine
        .insert(KnowledgeObject::new(
            "Later, Acme Corporation hired a new CEO.",
        ))
        .await
        .unwrap();
    // Shared entity "Acme"/"Corporation" → an auto co_mentions edge b↔a.
    let out_b = engine.graph_index.out_edges(b);
    assert!(
        out_b
            .iter()
            .any(|r| r.target_id == a && r.relation_type == "co_mentions"),
        "auto-graph should link co-mentioning objects"
    );
    // And it is bidirectional.
    assert!(engine
        .graph_index
        .out_edges(a)
        .iter()
        .any(|r| r.target_id == b));
}

#[tokio::test]
async fn test_strip_sql_fence() {
    assert_eq!(strip_sql_fence("```sql\nSELECT 1;\n```"), "SELECT 1");
    assert_eq!(
        strip_sql_fence("SELECT * FROM knowledge"),
        "SELECT * FROM knowledge"
    );
}

#[test]
fn test_is_read_only_sql() {
    assert!(is_read_only_sql("SELECT content FROM knowledge"));
    assert!(is_read_only_sql("  with x as (select 1) select * from x  "));
    assert!(is_read_only_sql(
        "SELECT count(*) FROM knowledge WHERE created_at > '2020'"
    ));
    // Writes / DDL / chaining / filesystem are rejected.
    assert!(!is_read_only_sql("DROP TABLE knowledge"));
    assert!(!is_read_only_sql("SELECT 1; DROP TABLE knowledge"));
    assert!(!is_read_only_sql("COPY knowledge TO '/tmp/x.csv'"));
    assert!(!is_read_only_sql("CREATE TABLE t AS SELECT 1"));
}

#[tokio::test]
async fn test_contextual_retrieval_finds_metadata_terms() {
    let engine = KowitoDBEngine::new_in_memory().unwrap();
    // Content does NOT mention "Acme"; only metadata does.
    let id = engine
        .insert(
            KnowledgeObject::new("Quarterly results were strong and the team grew.")
                .with_metadata("company", "Acme"),
        )
        .await
        .unwrap();

    // The metadata term is findable because it was folded into the embedded /
    // full-text-indexed text (Contextual Retrieval).
    let resp = engine.ask("Acme", 5).await.unwrap();
    assert!(
        resp.results.iter().any(|r| r.id == id.to_string()),
        "contextual retrieval should make metadata-only terms findable"
    );

    // Stored content remains the original (un-augmented).
    let stored = engine.get(id).await.unwrap().unwrap();
    assert_eq!(
        stored.content,
        "Quarterly results were strong and the team grew."
    );
}

#[tokio::test]
async fn test_memory_promoted_to_searchable_knowledge() {
    let engine = KowitoDBEngine::new_in_memory().unwrap();

    let n = engine
        .remember_turn("sess-1", "user", "I love hiking in the mountains".into())
        .await
        .unwrap();
    assert_eq!(n, 1);
    // Recorded in agent memory.
    assert_eq!(engine.agent_memory.get("sess-1").unwrap().turn_count(), 1);

    // Retrievable as knowledge via ai.ask().
    let resp = engine.ask("hiking", 5).await.unwrap();
    assert!(
        resp.results.iter().any(|r| r.content.contains("hiking")),
        "promoted memory should be retrievable"
    );

    // Idempotent: re-recording the same turn does not duplicate the memory.
    engine
        .remember_turn("sess-1", "user", "I love hiking in the mountains".into())
        .await
        .unwrap();
    let (objects, _) = engine.list(0, 100).await.unwrap();
    let memories = objects
        .iter()
        .filter(|o| o.metadata.get("kind").and_then(|v| v.as_str()) == Some("memory"))
        .count();
    assert_eq!(memories, 1, "duplicate memory must be deduped by stable id");

    // System turns are recorded but not promoted to knowledge.
    engine
        .remember_turn("sess-1", "system", "you are a helpful assistant".into())
        .await
        .unwrap();
    let (objects, _) = engine.list(0, 100).await.unwrap();
    assert!(
        !objects
            .iter()
            .any(|o| o.content.contains("helpful assistant")),
        "system turns are not promoted"
    );
}

#[tokio::test]
async fn test_memory_links_to_related_knowledge() {
    let engine = KowitoDBEngine::new_in_memory().unwrap();

    // An existing knowledge entity.
    let acme_id = engine
        .insert(
            KnowledgeObject::new("Acme Corp raised a Series A funding round")
                .with_keywords(vec!["acme".into()]),
        )
        .await
        .unwrap();

    // A turn that mentions it → the promoted memory links to it in the graph.
    engine
        .remember_turn("s1", "user", "I met with Acme about the renewal".into())
        .await
        .unwrap();

    let mem_id = stable_memory_id("s1", "I met with Acme about the renewal");
    let memory = engine.get(mem_id).await.unwrap().unwrap();
    assert!(
        memory
            .relationships
            .iter()
            .any(|r| r.target_id == acme_id && r.relation_type == "mentions"),
        "memory should be graph-linked to the Acme entity it mentions"
    );
}

#[tokio::test]
async fn test_update_and_versioning() {
    let engine = KowitoDBEngine::new_in_memory().unwrap();
    let id = engine
        .insert(KnowledgeObject::new("original content").with_metadata("k", "v1"))
        .await
        .unwrap();

    // First update: change content + metadata + importance.
    let v = engine
        .update(
            id,
            Some("updated content".into()),
            HashMap::from([("k".to_string(), "v2".to_string())]),
            vec![],
            Some(0.9),
            Some("edit 1".into()),
        )
        .await
        .unwrap();
    assert_eq!(v, Some(1));

    let obj = engine.get(id).await.unwrap().unwrap();
    assert_eq!(obj.content, "updated content");
    assert_eq!(obj.metadata.get("k").and_then(|x| x.as_str()), Some("v2"));
    assert!((obj.importance - 0.9).abs() < 1e-6);
    assert_eq!(obj.version_history.len(), 1);

    // Second update accumulates history — proving versions persist across
    // storage round-trips.
    let v2 = engine
        .update(
            id,
            None,
            HashMap::new(),
            vec![],
            None,
            Some("edit 2".into()),
        )
        .await
        .unwrap();
    assert_eq!(v2, Some(2));
    assert_eq!(
        engine.get(id).await.unwrap().unwrap().version_history.len(),
        2
    );

    // Updating a missing object returns None.
    let missing = engine
        .update(
            uuid::Uuid::new_v4(),
            Some("x".into()),
            HashMap::new(),
            vec![],
            None,
            None,
        )
        .await
        .unwrap();
    assert_eq!(missing, None);
}

#[tokio::test]
async fn test_batch_insert_and_list_pagination() {
    let engine = KowitoDBEngine::new_in_memory().unwrap();
    let objs: Vec<_> = (0..5)
        .map(|i| KnowledgeObject::new(format!("document number {i}")).with_metadata("kind", "note"))
        .collect();
    let ids = engine.batch_insert(objs).await.unwrap();
    assert_eq!(ids.len(), 5);

    let (page, total) = engine.list(0, 2).await.unwrap();
    assert_eq!(total, 5);
    assert_eq!(page.len(), 2);

    let (last, total) = engine.list(4, 10).await.unwrap();
    assert_eq!(total, 5);
    assert_eq!(last.len(), 1);

    // Offset past the end yields an empty page but the correct total.
    let (none, total) = engine.list(99, 10).await.unwrap();
    assert_eq!(total, 5);
    assert!(none.is_empty());
}

#[tokio::test]
async fn test_importance_weighted_ranking() {
    let engine = KowitoDBEngine::new_in_memory().unwrap();
    let low = engine
        .insert(
            KnowledgeObject::new("enterprise widget alpha")
                .with_keywords(vec!["enterprise".into()])
                .with_importance(0.1),
        )
        .await
        .unwrap();
    let high = engine
        .insert(
            KnowledgeObject::new("enterprise widget beta")
                .with_keywords(vec!["enterprise".into()])
                .with_importance(0.9),
        )
        .await
        .unwrap();

    let resp = engine.ask("enterprise widget", 5).await.unwrap();
    let pos = |id: ObjectId| resp.results.iter().position(|r| r.id == id.to_string());
    let (hp, lp) = (pos(high), pos(low));
    assert!(hp.is_some() && lp.is_some(), "both should be retrieved");
    assert!(
        hp.unwrap() < lp.unwrap(),
        "higher-importance object should rank above the lower-importance one"
    );
}

struct MockReranker;
#[async_trait::async_trait]
impl CrossEncoder for MockReranker {
    async fn rerank(&self, _query: &str, documents: &[String]) -> Vec<f32> {
        // Score documents mentioning "beta" much higher.
        documents
            .iter()
            .map(|d| if d.contains("beta") { 10.0 } else { 1.0 })
            .collect()
    }
}

#[tokio::test]
async fn test_cross_encoder_reranks_results() {
    let mut engine = KowitoDBEngine::new_in_memory().unwrap();
    engine
        .insert(
            KnowledgeObject::new("enterprise widget alpha")
                .with_keywords(vec!["enterprise".into()]),
        )
        .await
        .unwrap();
    engine
        .insert(
            KnowledgeObject::new("enterprise widget beta").with_keywords(vec!["enterprise".into()]),
        )
        .await
        .unwrap();

    // Inject a cross-encoder that prefers "beta"; it should reorder results.
    engine.reranker_model = Some(std::sync::Arc::new(MockReranker));
    let resp = engine.ask("enterprise widget", 5).await.unwrap();
    assert!(
        resp.results
            .first()
            .map(|r| r.content.contains("beta"))
            .unwrap_or(false),
        "cross-encoder's preferred document should rank first"
    );
}

#[tokio::test]
async fn test_recency_weighted_ranking() {
    let engine = KowitoDBEngine::new_in_memory().unwrap();
    // Same importance, same match terms; only age differs.
    let mut old = KnowledgeObject::new("enterprise widget historical")
        .with_keywords(vec!["enterprise".into()]);
    old.created_at = chrono::Utc::now() - chrono::Duration::days(120);
    let old_id = engine.insert(old).await.unwrap();
    let new_id = engine
        .insert(
            KnowledgeObject::new("enterprise widget current")
                .with_keywords(vec!["enterprise".into()]),
        )
        .await
        .unwrap();

    let resp = engine.ask("enterprise widget", 5).await.unwrap();
    let pos = |id: ObjectId| resp.results.iter().position(|r| r.id == id.to_string());
    assert!(
        pos(new_id).unwrap() < pos(old_id).unwrap(),
        "more recent object should rank above the older one"
    );
}

#[tokio::test]
async fn test_ask_with_metadata_filter() {
    let engine = KowitoDBEngine::new_in_memory().unwrap();
    engine
        .insert(
            KnowledgeObject::new("Acme enterprise renewal closed")
                .with_metadata("company", "Acme")
                .with_keywords(vec!["enterprise".into(), "renewal".into()]),
        )
        .await
        .unwrap();
    engine
        .insert(
            KnowledgeObject::new("Globex enterprise renewal closed")
                .with_metadata("company", "Globex")
                .with_keywords(vec!["enterprise".into(), "renewal".into()]),
        )
        .await
        .unwrap();

    let filter = HashMap::from([("company".to_string(), "Acme".to_string())]);
    let resp = engine
        .ask_filtered("enterprise renewal", 10, None, &filter)
        .await
        .unwrap();

    assert!(!resp.results.is_empty(), "filter excluded everything");
    for r in &resp.results {
        assert!(
            r.content.contains("Acme") && !r.content.contains("Globex"),
            "metadata filter leaked a non-matching object: {}",
            r.content
        );
    }
}

#[tokio::test]
async fn test_ask_honors_context_token_budget() {
    let engine = KowitoDBEngine::new_in_memory().unwrap();
    for i in 0..5 {
        engine
            .insert(KnowledgeObject::new(format!(
                "Document number {i} about enterprise renewals and funding rounds \
                     with enough words to consume a non-trivial number of tokens each."
            )))
            .await
            .unwrap();
    }

    // A tiny budget should yield a smaller assembled context than a large one.
    let small = engine
        .ask_with_budget("enterprise renewals", 5, Some(20))
        .await
        .unwrap();
    let large = engine
        .ask_with_budget("enterprise renewals", 5, Some(4096))
        .await
        .unwrap();
    assert!(small.total_tokens <= large.total_tokens);
    assert!(small.total_tokens <= 100, "small budget was not honored");
}

#[tokio::test]
async fn test_sql_select_datafusion_aggregate_and_order() {
    let engine = KowitoDBEngine::new_in_memory().unwrap();

    engine
        .insert(
            KnowledgeObject::new("Acme Corp content")
                .with_metadata("stage", "series_a")
                .with_importance(0.9),
        )
        .await
        .unwrap();
    engine
        .insert(
            KnowledgeObject::new("Globex Inc. content")
                .with_metadata("stage", "series_b")
                .with_importance(0.4),
        )
        .await
        .unwrap();
    engine
        .insert(
            KnowledgeObject::new("Initech content")
                .with_metadata("stage", "series_a")
                .with_importance(0.7),
        )
        .await
        .unwrap();

    // Aggregate via DataFusion (not expressible through the index-routed path).
    let rows = engine
        .sql_select("SELECT COUNT(*) AS n FROM knowledge")
        .await
        .unwrap();
    assert_eq!(rows[0]["n"], "3");

    // Projection + filter + ORDER BY, all executed by DataFusion.
    let rows = engine
        .sql_select(
            "SELECT content, importance FROM knowledge \
                 WHERE importance >= 0.5 ORDER BY importance DESC",
        )
        .await
        .unwrap();
    assert_eq!(rows.len(), 2);
    assert!(rows[0]["content"].contains("Acme"));
    assert!(rows[1]["content"].contains("Initech"));
}

#[tokio::test]
async fn test_sql_query_metadata_filter() {
    let engine = KowitoDBEngine::new_in_memory().unwrap();

    let acme = KnowledgeObject::new("Acme Corp content")
        .with_keywords(vec!["acme".into()])
        .with_metadata("company", "Acme Corp")
        .with_metadata("stage", "series_a");
    let globex = KnowledgeObject::new("Globex Inc. content")
        .with_keywords(vec!["globex".into()])
        .with_metadata("company", "Globex Inc.")
        .with_metadata("stage", "series_b");

    engine.insert(acme).await.unwrap();
    engine.insert(globex).await.unwrap();

    // SQL: filter by metadata
    let results = engine
        .sql_query("SELECT * FROM knowledge WHERE metadata.stage = 'series_a'")
        .await
        .unwrap();

    assert_eq!(results.len(), 1);
    assert!(results[0].content.contains("Acme"));

    // SQL: with LIMIT
    let results = engine
        .sql_query("SELECT content FROM knowledge WHERE metadata.company LIKE '%Inc%' LIMIT 5")
        .await
        .unwrap();

    assert_eq!(results.len(), 1);
    assert!(results[0].content.contains("Globex"));
}
