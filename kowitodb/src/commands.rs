//! Subcommand dispatch for the kowitodb binary.

use clap::CommandFactory;
use tracing::info;

use kowitodb_core::KnowledgeObject;
use kowitodb_server::{serve_gateway, serve_with_config, KowitoDBEngine, ServerConfig};

use crate::cli::{Cli, Commands, StorageKind};

/// Execute the parsed subcommand.
pub async fn run(command: Commands) -> anyhow::Result<()> {
    match command {
        Commands::Serve {
            addr,
            storage_path,
            index_path,
            storage,
            lance_uri,
            max_results,
            api_key,
            tls_cert,
            tls_key,
            metrics_addr,
        } => {
            info!("Starting KowitoDB v{}", env!("CARGO_PKG_VERSION"));

            std::fs::create_dir_all(&storage_path)?;
            std::fs::create_dir_all(&index_path)?;

            let engine = match storage {
                StorageKind::Sled => KowitoDBEngine::open(&storage_path, &index_path).await,
                StorageKind::Lance => {
                    #[cfg(feature = "lance")]
                    {
                        let uri = lance_uri.unwrap_or_else(|| {
                            storage_path.join("lance").to_string_lossy().into_owned()
                        });
                        KowitoDBEngine::new_with_lance(uri, &index_path).await
                    }
                    #[cfg(not(feature = "lance"))]
                    {
                        let _ = &lance_uri;
                        anyhow::bail!(
                            "The Lance backend requires building with --features lance \
                             (e.g. `cargo build -p kowitodb --features lance`)."
                        );
                    }
                }
            }
            .map_err(|e| anyhow::anyhow!("Failed to initialize engine: {}", e))?;

            let config = ServerConfig {
                api_key,
                tls_cert,
                tls_key,
                metrics_addr,
                max_results: Some(max_results),
            };
            serve_with_config(engine, addr, config).await?;
        }

        Commands::Gateway {
            addr,
            peers,
            replication_factor,
            write_quorum,
            api_key,
        } => {
            info!("Starting KowitoDB gateway v{}", env!("CARGO_PKG_VERSION"));
            if peers.is_empty() {
                anyhow::bail!(
                    "--peers is required: a comma-separated list of data node host:port addresses"
                );
            }
            serve_gateway(addr, peers, replication_factor, write_quorum, api_key).await?;
        }

        Commands::Ask {
            question,
            max_results,
            storage_path,
            index_path,
        } => {
            let question = question.join(" ");
            let engine = KowitoDBEngine::open(&storage_path, &index_path)
                .await
                .map_err(|e| anyhow::anyhow!("Failed to open database: {}", e))?;

            println!("🤖 Asking: \"{}\"\n", question);

            let response = engine
                .ask(&question, max_results.clamp(1, 20))
                .await
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;

            println!("Detected intent: {}", response.detected_intent);
            println!(
                "Context tokens: {} (compression: {:.0}%)",
                response.total_tokens,
                response.compression_ratio * 100.0
            );
            println!();

            if response.results.is_empty() {
                println!("  (no results found)");
            } else {
                println!("Results ({} found):", response.results.len());
                for (i, r) in response.results.iter().enumerate() {
                    println!();
                    println!(
                        "  #{}. [score: {:.2}] [source: {}]",
                        i + 1,
                        r.relevance_score,
                        r.retrieval_source,
                    );
                    println!("  ID: {}", r.id);
                    let preview: String = r.content.chars().take(200).collect();
                    println!("  {}", preview);
                    if r.content.len() > 200 {
                        println!("  ... ({} more chars)", r.content.len() - 200);
                    }
                }
            }

            println!();
            println!("Query plan:");
            println!("{}", response.plan_explanation);
        }

        Commands::Insert {
            content,
            file,
            keywords,
            metadata,
            importance,
            storage_path,
            index_path,
        } => {
            let engine = KowitoDBEngine::new(&storage_path, &index_path)
                .map_err(|e| anyhow::anyhow!("Failed to open database: {}", e))?;

            let (text, file_kws, file_meta) = if let Some(path) = file {
                let raw = std::fs::read_to_string(&path)?;
                if path.extension().is_some_and(|e| e == "json") {
                    let v: serde_json::Value = serde_json::from_str(&raw)?;
                    let content = v["content"].as_str().unwrap_or(&raw).to_string();
                    let kws: Vec<String> = v["keywords"]
                        .as_array()
                        .map(|a| {
                            a.iter()
                                .filter_map(|v| v.as_str().map(String::from))
                                .collect()
                        })
                        .unwrap_or_default();
                    let meta: Vec<(String, String)> = v["metadata"]
                        .as_object()
                        .map(|o| {
                            o.iter()
                                .map(|(k, v)| (k.clone(), v.as_str().unwrap_or("").to_string()))
                                .collect()
                        })
                        .unwrap_or_default();
                    (content, kws, meta)
                } else {
                    (raw, vec![], vec![])
                }
            } else {
                (content.join(" "), vec![], vec![])
            };

            // Merge command-line keywords with file keywords
            let all_keywords: Vec<String> = {
                let mut kws = file_kws;
                if let Some(ref kw_str) = keywords {
                    kws.extend(kw_str.split(',').map(|s| s.trim().to_string()));
                }
                kws
            };

            // Merge command-line metadata with file metadata
            let all_metadata: Vec<(String, String)> = {
                let mut meta = file_meta;
                if let Some(ref meta_str) = metadata {
                    for pair in meta_str.split(',') {
                        let parts: Vec<&str> = pair.splitn(2, '=').collect();
                        if parts.len() == 2 {
                            meta.push((parts[0].trim().to_string(), parts[1].trim().to_string()));
                        }
                    }
                }
                meta
            };

            let keywords_len = all_keywords.len();
            let metadata_len = all_metadata.len();

            let mut obj = KnowledgeObject::new(text)
                .with_keywords(all_keywords)
                .with_importance(importance);

            for (k, v) in &all_metadata {
                obj = obj.with_metadata(k.clone(), v.clone());
            }

            let id = engine
                .insert(obj)
                .await
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;

            println!("✅ Inserted knowledge object: {}", id);
            println!("   Keywords: {}", keywords_len);
            println!("   Metadata keys: {}", metadata_len);
        }

        Commands::Sql {
            query,
            storage_path,
            index_path,
        } => {
            let sql = query.join(" ");
            let engine = KowitoDBEngine::open(&storage_path, &index_path)
                .await
                .map_err(|e| anyhow::anyhow!("Failed to open database: {}", e))?;

            println!("📊 SQL: {}\n", sql);

            let results = engine
                .sql_query(&sql)
                .await
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;

            if results.is_empty() {
                println!("  (no results)");
            } else {
                println!("  {} row(s):\n", results.len());
                for (i, r) in results.iter().enumerate() {
                    println!("  {}. {}", i + 1, r.id);
                    let preview: String = r.content.chars().take(150).collect();
                    println!("     {}", preview);
                    if r.content.len() > 150 {
                        println!("     ... ({} more chars)", r.content.len() - 150);
                    }
                    println!();
                }
            }
        }

        Commands::Stats {
            storage_path,
            index_path,
        } => {
            let engine = KowitoDBEngine::open(&storage_path, &index_path)
                .await
                .map_err(|e| anyhow::anyhow!("Failed to open database: {}", e))?;

            let stats = engine
                .stats()
                .await
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;

            println!("📊 KowitoDB Statistics");
            println!("======================");
            println!("  Total objects:      {}", stats.total_objects);
            println!("  Vectors indexed:    {}", stats.vector_count);
            println!("  Graph nodes:        {}", stats.graph_nodes);
            println!("  Graph edges:        {}", stats.graph_edges);
            println!("  Active sessions:    {}", stats.active_agent_sessions);

            if let Some(ref cache) = stats.cache_stats {
                println!("  Cache entries:      {}", cache.entries);
                println!(
                    "  Cache hit rate:     {:.1}% (hits={}, misses={})",
                    cache.hit_rate * 100.0,
                    cache.hits,
                    cache.misses,
                );
            }

            println!("  Total cost (est.):  ${:.6}", stats.total_cost_usd);
        }

        Commands::Demo => run_demo().await?,

        Commands::Completions { shell } => {
            let mut cmd = Cli::command();
            let name = cmd.get_name().to_string();
            clap_complete::generate(shell, &mut cmd, name, &mut std::io::stdout());
        }
    }
    Ok(())
}

/// Seed an in-memory engine with a few facts and run example `ask()`/SQL
/// queries — a zero-setup tour of what KowitoDB does.
async fn run_demo() -> anyhow::Result<()> {
    println!("🚀 KowitoDB demo — in-memory, no server, no setup.\n");
    let engine = KowitoDBEngine::new_in_memory().map_err(|e| anyhow::anyhow!(e.to_string()))?;

    let facts: &[(&str, &[&str], &str, f32)] = &[
        (
            "Acme Corp renewed their enterprise license in March 2024 after a $15M Series A.",
            &["acme", "renewal", "series a"],
            "Acme Corp",
            0.9,
        ),
        (
            "Globex shipped their v2 platform and onboarded three enterprise customers in Q2.",
            &["globex", "launch"],
            "Globex",
            0.7,
        ),
        (
            "Initech churned in February after budget cuts.",
            &["initech", "churn"],
            "Initech",
            0.6,
        ),
        (
            "Umbrella signed a multi-year enterprise contract worth $2M.",
            &["umbrella", "contract"],
            "Umbrella",
            0.85,
        ),
    ];
    for (text, kws, company, importance) in facts {
        let obj = KnowledgeObject::new(*text)
            .with_keywords(kws.iter().map(|s| s.to_string()).collect())
            .with_metadata("company", *company)
            .with_importance(*importance);
        engine
            .insert(obj)
            .await
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    }
    println!("Seeded {} knowledge objects.\n", facts.len());

    for q in [
        "Which enterprise customers had activity?",
        "What happened with churn?",
    ] {
        println!("❯ ai.ask(\"{q}\")");
        let resp = engine
            .ask(q, 3)
            .await
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        println!("  intent: {}", resp.detected_intent);
        for r in &resp.results {
            println!(
                "  [{:.2}] ({}) {}",
                r.relevance_score, r.retrieval_source, r.content
            );
        }
        println!();
    }

    println!("❯ sql: SELECT content FROM knowledge WHERE importance >= 0.8");
    let rows = engine
        .sql_select("SELECT content FROM knowledge WHERE importance >= 0.8")
        .await
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    for row in &rows {
        if let Some(c) = row.get("content") {
            println!("  • {c}");
        }
    }

    println!("\n✅ Done. Next: `kowitodb serve` to run the server, or see the README.");
    Ok(())
}
