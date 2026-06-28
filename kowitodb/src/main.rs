use std::net::SocketAddr;
use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};
use tracing::info;
use tracing_subscriber::EnvFilter;

use kowitodb_core::KnowledgeObject;
use kowitodb_server::{serve_gateway, serve_with_config, KowitoDBEngine, ServerConfig};

/// KowitoDB — AI Knowledge Operating System
///
/// An open-source database that combines vector search, full-text search,
/// knowledge graph traversal, and AI query planning behind a single `ai.ask()`
/// interface.
#[derive(Parser)]
#[command(name = "kowitodb")]
#[command(version)]
#[command(about = "AI Knowledge Operating System", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

/// Storage backend for the server.
#[derive(Copy, Clone, Debug, ValueEnum)]
enum StorageKind {
    /// Default embedded sled key-value store.
    Sled,
    /// Lance columnar dataset (requires building with `--features lance`).
    Lance,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the KowitoDB gRPC server
    Serve {
        /// Address to bind
        #[arg(short, long, default_value = "127.0.0.1:50051")]
        addr: SocketAddr,

        /// Persistence path
        #[arg(short, long, default_value = "./data/storage")]
        storage_path: PathBuf,

        /// Index path
        #[arg(short, long, default_value = "./data/index")]
        index_path: PathBuf,

        /// Storage backend (lance requires a build with --features lance).
        #[arg(long, value_enum, default_value = "sled", env = "KOWITODB_STORAGE")]
        storage: StorageKind,

        /// Lance dataset URI/path (used when --storage lance; defaults to
        /// {storage-path}/lance).
        #[arg(long, env = "KOWITODB_LANCE_URI")]
        lance_uri: Option<String>,

        /// Cap on results returned by Ask/Search.
        #[arg(long, default_value = "100", env = "KOWITODB_MAX_RESULTS")]
        max_results: usize,

        /// Require this API key on every request (Bearer or x-api-key header).
        #[arg(long, env = "KOWITODB_API_KEY")]
        api_key: Option<String>,

        /// PEM TLS certificate chain (enables TLS together with --tls-key).
        #[arg(long, env = "KOWITODB_TLS_CERT")]
        tls_cert: Option<PathBuf>,

        /// PEM TLS private key.
        #[arg(long, env = "KOWITODB_TLS_KEY")]
        tls_key: Option<PathBuf>,

        /// Expose Prometheus /metrics + /healthz on this address (e.g. 0.0.0.0:9090).
        #[arg(long, env = "KOWITODB_METRICS_ADDR")]
        metrics_addr: Option<SocketAddr>,
    },

    /// Run a cluster gateway that distributes over data nodes (distributed mode)
    Gateway {
        /// Address to bind the gateway
        #[arg(short, long, default_value = "127.0.0.1:50050")]
        addr: SocketAddr,

        /// Comma-separated data node addresses (e.g. host1:50051,host2:50051)
        #[arg(long, value_delimiter = ',', env = "KOWITODB_PEERS")]
        peers: Vec<String>,

        /// Replication factor — write each object to this many nodes
        #[arg(long, default_value = "1", env = "KOWITODB_REPLICATION_FACTOR")]
        replication_factor: usize,

        /// Write quorum — replica acks required per write (clamped to RF;
        /// `>= ceil((RF+1)/2)` gives majority durability)
        #[arg(long, default_value = "1", env = "KOWITODB_WRITE_QUORUM")]
        write_quorum: usize,

        /// Require this API key on every request, and present it to data nodes.
        #[arg(long, env = "KOWITODB_API_KEY")]
        api_key: Option<String>,
    },

    /// Ask a question (embedded mode — no server required)
    Ask {
        /// The question
        question: Vec<String>,

        /// Max results to return
        #[arg(short, long, default_value = "5")]
        max_results: usize,

        /// Storage path (must match the data directory)
        #[arg(short, long, default_value = "./data/storage")]
        storage_path: PathBuf,

        /// Index path (must match the data directory)
        #[arg(short, long, default_value = "./data/index")]
        index_path: PathBuf,
    },

    /// Insert a knowledge object from a JSON file or inline text
    Insert {
        /// Content text (or path to JSON file with --file)
        content: Vec<String>,

        /// Read from a JSON file instead of inline text
        #[arg(short, long)]
        file: Option<PathBuf>,

        /// Comma-separated keywords
        #[arg(short, long)]
        keywords: Option<String>,

        /// Comma-separated key=value metadata pairs
        #[arg(short, long)]
        metadata: Option<String>,

        /// Importance score (0.0 - 1.0)
        #[arg(long, default_value = "0.5")]
        importance: f32,

        /// Storage path
        #[arg(short, long, default_value = "./data/storage")]
        storage_path: PathBuf,

        /// Index path
        #[arg(short, long, default_value = "./data/index")]
        index_path: PathBuf,
    },

    /// Execute a SQL query over knowledge objects
    Sql {
        /// The SQL query
        query: Vec<String>,

        /// Storage path
        #[arg(short, long, default_value = "./data/storage")]
        storage_path: PathBuf,

        /// Index path
        #[arg(short, long, default_value = "./data/index")]
        index_path: PathBuf,
    },

    /// Show database statistics
    Stats {
        /// Storage path
        #[arg(short, long, default_value = "./data/storage")]
        storage_path: PathBuf,

        /// Index path
        #[arg(short, long, default_value = "./data/index")]
        index_path: PathBuf,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
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
    }

    Ok(())
}
