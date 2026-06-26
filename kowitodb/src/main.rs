use clap::{Parser, Subcommand};
use std::net::SocketAddr;
use tracing::info;
use tracing_subscriber::EnvFilter;

use kowitodb_server::{serve, KowitoDBEngine};

/// KowitoDB — AI Knowledge Operating System
///
/// An open-source database that combines vector search, full-text search,
/// knowledge graph traversal, and AI query planning behind a single `ai.ask()`
/// interface.
#[derive(Parser)]
#[command(name = "kowitodb")]
#[command(version = "0.1.0")]
#[command(about = "AI Knowledge Operating System", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the KowitoDB server (gRPC)
    Serve {
        /// Address to bind the gRPC server
        #[arg(short, long, default_value = "127.0.0.1:50051")]
        addr: SocketAddr,

        /// Path for persistent storage
        #[arg(short, long, default_value = "./data/storage")]
        storage_path: String,

        /// Path for the full-text index
        #[arg(short, long, default_value = "./data/index")]
        index_path: String,
    },

    /// Ask a question against a running server (demo/CLI mode)
    Ask {
        /// The question to ask
        question: Vec<String>,

        /// Server address
        #[arg(short, long, default_value = "http://127.0.0.1:50051")]
        server: String,
    },

    /// Insert a knowledge object from a JSON file
    Insert {
        /// Path to JSON file describing the knowledge object
        path: String,

        /// Server address
        #[arg(short, long, default_value = "http://127.0.0.1:50051")]
        server: String,
    },

    /// Show database stats from a running server
    Stats {
        /// Server address
        #[arg(short, long, default_value = "http://127.0.0.1:50051")]
        server: String,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Serve {
            addr,
            storage_path,
            index_path,
        } => {
            info!("Starting KowitoDB v{}", env!("CARGO_PKG_VERSION"));
            info!("Storage path: {}", storage_path);
            info!("Index path: {}", index_path);

            // Create directories
            std::fs::create_dir_all(&storage_path)?;
            std::fs::create_dir_all(&index_path)?;

            let engine = KowitoDBEngine::new(&storage_path, &index_path)
                .map_err(|e| anyhow::anyhow!("Failed to initialize engine: {}", e))?;

            serve(engine, addr).await?;
        }

        Commands::Ask { question, server } => {
            let question = question.join(" ");
            println!("Asking: \"{}\"", question);
            println!("Server: {}", server);
            println!();
            println!("(Connect to a running KowitoDB server to get results.)");
            println!("For local demo, run: cargo run -- serve");
        }

        Commands::Insert { path, server } => {
            println!("Insert from: {}", path);
            println!("Server: {}", server);
            println!();
            println!("(Connect to a running KowitoDB server to insert.)");
            println!("For local demo, run: cargo run -- serve");
        }

        Commands::Stats { server } => {
            println!("Fetching stats from: {}", server);
            println!();
            println!("(Connect to a running KowitoDB server for stats.)");
            println!("For local demo, run: cargo run -- serve");
        }
    }

    Ok(())
}
