//! CLI surface (clap definitions) for the kowitodb binary.

use std::net::SocketAddr;
use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};
use clap_complete::Shell;

/// KowitoDB — AI Knowledge Operating System
///
/// An open-source database that combines vector search, full-text search,
/// knowledge graph traversal, and AI query planning behind a single `ai.ask()`
/// interface.
#[derive(Parser)]
#[command(name = "kowitodb")]
#[command(version)]
#[command(about = "AI Knowledge Operating System", long_about = None)]
#[command(after_help = "EXAMPLES:
  kowitodb demo                       See it work in 2s (in-memory, no setup)
  kowitodb serve                      Start the gRPC server on 127.0.0.1:50051
  kowitodb insert \"Acme renewed\" -k acme,renewal   Add a fact
  kowitodb ask \"who renewed?\"          Query across all indexes (embedded)
  kowitodb sql \"SELECT content FROM knowledge LIMIT 5\"
  kowitodb gateway --peers host1:50051,host2:50051  Distributed coordinator

Configuration is via KOWITODB_* env vars — see the README.")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

/// Storage backend for the server.
#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum StorageKind {
    /// Default embedded sled key-value store.
    Sled,
    /// Lance columnar dataset (requires building with `--features lance`).
    Lance,
}

#[derive(Subcommand)]
pub enum Commands {
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

    /// Seed an in-memory database with sample data and run example queries —
    /// the fastest way to see KowitoDB work (no server, no setup, no disk).
    Demo,

    /// Print a shell completion script (bash, zsh, fish, powershell, elvish).
    ///
    /// e.g.  kowitodb completions zsh > ~/.zsh/completions/_kowitodb
    Completions {
        /// Target shell
        shell: Shell,
    },
}
