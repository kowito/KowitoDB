//! kowitodb — CLI entrypoint.

use clap::Parser;
use tracing_subscriber::EnvFilter;

mod cli;
mod commands;

use cli::{Cli, Commands};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // `demo` (a human-facing tour) and `completions` (prints a script) default
    // to quiet — warnings only — unless the user set RUST_LOG. Everything else
    // defaults to info-level logs.
    let default_filter = if matches!(cli.command, Commands::Demo | Commands::Completions { .. }) {
        "warn"
    } else {
        "info"
    };
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_filter)),
        )
        .init();

    commands::run(cli.command).await
}
