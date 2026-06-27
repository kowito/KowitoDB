//! Server runtime configuration.
//!
//! Values come from CLI flags (which fall back to environment variables — see
//! the `kowitodb serve` command) and are assembled into a [`ServerConfig`] that
//! [`crate::serve`] uses to enable auth, TLS, and the metrics endpoint.

use std::net::SocketAddr;
use std::path::PathBuf;

/// Optional production hardening for the gRPC server. All fields default to
/// "off", preserving the plaintext, unauthenticated dev behavior.
#[derive(Debug, Clone, Default)]
pub struct ServerConfig {
    /// When set, every gRPC call must present this key via an
    /// `authorization: Bearer <key>` or `x-api-key: <key>` metadata header.
    pub api_key: Option<String>,
    /// PEM-encoded TLS certificate chain path. Requires `tls_key` too.
    pub tls_cert: Option<PathBuf>,
    /// PEM-encoded TLS private key path. Requires `tls_cert` too.
    pub tls_key: Option<PathBuf>,
    /// When set, an HTTP server exposes `/metrics` (Prometheus) and `/healthz`.
    pub metrics_addr: Option<SocketAddr>,
}

impl ServerConfig {
    /// Whether TLS is fully configured (both cert and key present).
    pub fn tls_enabled(&self) -> bool {
        self.tls_cert.is_some() && self.tls_key.is_some()
    }
}
