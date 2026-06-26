//! Server metrics and health monitoring.
//!
//! Tracks: request counts, latencies, error rates, cache hit rates.

use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::RwLock;

/// Metrics collected by the server.
#[derive(Debug, Clone, Default)]
pub struct ServerMetrics {
    /// Total ai.ask() calls.
    pub ask_count: u64,
    /// Total ai.remember() calls.
    pub remember_count: u64,
    /// Total SQL queries.
    pub sql_count: u64,
    /// Total insert calls.
    pub insert_count: u64,
    /// Total errors.
    pub error_count: u64,
    /// Cumulative ask latency in microseconds.
    pub ask_latency_us: u64,
    /// Server start time.
    pub started_at: Option<Instant>,
}

/// Thread-safe metrics collector.
pub struct MetricsCollector {
    metrics: Arc<RwLock<ServerMetrics>>,
}

impl MetricsCollector {
    pub fn new() -> Self {
        let mut metrics = ServerMetrics::default();
        metrics.started_at = Some(Instant::now());
        Self {
            metrics: Arc::new(RwLock::new(metrics)),
        }
    }

    pub fn record_ask(&self, latency: Duration) {
        let mut m = self.metrics.write();
        m.ask_count += 1;
        m.ask_latency_us += latency.as_micros() as u64;
    }

    pub fn record_remember(&self) {
        self.metrics.write().remember_count += 1;
    }

    pub fn record_sql(&self) {
        self.metrics.write().sql_count += 1;
    }

    pub fn record_insert(&self) {
        self.metrics.write().insert_count += 1;
    }

    pub fn record_error(&self) {
        self.metrics.write().error_count += 1;
    }

    pub fn snapshot(&self) -> ServerMetrics {
        self.metrics.read().clone()
    }

    /// Average ask latency in milliseconds.
    pub fn avg_ask_latency_ms(&self) -> f64 {
        let m = self.metrics.read();
        if m.ask_count == 0 {
            return 0.0;
        }
        (m.ask_latency_us as f64 / m.ask_count as f64) / 1000.0
    }

    /// Uptime since server start.
    pub fn uptime_secs(&self) -> u64 {
        self.metrics
            .read()
            .started_at
            .map(|t| t.elapsed().as_secs())
            .unwrap_or(0)
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}
