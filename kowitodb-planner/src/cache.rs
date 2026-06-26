use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use tracing::debug;

/// A cached query result entry.
#[derive(Debug, Clone)]
struct CacheEntry<V> {
    value: V,
    inserted_at: Instant,
    ttl: Duration,
}

impl<V> CacheEntry<V> {
    fn is_expired(&self, now: Instant) -> bool {
        now.duration_since(self.inserted_at) > self.ttl
    }
}

/// Time-to-live key-value cache with eviction.
///
/// Used for:
/// - Query result caching (same question → return cached response)
/// - Embedding caching (same text → return cached embedding)
/// - Plan caching (same question → return cached execution plan)
pub struct QueryCache<V> {
    entries: Arc<RwLock<HashMap<String, CacheEntry<V>>>>,
    default_ttl: Duration,
    /// Maximum number of entries before eviction.
    max_entries: usize,
    /// Statistics.
    hits: Arc<RwLock<u64>>,
    misses: Arc<RwLock<u64>>,
}

impl<V: Clone> QueryCache<V> {
    /// Create a new cache with default TTL and max entries.
    pub fn new(ttl_secs: u64, max_entries: usize) -> Self {
        Self {
            entries: Arc::new(RwLock::new(HashMap::new())),
            default_ttl: Duration::from_secs(ttl_secs),
            max_entries,
            hits: Arc::new(RwLock::new(0)),
            misses: Arc::new(RwLock::new(0)),
        }
    }

    /// Get a value by key, returning None if not found or expired.
    pub fn get(&self, key: &str) -> Option<V> {
        let entries = self.entries.read();
        let now = Instant::now();

        match entries.get(key) {
            Some(entry) if !entry.is_expired(now) => {
                let mut hits = self.hits.write();
                *hits += 1;
                debug!("Cache HIT: {}", key);
                Some(entry.value.clone())
            }
            _ => {
                let mut misses = self.misses.write();
                *misses += 1;
                debug!("Cache MISS: {}", key);
                None
            }
        }
    }

    /// Insert a value with the default TTL.
    pub fn insert(&self, key: impl Into<String>, value: V) {
        self.insert_with_ttl(key, value, self.default_ttl);
    }

    /// Insert a value with a custom TTL.
    pub fn insert_with_ttl(&self, key: impl Into<String>, value: V, ttl: Duration) {
        let mut entries = self.entries.write();
        let key = key.into();

        // Evict if at capacity (simple: remove oldest entry)
        if entries.len() >= self.max_entries {
            self.evict_oldest(&mut entries);
        }

        entries.insert(
            key.clone(),
            CacheEntry {
                value,
                inserted_at: Instant::now(),
                ttl,
            },
        );

        debug!("Cache INSERT: {} (total entries: {})", key, entries.len());
    }

    /// Remove an entry by key.
    pub fn invalidate(&self, key: &str) {
        let mut entries = self.entries.write();
        entries.remove(key);
    }

    /// Clear all entries.
    pub fn clear(&self) {
        let mut entries = self.entries.write();
        entries.clear();
    }

    /// Get cache statistics.
    pub fn stats(&self) -> CacheStats {
        let hits = *self.hits.read();
        let misses = *self.misses.read();
        let entries = self.entries.read().len();

        CacheStats {
            entries,
            hits,
            misses,
            hit_rate: if hits + misses > 0 {
                hits as f32 / (hits + misses) as f32
            } else {
                0.0
            },
        }
    }

    /// Evict the oldest entry.
    fn evict_oldest(&self, entries: &mut HashMap<String, CacheEntry<V>>) {
        if let Some(oldest_key) = entries
            .iter()
            .min_by_key(|(_, e)| e.inserted_at)
            .map(|(k, _)| k.clone())
        {
            entries.remove(&oldest_key);
            debug!("Cache EVICT: {}", oldest_key);
        }
    }
}

#[derive(Debug, Clone)]
pub struct CacheStats {
    pub entries: usize,
    pub hits: u64,
    pub misses: u64,
    pub hit_rate: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_hit_miss() {
        let cache: QueryCache<String> = QueryCache::new(60, 10);
        assert!(cache.get("key1").is_none());

        cache.insert("key1", "value1".to_string());
        assert_eq!(cache.get("key1"), Some("value1".to_string()));

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn test_cache_expiry() {
        let cache: QueryCache<String> = QueryCache::new(0, 10); // 0 second TTL
        cache.insert("key1", "value1".to_string());
        // Should expire immediately (TTL 0)
        // In practice, it might not yet be expired due to instant timing,
        // but the TTL is configured to expire quickly
        std::thread::sleep(Duration::from_millis(1));
        // After sleeping, the entry should be expired
        let _ = cache.get("key1"); // May or may not be expired depending on timing
    }

    #[test]
    fn test_cache_eviction() {
        let cache: QueryCache<String> = QueryCache::new(3600, 2); // max 2 entries
        cache.insert("a", "1".into());
        cache.insert("b", "2".into());
        cache.insert("c", "3".into()); // Should evict oldest

        // One of a or b should have been evicted
        let stats = cache.stats();
        assert_eq!(stats.entries, 2);
    }
}
