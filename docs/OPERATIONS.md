# KowitoDB Operations

Running KowitoDB in production: data safety, upgrades, the metrics and cost
surface, plan-cache behavior, and the limits you must design around. This
complements [DEPLOYMENT.md](DEPLOYMENT.md).

## Contents

- [Data directory layout](#data-directory-layout)
- [Backup and restore](#backup-and-restore)
- [Index persistence and restarts](#index-persistence-and-restarts)
- [Upgrades](#upgrades)
- [Metrics and the Stats RPC](#metrics-and-the-stats-rpc)
- [The cost tracker](#the-cost-tracker)
- [Plan cache behavior](#plan-cache-behavior)
- [Known limitations and scaling boundaries](#known-limitations-and-scaling-boundaries)
- [Operational runbook](#operational-runbook)

## Data directory layout

A running server owns two persistent directories (the paths passed to `serve`):

```
{storage-path}/          sled object store          (persistent)
{index-path}/tantivy/    Tantivy full-text index    (persistent)
```

Everything else (HNSW vector, brute-force vector, metadata, time, and graph
indexes; plan cache; agent memory) is **in-memory only**. With the Lance backend
the object store is a Lance dataset at the configured URI instead of the sled
directory.

## Backup and restore

### What to back up

Back up the object store and the full-text index **as a consistent pair**. The
in-memory indexes need not (and cannot) be backed up — they are derived from the
object store.

### Recommended procedure (sled, default)

sled holds an exclusive lock on its directory while the process runs, so the
safest approach is a brief stop-the-world snapshot:

```bash
# 1. Stop the server (or take it out of rotation).
systemctl stop kowitodb        # or: docker stop <container>

# 2. Snapshot both directories atomically together.
tar czf kowitodb-backup-$(date +%F).tgz \
  -C /var/lib/kowitodb storage index

# 3. Restart.
systemctl start kowitodb
```

If you cannot stop the process, use a filesystem-level snapshot (LVM, ZFS, EBS
snapshot) of the volume holding both directories, taken at the same instant, so
the object store and full-text index stay consistent with each other.

### Restore

```bash
systemctl stop kowitodb
rm -rf /var/lib/kowitodb/storage /var/lib/kowitodb/index
tar xzf kowitodb-backup-YYYY-MM-DD.tgz -C /var/lib/kowitodb
systemctl start kowitodb
```

After restore, the object store and full-text index are intact, but the
in-memory indexes start empty — see the next section.

### Lance backend

Back up the Lance dataset directory/URI together with the same `{index-path}`
Tantivy directory. Lance is versioned, but for operational backups treat the
dataset directory as the unit to snapshot, paired with the full-text index.

## Index persistence and restarts

This is the single most important operational fact about KowitoDB v0.1.0.

**Only the object store and the Tantivy full-text index survive a restart.** The
HNSW vector index, metadata index, time index, and graph index are in-memory and
are populated **only** by `insert` calls at runtime. The engine constructors
open storage and the full-text index and create *empty* in-memory indexes; there
is no startup pass that re-reads the object store to rebuild them.

Consequences after any restart or restore:

- `Get` by ID, full-text keyword search, and `sql_select` (DataFusion, which
  scans the object store directly) all keep working.
- Vector search, metadata filters, time filters, and graph traversal return
  nothing for previously-stored objects — so `ask` quality degrades to
  full-text-only — until those objects are re-inserted.

### Mitigations

Until index persistence/rebuild-on-start is implemented, treat the object store
as the source of truth and re-ingest after restart:

- **Re-ingestion pass on startup.** Drive a one-time re-`insert` of every stored
  object after the server comes up. You can enumerate IDs via the storage
  layer's `list_ids` / `search` (e.g. through a small admin tool or a
  `sql_select('SELECT id, content, ... FROM knowledge')` followed by
  `Insert`/`Remember` calls). This rebuilds all in-memory indexes.
- **Keep the source corpus.** Maintain the authoritative documents outside
  KowitoDB so you can always re-ingest deterministically.
- **Plan restarts as full re-index windows**, and avoid relying on vector/graph
  results during the warm-up period.

Schedule restarts during low-traffic windows and validate `Stats.vector_count`
has returned to the expected level before resuming vector-dependent traffic.

## Upgrades

- **Binary upgrades.** Replace the binary/image and restart. Re-indexing of the
  in-memory indexes applies as described above; plan for it.
- **On-disk format compatibility.** The object store is JSON-per-object in sled,
  and the full-text index is Tantivy. There is no schema-migration tooling in
  v0.1.0. Across versions:
  - The `StoredObject` JSON shape and the Tantivy schema must remain compatible,
    or the existing data directory may not load.
  - Tantivy index format compatibility is tied to the Tantivy version; a major
    Tantivy bump can require rebuilding the full-text index (delete
    `{index-path}/tantivy/` and re-ingest).
- **Recommended upgrade flow:** back up → deploy new version to a staging copy of
  the data directory → verify it opens and `ask`/`sql_select` behave → cut over.
- **Proto/SDK changes.** If you change `proto/kowitodb.proto`, regenerate SDK
  stubs (see [SDKS.md](SDKS.md)) and roll clients and server together; there is
  no negotiated API versioning.

## Metrics and the Stats RPC

The `Stats` RPC (`StatsResponse`) is the primary in-band telemetry surface:

| Field | Source | Meaning |
| --- | --- | --- |
| `total_objects` | storage `count()` | Number of stored objects. |
| `vector_count` | HNSW index `len()` | Vectors currently in the in-memory HNSW index. After a restart this is 0 until re-ingestion — a useful warm-up indicator. |
| `index_size_bytes` | mixed | Currently `0` plus the server's cumulative `ask` count folded in; **do not** read this as a real byte size. |

The engine computes a richer internal `StatsResponse` (graph node/edge counts,
plan-cache stats, total estimated cost, active agent sessions) that the
`kowitodb stats` CLI command prints, but the **gRPC** `Stats` response only
carries the three fields above. For the full picture (cache hit rate, cost,
graph size, sessions) use the CLI `stats` command against the data directory or
read the server logs.

`MetricsCollector` also tracks ask/remember/insert/sql/error counts, cumulative
and average ask latency, and uptime in-process. These are not exported on a
metrics endpoint; surface them via logs or by extending the service if you need
Prometheus.

## The cost tracker

`CostTracker` maintains a running USD estimate from a `CostModel`:

- Per-1k rates approximate small commercial models (embedding ≈ OpenAI
  `text-embedding-3-small`; LLM input/output ≈ GPT-4o-mini-class). Local index
  lookups are priced at 0.
- It accumulates: embedding calls (one per query embed, plus one per auto-embed
  on insert), index lookups, and LLM input tokens (the assembled context size
  recorded per `ask`).
- The total is exposed as `total_cost_usd` and printed by `kowitodb stats`
  (`Total cost (est.)`).

Caveats:

- This is an **estimate** for capacity planning. With the default proxy
  embedder there is no real provider cost; the figure reflects the model rates,
  not your bill.
- LLM *output* tokens are part of the model but KowitoDB does not call an LLM
  itself, so output-token cost is only recorded if you record it.

Use `total_cost_usd` to reason about relative query expense (e.g. context-token
growth driving cost), not as an authoritative invoice.

## Plan cache behavior

The engine caches `(DetectedIntent, ExecutionPlan)` keyed by the **raw question
string**, via a `QueryCache` configured with a TTL (300s) and a max entry count
(1000):

- A repeated identical question reuses the cached plan, skipping intent analysis
  and rule evaluation (it does **not** cache results — retrieval still runs).
- **Every `insert` and every successful `delete` clears the entire plan cache**,
  so plans are never reused against a changed corpus. Under a heavy write load
  the cache is effectively cold, which is correct but means little hit-rate
  benefit during ingestion-heavy periods.
- Entries also expire by TTL and are evicted when capacity is exceeded.
- Cache hit/miss/entry stats are available via `CacheStats` and printed by
  `kowitodb stats` (`Cache entries`, `Cache hit rate`).

Operational implication: plan-cache hit rate is highest for read-mostly
workloads with recurring questions, and near-zero during bulk ingestion. This is
expected.

## Known limitations and scaling boundaries

State these honestly when planning a deployment:

- **Single node.** No replication, sharding, failover, or clustering. The
  ceiling is one machine. There is no horizontal scale-out path in v0.1.0.
- **In-memory indexes, not rebuilt on restart.** See
  [Index persistence and restarts](#index-persistence-and-restarts). The working
  set must fit in RAM, and restarts require re-ingestion to restore
  vector/metadata/time/graph search.
- **No authentication, no TLS.** The gRPC endpoint is unauthenticated plaintext.
  Secure it at the network/proxy layer (see [DEPLOYMENT.md](DEPLOYMENT.md)).
- **No health/reflection services.** Use TCP or `Stats`-based checks.
- **Default embeddings are a deterministic proxy**, not a semantic model.
  Wire the OpenAI-compatible client (or another real embedder) before relying on
  vector relevance in production. Token counts are heuristic (~4 chars/token).
- **`Search`/`Ask` results cap at 100**; `max_context_tokens` in the proto is
  ignored (the optimizer uses its 4096-token default).
- **No dedicated SQL RPC.** The SDKs' `sql()` helpers route through the `Search`
  RPC (the `ask` pipeline), **not** the DataFusion engine. Full SQL is available
  only through the Rust engine API (`sql_select`) and the `kowitodb sql` CLI
  command (index-routed path).
- **Storage `search` is a full scan.** The sled and Lance backends both scan all
  rows and filter in Rust; the DataFusion provider materializes the whole table
  per query. This is fine at modest scale but is O(n) per query and bounds how
  large a corpus stays responsive.
- **Embedded CLI sees only its own process's indexes.** Because the non-full-text
  indexes are in-memory, `kowitodb ask`/`sql` started fresh will not have the
  vector/graph/metadata/time indexes populated for data inserted by a different
  process. For shared querying, run `serve` and use clients.

## Operational runbook

| Situation | Action |
| --- | --- |
| After a restart, `ask` returns weak/empty results | Expected if you have not re-ingested. Check `Stats.vector_count`; run the re-ingestion pass; resume vector traffic once counts recover. |
| Plan-cache hit rate near zero during a bulk load | Expected — writes clear the cache. It recovers once writes settle. |
| Memory pressure / OOM risk | The full working set lives in RAM. Shed objects, raise host memory, or move to a larger node; there is no spill-to-disk for the in-memory indexes. |
| Need to expose the port externally | Don't, without a TLS-terminating, authenticating proxy in front and network ACLs. |
| Need a real liveness probe | TCP connect to `--addr`, or a periodic `Stats` RPC from a sidecar. |
| Corrupt/incompatible data dir after upgrade | Restore from backup, or delete `{index-path}/tantivy/` and re-ingest to rebuild the full-text index. |
| Need full SQL (aggregates, ORDER BY) | Use `kowitodb sql` (CLI) or the engine `sql_select` API — not the SDK `sql()` helper. |
