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
- [Updates and version history](#updates-and-version-history)
- [Plan cache behavior](#plan-cache-behavior)
- [Known limitations and scaling boundaries](#known-limitations-and-scaling-boundaries)
- [Operational runbook](#operational-runbook)

## Data directory layout

A running server owns two persistent directories (the paths passed to `serve`):

```
{storage-path}/          sled object store          (persistent)
{index-path}/tantivy/    Tantivy full-text index    (persistent)
```

The in-memory indexes work as follows. The **HNSW vector index is persisted**
as a snapshot at `{index_path}/hnsw.bin` (checkpointed periodically and on
graceful shutdown) and loaded on startup, so it is not rebuilt from scratch; if
the snapshot is missing (e.g. after a hard kill) it falls back to a rebuild from
stored embeddings. The metadata, time, and graph indexes are still rebuilt from
the object store on startup (see
[Index persistence and restarts](#index-persistence-and-restarts)). **Agent
memory is persisted** to a sled store at `{index_path}/sessions`. The plan cache
is ephemeral. With the Lance backend the object store is a Lance dataset at the
configured URI instead of the sled directory.

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

After restore, the object store and full-text index are intact, and the
in-memory indexes are rebuilt from the object store the next time the engine is
opened — see the next section.

### Lance backend

Back up the Lance dataset directory/URI together with the same `{index-path}`
Tantivy directory. Lance is versioned, but for operational backups treat the
dataset directory as the unit to snapshot, paired with the full-text index.

## Index persistence and restarts

**The in-memory indexes are rebuilt from the object store on startup.** Only the
object store (sled or Lance) and the Tantivy full-text index are persisted to
disk, but `KowitoDBEngine::open()` — used by `serve` and by the
`ask`/`sql`/`stats` CLI commands — runs a `reindex_from_storage()` pass before
serving. That pass re-reads every persisted object and repopulates the HNSW
vector, metadata, time, and graph indexes from the stored fields (including the
**persisted embeddings**, so it makes no embedding API calls). The full-text
index is skipped because it already persists; the brute-force vector index is
not rebuilt because it is not on the live `ask` path.

Net effect: after a restart or restore, **all search modes work immediately** —
no manual re-ingestion is required. The earlier "indexes are empty until you
re-insert" caveat no longer applies.

What to know operationally:

- **Startup cost is O(stored objects).** The reindex pass reads and deserializes
  the whole object set once at boot, so a large corpus adds to startup time
  (and transiently to memory while batches are loaded). On very large corpora,
  budget for a longer warm-up before the server reports healthy.
- **Embeddings must be present in storage** to rebuild the vector index. Objects
  inserted with auto-embedding have their generated embedding written back and
  persisted, so they reindex correctly. Objects stored without any embedding are
  reindexed into the metadata/time/graph indexes but not the vector index.
- **The full-text index is independent.** If you delete `{index-path}/tantivy/`,
  it is *not* rebuilt by the reindex pass — you must re-ingest to repopulate it
  (see [Upgrades](#upgrades)).

Validate readiness after a restart by checking that `Stats.vector_count` matches
`Stats.total_objects` (modulo objects with no embedding). Schedule restarts of
very large instances during low-traffic windows to absorb the reindex time.

## Upgrades

- **Binary upgrades.** Replace the binary/image and restart. The startup reindex
  pass rebuilds the in-memory indexes from storage automatically (see
  [Index persistence and restarts](#index-persistence-and-restarts)); on large
  corpora budget for the extra startup time.
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

The `Stats` RPC (`StatsResponse`) and the `kowitodb stats` CLI command now report
the **same set of fields** — the earlier discrepancy (gRPC returning only three
fields) is resolved.

| Field | Source | Meaning |
| --- | --- | --- |
| `total_objects` | storage `count()` | Number of stored objects. |
| `vector_count` | HNSW index `len()` | Vectors in the in-memory HNSW index. After a restart this should match `total_objects` (minus any objects with no embedding) once the reindex pass completes — a useful warm-up indicator. |
| `graph_nodes` / `graph_edges` | graph index | Nodes/edges in the relationship graph. |
| `active_agent_sessions` | agent memory | Live conversation sessions (in-memory; reset on restart). |
| `total_cost_usd` | cost tracker | Running estimated USD cost (see below). |
| `cache_entries` / `cache_hit_rate` | plan cache | Plan-cache size and hit rate. |
| `index_size_bytes` | — | Currently always `0`; **not** a real byte size — ignore it. |

For Prometheus-style scraping, run the server with `--metrics-addr` and scrape
`GET /metrics`. That endpoint renders `MetricsCollector`'s
ask/remember/insert/sql/error counts, cumulative and average ask latency, and
uptime in Prometheus text format. `GET /healthz` on the same address returns
`ok`. The gRPC health-checking service (`grpc.health.v1.Health`) and reflection
are always on as well — see [DEPLOYMENT.md → Observability](DEPLOYMENT.md#observability).

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

## Updates and version history

The `Update` RPC (and `KowitoDBEngine::update`) edits an object in place, keyed
by its existing id:

- Only the fields you supply change: `content` (replaces and triggers
  re-embedding), `metadata` (merged — keys overwrite), `keywords` (replaces when
  non-empty), and `importance`. An optional `change_description` is recorded.
- Each update appends an entry to the object's `version_history`, which is
  **persisted to storage** (`version_history_json` in sled and Lance) and so
  accumulates across restarts. The RPC returns the new version count.
- Updating an object whose content changed clears its stored embedding so a fresh
  one is generated on re-index, keeping the vector index accurate. If you switch
  embedding providers/models, re-`update` (or re-insert) affected objects to
  re-embed them.
- Operationally, version history grows unbounded with edit frequency; there is no
  pruning. For heavily-edited objects, account for the storage growth.

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
- **Working set must fit in RAM.** The secondary indexes are in-memory (rebuilt
  from storage on startup, so no re-ingestion is needed), but the full object set
  still has to fit in memory across those indexes. There is no spill-to-disk.
- **Auth and TLS are off by default.** Enable `--api-key` and `--tls-cert`/
  `--tls-key`, and/or secure the endpoint at the network/proxy layer (see
  [DEPLOYMENT.md](DEPLOYMENT.md)). When off, the gRPC endpoint is unauthenticated
  plaintext.
- **Health-check and reflection services are unauthenticated by design.** The
  always-on gRPC health/reflection services and the `--metrics-addr` HTTP
  endpoint do not honor the API key, so probes and tooling work without
  credentials. Keep them off the public internet.
- **Agent memory is in-memory and not persisted.** `RecordTurn`/`GetSession`
  expose conversation sessions over gRPC and they count toward
  `active_agent_sessions`, but all sessions are lost on restart. Do not treat
  them as durable storage.
- **Default embeddings are a deterministic proxy**, not a semantic model. Set
  `KOWITODB_EMBEDDING_PROVIDER` (openai/ollama) before relying on vector
  relevance in production. Token counts are heuristic (~4 chars/token).
- **`Search`/`Ask` results cap at 100.** (`max_context_tokens`, when > 0, is now
  honored as the per-request context budget; 0/unset uses the 4096-token
  default.)
- **Keyword and time predicates still scan storage.** `id` filters take a direct
  key lookup, and `id`/`min_importance` are pushed down (into the sled fast path
  and the Lance native scan); the DataFusion provider pushes `LIMIT` into storage
  only when there is no `WHERE` clause. But keyword and date-range predicates,
  and any unfiltered scan, are still O(n) over the object store — this bounds how
  large a corpus stays responsive.
- **Single-writer data directory.** sled holds an exclusive lock on its
  directory, so only one process (the server, or one embedded CLI command) can
  open a given data directory at a time. For concurrent access, run `serve` and
  use clients.

## Operational runbook

| Situation | Action |
| --- | --- |
| After a restart, `ask` returns weak/empty results | The startup reindex pass should have rebuilt the indexes. Check that `Stats.vector_count` ≈ `Stats.total_objects`; if it is still climbing, the reindex pass is still running — wait for it. If counts never recover, check the logs for reindex errors and that objects actually carry embeddings. |
| Slow startup on a large corpus | Expected — the reindex pass is O(stored objects). Budget for it; schedule restarts off-peak. |
| Plan-cache hit rate near zero during a bulk load | Expected — writes clear the cache. It recovers once writes settle. |
| Memory pressure / OOM risk | The full working set lives in RAM. Shed objects, raise host memory, or move to a larger node; there is no spill-to-disk for the in-memory indexes. |
| Need to expose the port externally | Enable `--api-key` and TLS (`--tls-cert`/`--tls-key`), and put network ACLs in front. |
| Need a real liveness probe | Use the gRPC health service (`grpc_health_probe`), `GET /healthz` on `--metrics-addr`, or a periodic `Stats` RPC. |
| Corrupt/incompatible data dir after upgrade | Restore from backup, or delete `{index-path}/tantivy/` and re-ingest to rebuild the full-text index. |
| Need full SQL (aggregates, ORDER BY) | Use the `Sql` RPC / SDK `sql()` (DataFusion), the engine `sql_select` API, or the `kowitodb sql` CLI (index-routed path). |
