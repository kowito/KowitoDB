# KowitoDB SDKs

Client libraries for the KowitoDB gRPC service in Python, TypeScript, and Go.
All three target the same `KowitoDB` service defined in
[`proto/kowitodb.proto`](../proto/kowitodb.proto) and expose the same core
surface.

The default server address is `localhost:50051`. All clients default to
**insecure (plaintext)** connections. The server's auth and TLS are off by
default; if you enable them (`--api-key`, `--tls-cert`/`--tls-key` — see
[DEPLOYMENT.md](DEPLOYMENT.md#security-posture)), pass the matching credentials
through the client's transport options (e.g. a Bearer / `x-api-key` metadata
header and TLS channel credentials).

## Contents

- [Capability matrix](#capability-matrix)
- [Python](#python)
- [TypeScript](#typescript)
- [Go](#go)
- [Regenerating gRPC stubs](#regenerating-grpc-stubs)
- [The `sql()` helper](#the-sql-helper)

## Capability matrix

| Method | Python | TypeScript | Go | Underlying RPC |
| --- | :---: | :---: | :---: | --- |
| `remember(content, …)` | yes | yes | yes | `Remember` |
| `ask(question, maxResults?)` | yes | yes | yes | `Ask` |
| `forget(id)` / `Delete` | yes (`forget`) | yes (`forget`) | yes (`Delete`) | `Delete` |
| `insert(content, …)` | yes | yes | yes | `Insert` |
| `get(id)` | yes | yes | yes | `Get` |
| `update(id, …)` | yes | yes | yes | `Update` |
| `search(query, topK?)` | yes | yes | yes | `Search` |
| `sql(query)` | yes | yes | yes | `Sql` (DataFusion; returns rows) |
| `record_turn` / `recordTurn` / `RecordTurn` | yes | yes | yes | `RecordTurn` |
| `get_session` / `getSession` / `GetSession` | yes | yes | yes | `GetSession` |
| `stats()` | yes | yes | yes | `Stats` |

`sql(query)` calls the dedicated `Sql` RPC and returns rows as a list of
column-name → value maps (Python/TS: `list`/`Array` of dict/`Record`; Go:
`[]map[string]string`). `stats()` returns the full field set: `total_objects`,
`vector_count`, `index_size_bytes`, `graph_nodes`, `graph_edges`,
`active_agent_sessions`, `total_cost_usd`, `cache_entries`, `cache_hit_rate`.

Connection lifecycle: every client connects lazily on first call. Python and
TypeScript expose explicit `connect()`/`close()`; Go opens on `NewClient` and
closes on `Close()`.

## Python

Package: `kowitodb` (`sdk/python`), transport `grpcio`.

### Install

```bash
# From the repository:
cd sdk/python
pip install -e ".[grpc]"
```

The `grpc` extra pulls in `grpcio` and `grpcio-tools`. Requires Python ≥ 3.10.
Generated stubs (`kowitodb_pb2.py`, `kowitodb_pb2_grpc.py`) are checked into the
package.

### Remember, then ask

```python
from kowitodb import KowitoDBClient

db = KowitoDBClient("localhost:50051")

# Store knowledge (ai.remember()).
obj_id = db.remember(
    "Acme Corp renewed their enterprise license in March 2024 after Series A funding of $15M.",
    keywords=["acme", "renewal", "series a", "enterprise"],
    metadata={"company": "Acme Corp", "stage": "series_a"},
    importance=0.9,
)
print("stored:", obj_id)

# Ask a natural-language question (ai.ask()).
resp = db.ask("Which enterprise customers renewed after Series A?", max_results=5)
print("intent:", resp.detected_intent)
print(resp.plan_explanation)
for r in resp.results:
    print(f"[{r.relevance_score:.2f}] ({r.retrieval_source}) {r.content}")

db.close()
```

`KowitoDBClient` is also a context manager (`with KowitoDBClient(...) as db:`).

### Update, SQL, and agent memory

```python
# Edit in place (only the fields you pass change); records version history.
res = db.update(obj_id, importance=0.95, change_description="bumped priority")
print("updated:", res.updated, "version:", res.version)

# Full SQL via the DataFusion Sql RPC — rows come back as dicts.
rows = db.sql("SELECT content, importance FROM knowledge "
              "WHERE importance >= 0.5 ORDER BY importance DESC")
for row in rows:
    print(row["importance"], row["content"])

# Agent conversation memory.
db.record_turn("session-1", "user", "Who renewed after Series A?")
db.record_turn("session-1", "assistant", "Acme Corp did.")
turns = db.get_session("session-1")  # None if the session does not exist
for t in turns or []:
    print(t.role, t.content)
```

## TypeScript

Package: `@kowitodb/sdk` (`sdk/typescript`), transport `@grpc/grpc-js` +
`@grpc/proto-loader`. The proto is loaded dynamically at runtime; there is no
generated `*_pb` code. Requires Node ≥ 18.

### Install

```bash
npm install @kowitodb/sdk
# or, from the repo:
cd sdk/typescript && npm install && npm run build
```

### Remember, then ask

```ts
import { KowitoDBClient } from "@kowitodb/sdk";

async function main() {
  const db = new KowitoDBClient("localhost:50051");

  // Store knowledge (ai.remember()).
  const id = await db.remember(
    "Acme Corp renewed their enterprise license in March 2024 after Series A funding of $15M.",
    {
      keywords: ["acme", "renewal", "series a", "enterprise"],
      metadata: { company: "Acme Corp", stage: "series_a" },
      importance: 0.9,
    },
  );
  console.log("stored:", id);

  // Ask a natural-language question (ai.ask()).
  const res = await db.ask("Which enterprise customers renewed after Series A?", 5);
  console.log("intent:", res.detected_intent);
  console.log(res.plan_explanation);
  for (const r of res.results) {
    console.log(`[${r.relevance_score.toFixed(2)}] (${r.retrieval_source}) ${r.content}`);
  }

  db.close();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
```

### Update, SQL, and agent memory

```ts
// Edit in place; only the provided options change. Records version history.
const upd = await db.update(id, { importance: 0.95, changeDescription: "bump" });
console.log("updated:", upd.updated, "version:", upd.version);

// Full SQL via the Sql RPC — rows are Record<string, string>.
const rows = await db.sql(
  "SELECT content, importance FROM knowledge WHERE importance >= 0.5 ORDER BY importance DESC",
);
for (const row of rows) console.log(row.importance, row.content);

// Agent conversation memory.
await db.recordTurn("session-1", "user", "Who renewed after Series A?");
await db.recordTurn("session-1", "assistant", "Acme Corp did.");
const turns = await db.getSession("session-1"); // null if not found
for (const t of turns ?? []) console.log(t.role, t.content);
```

The connection is lazy; calling a method auto-connects. All message types
(`AskResponse`, `AskResult`, `SearchResult`, `KnowledgeObject`, `StatsResponse`,
`UpdateResponse`, `ConversationTurnProto`, …) are exported for type annotations.

## Go

Module: `github.com/kowito/kowitodb/sdk/go`, transport
`google.golang.org/grpc`. Generated protobuf code lives in `kowitodbpb/`.

### Install

```sh
go get github.com/kowito/kowitodb/sdk/go@latest
```

### Remember, then ask

```go
package main

import (
	"context"
	"fmt"
	"log"

	kowitodb "github.com/kowito/kowitodb/sdk/go"
)

func main() {
	db, err := kowitodb.NewClient("localhost:50051")
	if err != nil {
		log.Fatal(err)
	}
	defer db.Close()

	ctx := context.Background()

	// Store knowledge (ai.remember()).
	id, err := db.Remember(ctx,
		"Acme Corp renewed their enterprise license in March 2024 after Series A funding of $15M.",
		kowitodb.WithKeywords("acme", "renewal", "series a", "enterprise"),
		kowitodb.WithMetadata(map[string]string{"company": "Acme Corp", "stage": "series_a"}),
		kowitodb.WithImportance(0.9),
	)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("stored:", id)

	// Ask a natural-language question (ai.ask()).
	resp, err := db.Ask(ctx, "Which enterprise customers renewed after Series A?", 5)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("intent:", resp.DetectedIntent)
	for _, r := range resp.Results {
		fmt.Printf("[%.2f] (%s) %s\n", r.RelevanceScore, r.RetrievalSource, r.Content)
	}
}
```

`NewClient("")` uses the default address `localhost:50051`. Pass
`grpc.DialOption`s to `NewClient` to customize transport credentials or
interceptors (e.g. TLS credentials, or a per-call API-key metadata header when
the server is run with `--api-key`). `Remember`/`Insert` accept functional
options: `WithKeywords`, `WithMetadata`, `WithImportance`, and
`WithRelationships` (`Insert` only).

### Update, SQL, and agent memory

```go
// Edit in place; UpdateOptions decide which fields change. Records version history.
updated, version, err := db.Update(ctx, id,
    kowitodb.WithUpdatedImportance(0.95),
    kowitodb.WithChangeDescription("bump"),
)
// ... handle err
fmt.Println("updated:", updated, "version:", version)

// Full SQL via the Sql RPC — rows are []map[string]string.
rows, err := db.Sql(ctx,
    "SELECT content, importance FROM knowledge WHERE importance >= 0.5 ORDER BY importance DESC")
// ... handle err
for _, row := range rows {
    fmt.Println(row["importance"], row["content"])
}

// Agent conversation memory.
_, _ = db.RecordTurn(ctx, "session-1", "user", "Who renewed after Series A?")
_, _ = db.RecordTurn(ctx, "session-1", "assistant", "Acme Corp did.")
turns, err := db.GetSession(ctx, "session-1") // empty slice if not found
// ... handle err
for _, t := range turns {
    fmt.Println(t.Role, t.Content)
}
```

Update options: `WithUpdatedContent`, `WithUpdatedMetadata`,
`WithUpdatedKeywords`, `WithUpdatedImportance`, `WithChangeDescription`.

## Regenerating gRPC stubs

All SDKs derive from [`proto/kowitodb.proto`](../proto/kowitodb.proto). If you
change the proto, regenerate the stubs and roll clients with the server (there
is no negotiated API versioning).

### Python

The `grpc` extra installs `grpcio-tools`. From `sdk/python`:

```bash
python -m grpc_tools.protoc \
  -I ../../proto \
  --python_out=kowitodb \
  --grpc_python_out=kowitodb \
  ../../proto/kowitodb.proto
```

This regenerates `kowitodb/kowitodb_pb2.py` and `kowitodb/kowitodb_pb2_grpc.py`.

### TypeScript

The TypeScript SDK loads the proto dynamically, so there is no `*_pb` code to
regenerate — only the bundled copy of the proto to refresh:

```bash
cd sdk/typescript
npm run proto:gen   # copies proto/kowitodb.proto into the package's proto/ dir
```

If you add or change message fields, also update the hand-written interfaces in
`src/types.ts` to match, then `npm run build`.

### Go

From `sdk/go` (requires `protoc`, `protoc-gen-go`, `protoc-gen-go-grpc`):

```sh
make tools      # install the protoc Go plugins (one-time)
make generate   # regenerate kowitodbpb/*.pb.go from ../../proto/kowitodb.proto
```

## The `sql()` helper

All three clients (`sql` in Python/TypeScript, `Sql` in Go) call the dedicated
**`Sql` RPC**, which runs the query through the **DataFusion engine**
(`KowitoDBEngine::sql_select`). This supports full SQL — projection, `WHERE`,
`ORDER BY`, `GROUP BY`, aggregates, and `LIMIT` — over the `knowledge` table
(columns `id`, `content`, `importance`, `created_at`, `updated_at`, `keywords`,
`metadata`). Rows are returned as column-name → value maps (every value
stringified):

```sql
SELECT COUNT(*) AS n FROM knowledge;
SELECT content, importance FROM knowledge WHERE importance >= 0.5 ORDER BY importance DESC;
```

The `kowitodb sql` CLI command uses a separate, lighter index-routed parser
path (`sql_query`); the SDK `sql()` helpers do **not** use it. See
[ARCHITECTURE.md → The DataFusion SQL path](ARCHITECTURE.md#the-datafusion-sql-path)
for details.

## Framework integrations (Python)

The Python SDK ships LangChain and LlamaIndex adapters so KowitoDB can be used
as a retriever / vector store directly inside those frameworks. Install the
extra for the framework you use:

```bash
pip install "kowitodb[langchain]"
pip install "kowitodb[llamaindex]"
```

**LangChain** — `kowitodb.integrations.langchain`:

```python
from kowitodb import KowitoDBClient
from kowitodb.integrations.langchain import KowitoDBRetriever, KowitoDBVectorStore

client = KowitoDBClient("localhost:50051")

retriever = KowitoDBRetriever(client=client, max_results=5)   # uses ai.ask() by default
docs = retriever.invoke("which customers renewed after Series A?")

store = KowitoDBVectorStore(client)                            # embedding happens server-side
store.add_texts(["Acme renewed.", "Globex churned."],
                metadatas=[{"company": "Acme"}, {"company": "Globex"}])
hits = store.similarity_search("renewals", k=3)
```

**LlamaIndex** — `kowitodb.integrations.llamaindex`:

```python
from kowitodb import KowitoDBClient
from kowitodb.integrations.llamaindex import KowitoDBRetriever

retriever = KowitoDBRetriever(KowitoDBClient("localhost:50051"), top_k=5)
nodes = retriever.retrieve("which customers renewed after Series A?")
```

Both retrievers default to the `ai.ask()` pipeline (intent detection,
multi-index retrieval, graph traversal, rerank); pass `use_ask=False` for the
raw `search()` path, and `metadata_filter=` to constrain by metadata.
