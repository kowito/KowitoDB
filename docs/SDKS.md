# KowitoDB SDKs

Client libraries for the KowitoDB gRPC service in Python, TypeScript, and Go.
All three target the same `KowitoDB` service defined in
[`proto/kowitodb.proto`](../proto/kowitodb.proto) and expose the same core
surface.

The default server address is `localhost:50051`. All clients use **insecure
(plaintext)** connections by default — the server does not terminate TLS or
authenticate (see [DEPLOYMENT.md](DEPLOYMENT.md#security-posture)).

## Contents

- [Capability matrix](#capability-matrix)
- [Python](#python)
- [TypeScript](#typescript)
- [Go](#go)
- [Regenerating gRPC stubs](#regenerating-grpc-stubs)
- [A note on `sql()`](#a-note-on-sql)

## Capability matrix

| Method | Python | TypeScript | Go | Underlying RPC |
| --- | :---: | :---: | :---: | --- |
| `remember(content, …)` | yes | yes | yes | `Remember` |
| `ask(question, maxResults?)` | yes | yes | yes | `Ask` |
| `forget(id)` / `Delete` | yes (`forget`) | yes (`forget`) | yes (`Delete`) | `Delete` |
| `insert(content, …)` | yes | yes | yes | `Insert` |
| `get(id)` | yes | yes | yes | `Get` |
| `search(query, topK?)` | yes | yes | yes | `Search` |
| `stats()` | yes | yes | yes | `Stats` |
| `sql(query)` | yes | yes | — | `Search` (see [note](#a-note-on-sql)) |

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

The connection is lazy; calling a method auto-connects. All message types
(`AskResponse`, `AskResult`, `SearchResult`, `KnowledgeObject`, `StatsResponse`,
…) are exported for type annotations.

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
interceptors. `Remember`/`Insert` accept functional options: `WithKeywords`,
`WithMetadata`, `WithImportance`, and `WithRelationships` (`Insert` only).

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

## A note on `sql()`

The Python and TypeScript clients expose a `sql(query)` helper, but it routes
the query string through the **`Search` RPC** — i.e. the `ask` retrieval
pipeline — **not** the SQL engine. There is no dedicated SQL RPC in the proto.

Full SQL (projection, `WHERE`, `ORDER BY`, `GROUP BY`, aggregates, `LIMIT`) is
available only:

- through the Rust engine API `KowitoDBEngine::sql_select` (DataFusion path), or
- via the `kowitodb sql` CLI command (the lighter index-routed parser path).

See [ARCHITECTURE.md → The DataFusion SQL path](ARCHITECTURE.md#the-datafusion-sql-path)
for details.
