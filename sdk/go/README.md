# KowitoDB Go SDK

An idiomatic Go gRPC client for [KowitoDB](../../), mirroring the
[Python SDK](../python).

## Install

```sh
go get github.com/kowito/kowitodb/sdk/go@latest
```

```go
import kowitodb "github.com/kowito/kowitodb/sdk/go"
```

## Usage

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

	// Store knowledge
	id, err := db.Remember(ctx, "OpenAI raised $6.6B in 2024",
		kowitodb.WithKeywords("openai", "funding"),
		kowitodb.WithMetadata(map[string]string{"company": "OpenAI"}),
	)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("stored:", id)

	// Ask a natural-language question
	resp, err := db.Ask(ctx, "Which companies raised funding?", 10)
	if err != nil {
		log.Fatal(err)
	}
	for _, r := range resp.Results {
		fmt.Printf("[%.2f] %s\n", r.RelevanceScore, r.Content)
	}

	// Insert several objects at once
	ids, err := db.BatchInsert(ctx, []kowitodb.InsertItem{
		{Content: "Anthropic raised $4B in 2024", Metadata: map[string]string{"company": "Anthropic"}},
		{Content: "Mistral raised €600M in 2024", Keywords: []string{"mistral", "funding"}},
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("stored:", ids)

	// Page through stored objects
	objects, total, err := db.List(ctx, 0, 50)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("showing %d of %d objects\n", len(objects), total)

	// Ask / Search filtered by metadata
	resp, err = db.Ask(ctx, "funding rounds", 10,
		kowitodb.WithMetadataFilter(map[string]string{"company": "OpenAI"}))
	if err != nil {
		log.Fatal(err)
	}
}
```

## API

`NewClient(addr string, opts ...grpc.DialOption) (*Client, error)` creates a
client (the connection is made lazily, on the first RPC). Pass `""` to use the
default address `localhost:50051`. The default transport is plaintext
(insecure); any `grpc.DialOption`s you pass are applied after it, so adding
interceptors keeps the default, and
`grpc.WithTransportCredentials(credentials.NewTLS(...))` switches to TLS. Call
`Close()` when done.

Helpers:

- `WithAPIKey(key string) grpc.DialOption` — sends `authorization: Bearer <key>`
  on every RPC (matches the server's `--api-key` / `KOWITODB_API_KEY`).
- `WithDefaultTimeout(d time.Duration) grpc.DialOption` — applies a deadline to
  every RPC whose context has none (a caller's context deadline wins).

```go
db, err := kowitodb.NewClient("db.example.com:50051",
	kowitodb.WithAPIKey(os.Getenv("KOWITODB_API_KEY")),
	kowitodb.WithDefaultTimeout(30*time.Second),
	// grpc.WithTransportCredentials(credentials.NewTLS(&tls.Config{})), // TLS
)
```

All RPC methods take a `context.Context` as the first argument and return a
typed response plus an `error`.

| Method | Description |
| --- | --- |
| `Remember(ctx, content, ...WriteOption) (string, error)` | Store knowledge (high-level `ai.remember()`); returns the object ID. |
| `Ask(ctx, question, maxResults, ...QueryOption) (*AskResponse, error)` | Natural-language query with automatic retrieval (`ai.ask()`). `maxResults <= 0` defaults to 10. Accepts `WithMetadataFilter`. |
| `Insert(ctx, content, ...WriteOption) (string, error)` | Explicitly insert a knowledge object; returns the ID. |
| `BatchInsert(ctx, []InsertItem) ([]string, error)` | Insert multiple objects in one request; returns their IDs in order. |
| `Get(ctx, id) (*KnowledgeObject, error)` | Fetch an object by ID; returns `(nil, nil)` if not found. |
| `List(ctx, offset, limit) ([]KnowledgeObject, uint64, error)` | Page through stored objects; returns the page plus the total object count. `limit == 0` uses the server default. |
| `Update(ctx, id, ...UpdateOption) (updated bool, version uint32, err error)` | Update an object in place; returns whether it changed and the new version-history length. |
| `Delete(ctx, id) (bool, error)` | Delete by ID; returns whether it existed. |
| `Search(ctx, query, topK, ...QueryOption) ([]SearchResult, error)` | Direct search, bypassing the AI planner. `topK <= 0` defaults to 20. Accepts `WithMetadataFilter`. |
| `Sql(ctx, query) ([]map[string]string, error)` | Run a SQL query against the DataFusion engine; each row maps column name to value. |
| `RecordTurn(ctx, sessionID, role, content) (uint32, error)` | Append a turn to an agent session; returns the new turn count. |
| `GetSession(ctx, sessionID) ([]ConversationTurn, error)` | Fetch a session's turns; returns `(nil, nil)` if not found. |
| `Stats(ctx) (*Stats, error)` | Database statistics (objects, vectors, graph, agent sessions, cost, cache). |

### Write options

`Remember` and `Insert` accept functional options:

- `WithID(id string)` — assign the object id (a UUID); otherwise server-generated
- `WithKeywords(keywords ...string)`
- `WithMetadata(map[string]string)`
- `WithImportance(float32)` — default `0.5`
- `WithRelationships(...Relationship)` — `Insert` only

`BatchInsert` takes a slice of `InsertItem` instead of options:

```go
type InsertItem struct {
	ID            string         // optional UUID; empty = server-generated
	Content       string
	Keywords      []string
	Metadata      map[string]string
	Importance    float32        // defaults to 0.5 when zero
	Relationships []Relationship
}
```

### Query options

`Ask` and `Search` accept functional options:

- `WithMetadataFilter(map[string]string)` — restrict results to objects whose
  metadata matches every given key/value pair (exact match, ANDed). An empty or
  nil map means no filtering.

### Update options

`Update` accepts functional options; only the fields you set are changed:

- `WithUpdatedContent(string)` — replaces content (re-embeds)
- `WithUpdatedMetadata(map[string]string)` — merged into existing metadata
- `WithUpdatedKeywords(...string)` — replaces keywords
- `WithUpdatedImportance(float32)`
- `WithChangeDescription(string)` — recorded in version history

## Regenerating protobuf code

The generated code lives in [`kowitodbpb/`](./kowitodbpb). Regenerate it from
[`../../kowitodb-server/proto/kowitodb.proto`](../../kowitodb-server/proto/kowitodb.proto) with:

```sh
make generate
```

This requires `protoc`, `protoc-gen-go`, and `protoc-gen-go-grpc`. Install the
Go plugins (pinned to the versions in the generated file headers) with:

```sh
make tools
```

No system `protoc`? The one bundled with `grpcio-tools` works too:
`make generate PROTOC="python -m grpc_tools.protoc"`.
