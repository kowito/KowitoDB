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
}
```

## API

`NewClient(addr string, opts ...grpc.DialOption) (*Client, error)` opens a
connection. Pass `""` to use the default address `localhost:50051`. By default
an insecure (plaintext) connection is used; supply `grpc.DialOption`s to
customise transport credentials, interceptors, etc. Call `Close()` when done.

All RPC methods take a `context.Context` as the first argument and return a
typed response plus an `error`.

| Method | Description |
| --- | --- |
| `Remember(ctx, content, ...WriteOption) (string, error)` | Store knowledge (high-level `ai.remember()`); returns the object ID. |
| `Ask(ctx, question, maxResults) (*AskResponse, error)` | Natural-language query with automatic retrieval (`ai.ask()`). `maxResults <= 0` defaults to 10. |
| `Insert(ctx, content, ...WriteOption) (string, error)` | Explicitly insert a knowledge object; returns the ID. |
| `Get(ctx, id) (*KnowledgeObject, error)` | Fetch an object by ID; returns `(nil, nil)` if not found. |
| `Update(ctx, id, ...UpdateOption) (updated bool, version uint32, err error)` | Update an object in place; returns whether it changed and the new version-history length. |
| `Delete(ctx, id) (bool, error)` | Delete by ID; returns whether it existed. |
| `Search(ctx, query, topK) ([]SearchResult, error)` | Direct search, bypassing the AI planner. `topK <= 0` defaults to 20. |
| `Sql(ctx, query) ([]map[string]string, error)` | Run a SQL query against the DataFusion engine; each row maps column name to value. |
| `RecordTurn(ctx, sessionID, role, content) (uint32, error)` | Append a turn to an agent session; returns the new turn count. |
| `GetSession(ctx, sessionID) ([]ConversationTurn, error)` | Fetch a session's turns; returns `(nil, nil)` if not found. |
| `Stats(ctx) (*Stats, error)` | Database statistics (objects, vectors, graph, agent sessions, cost, cache). |

### Write options

`Remember` and `Insert` accept functional options:

- `WithKeywords(keywords ...string)`
- `WithMetadata(map[string]string)`
- `WithImportance(float32)` — default `0.5`
- `WithRelationships(...Relationship)` — `Insert` only

### Update options

`Update` accepts functional options; only the fields you set are changed:

- `WithUpdatedContent(string)` — replaces content (re-embeds)
- `WithUpdatedMetadata(map[string]string)` — merged into existing metadata
- `WithUpdatedKeywords(...string)` — replaces keywords
- `WithUpdatedImportance(float32)`
- `WithChangeDescription(string)` — recorded in version history

## Regenerating protobuf code

The generated code lives in [`kowitodbpb/`](./kowitodbpb). Regenerate it from
[`../../proto/kowitodb.proto`](../../proto/kowitodb.proto) with:

```sh
make generate
```

This requires `protoc`, `protoc-gen-go`, and `protoc-gen-go-grpc`. Install the
Go plugins with:

```sh
make tools
```
