# @kowitodb/sdk

TypeScript gRPC client for **KowitoDB**, the AI Knowledge Operating System.

It mirrors the [Python SDK](../python) one-to-one: the same `KowitoDBClient`
class with `remember`, `ask`, `forget`, `sql`, `insert`, `batchInsert`, `get`,
`list`, `update`, `search`, `recordTurn`, `getSession`, and `stats` methods. All
methods are async and return Promises.

## Install

```bash
npm install @kowitodb/sdk
```

## Usage

```ts
import { KowitoDBClient } from "@kowitodb/sdk";

async function main() {
  const db = new KowitoDBClient("localhost:50051");
  db.connect();

  // Store knowledge
  await db.remember("OpenAI raised $6.6B in 2024", {
    keywords: ["openai", "funding"],
    metadata: { company: "OpenAI" },
  });

  // Insert many objects in one call
  const ids = await db.batchInsert([
    { content: "Anthropic released Claude", metadata: { company: "Anthropic" } },
    { content: "Mistral shipped a new model", metadata: { company: "Mistral" } },
  ]);
  console.log(ids); // string[]

  // Natural-language question with automatic retrieval
  const res = await db.ask("Which companies raised funding?");
  for (const r of res.results) {
    console.log(`[${r.relevance_score.toFixed(2)}] ${r.content}`);
  }

  // Ask / search constrained to objects matching exact metadata
  const filtered = await db.ask("What happened?", 10, {
    metadataFilter: { company: "OpenAI" },
  });
  console.log(filtered.results.length);
  const hits = await db.search("funding", 20, {
    metadataFilter: { company: "OpenAI" },
  });
  console.log(hits.length);

  // Paginate through stored objects
  const page = await db.list(0, 50);
  console.log(`${page.objects.length} of ${page.total}`);

  // SQL query (DataFusion engine) — returns an array of column->value maps
  const rows = await db.sql(
    "SELECT id, content FROM knowledge WHERE metadata.company = 'OpenAI'",
  );
  for (const row of rows) {
    console.log(row.id, row.content);
  }

  // Update an existing object (partial; only provided fields change)
  await db.update("obj-123", {
    metadata: { reviewed: "true" },
    changeDescription: "marked as reviewed",
  });

  // Agent conversation memory
  await db.recordTurn("session-1", "user", "What did OpenAI raise?");
  const turns = await db.getSession("session-1");
  console.log(turns); // ConversationTurnProto[] | null

  db.close();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
```

### Connection lifecycle

The connection is established lazily, so you can skip `connect()` and just call
a method — it will connect on first use. Call `close()` when you are done to
release the channel.

```ts
const db = new KowitoDBClient(); // defaults to "localhost:50051"
const stats = await db.stats();  // auto-connects
db.close();
```

## API

| Method | Description | Returns |
| --- | --- | --- |
| `connect()` | Open the gRPC channel (idempotent). | `void` |
| `close()` | Close the gRPC channel. | `void` |
| `remember(content, { keywords?, metadata?, importance? })` | Store knowledge. | `Promise<string>` (id) |
| `ask(question, maxResults?, { maxResults?, metadataFilter? })` | Natural-language query with auto retrieval. | `Promise<AskResponse>` |
| `forget(id)` | Delete a knowledge object. | `Promise<boolean>` (existed) |
| `sql(query)` | SQL query over the DataFusion engine. | `Promise<Array<Record<string, string>>>` (rows) |
| `insert(content, { keywords?, metadata?, relationships?, importance? })` | Explicit insert. | `Promise<string>` (id) |
| `batchInsert(items)` | Insert many objects; each item is `{ content, ...insertOptions }`. | `Promise<string[]>` (ids) |
| `get(id)` | Fetch a knowledge object. | `Promise<KnowledgeObject \| null>` |
| `list(offset?, limit?)` | Paginate stored objects (`limit` 0 = server default). | `Promise<{ objects: KnowledgeObject[]; total: number }>` |
| `update(id, { content?, metadata?, keywords?, importance?, changeDescription? })` | Partial update of an object. | `Promise<UpdateResponse>` |
| `search(query, topK?, { topK?, metadataFilter? })` | Direct search (no AI planner). | `Promise<SearchResult[]>` |
| `recordTurn(sessionId, role, content)` | Append a turn to an agent session. | `Promise<number>` (turn count) |
| `getSession(sessionId)` | Fetch a session's turns. | `Promise<ConversationTurnProto[] \| null>` |
| `stats()` | Database statistics. | `Promise<StatsResponse>` |

All message types (`AskResponse`, `AskResult`, `SearchResult`,
`KnowledgeObject`, `StatsResponse`, …) are exported for use in type
annotations.

## Codegen approach

This SDK loads `proto/kowitodb.proto` **dynamically** at runtime using
[`@grpc/proto-loader`](https://www.npmjs.com/package/@grpc/proto-loader) +
[`@grpc/grpc-js`](https://www.npmjs.com/package/@grpc/grpc-js). There is no
generated `*_pb` code to maintain. Type safety is provided by hand-written
TypeScript interfaces in [`src/types.ts`](./src/types.ts) that mirror the proto
messages, layered over the dynamically-loaded service in
[`src/service.ts`](./src/service.ts).

Why dynamic loading: the proto is small and stable, and this avoids requiring a
`protoc`/`ts-proto` toolchain in the build. The bundled proto is kept in sync
with the repository's canonical proto via `npm run proto:gen`.

```bash
npm run proto:gen   # copy proto/kowitodb.proto from the repo root into the package
```

If you change the proto, re-run `proto:gen` and update `src/types.ts` to match.

## Build

```bash
npm install
npm run build       # emit dist/ (JS + d.ts)
npm run typecheck   # type-check without emitting
```
