# @kowitodb/sdk

TypeScript gRPC client for **KowitoDB**, the AI Knowledge Operating System.

It mirrors the [Python SDK](../python) one-to-one: the same `KowitoDBClient`
class with `remember`, `ask`, `forget`, `sql`, `insert`, `get`, `search`, and
`stats` methods. All methods are async and return Promises.

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

  // Natural-language question with automatic retrieval
  const res = await db.ask("Which companies raised funding?");
  for (const r of res.results) {
    console.log(`[${r.relevance_score.toFixed(2)}] ${r.content}`);
  }

  // SQL-style query (routed through search)
  const rows = await db.sql(
    "SELECT * FROM knowledge WHERE metadata.company = 'OpenAI'",
  );
  console.log(rows);

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
| `ask(question, maxResults?)` | Natural-language query with auto retrieval. | `Promise<AskResponse>` |
| `forget(id)` | Delete a knowledge object. | `Promise<boolean>` (existed) |
| `sql(query)` | SQL-style query (routed through search). | `Promise<AskResult[]>` |
| `insert(content, { keywords?, metadata?, relationships?, importance? })` | Explicit insert. | `Promise<string>` (id) |
| `get(id)` | Fetch a knowledge object. | `Promise<KnowledgeObject \| null>` |
| `search(query, topK?)` | Direct search (no AI planner). | `Promise<SearchResult[]>` |
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
