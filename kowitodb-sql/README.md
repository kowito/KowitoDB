# kowitodb-sql

The SQL bridge for **[KowitoDB](https://github.com/kowito/KowitoDB)** — the
database built for AI agents. Run real `SELECT`s over the same knowledge objects
that `ai.ask()` searches.

## What's inside

- **`SqlStatement` / `SelectColumn` / `WhereClause`** — a parsed query model that
  routes `WHERE` predicates to the matching index (metadata, keywords, time
  ranges) instead of a full scan.
- **`SqlQueryResult`** — typed rows back out.
- **`SqlError`** — parse / execution errors.
- **DataFusion integration** — the upgrade path to full SQL (joins, aggregates,
  expressions) over the columnar store.

```sql
SELECT content FROM knowledge WHERE importance >= 0.8 AND keyword = 'acme'
```

```bash
kowitodb sql "SELECT content FROM knowledge WHERE importance >= 0.8"
```

## Where it fits

```
kowitodb-core (types) + kowitodb-index (retrieval)
  └─ kowitodb-sql  ← you are here (SQL surface)
       └─ kowitodb-server (engine + gRPC)  →  kowitodb (CLI)
```

For the full feature tour, quickstart, and SDKs, see the
**[project README](https://github.com/kowito/KowitoDB#readme)**.

## License

[MIT](https://github.com/kowito/KowitoDB/blob/main/LICENSE)
