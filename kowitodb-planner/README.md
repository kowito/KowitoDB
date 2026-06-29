# kowitodb-planner

The AI query planner behind **[KowitoDB](https://github.com/kowito/KowitoDB)** —
the database built for AI agents. This is what turns a natural-language
`ai.ask()` into a multi-index retrieval plan and a token-budgeted answer context.

## The pipeline

1. **`IntentAnalyzer`** → `DetectedIntent` — classify the query (lookup, semantic,
   graph, temporal, …) to decide which indexes to hit.
2. **`QueryPlanner`** → `ExecutionPlan` of `PlanStep`s — pick and order the
   retrieval steps across vector / full-text / graph / metadata / time.
3. **`Reranker`** → `RankedResult` — fuse and re-score candidates from every index.
4. **`ContextOptimizer`** → `AssembledContext` — pack the top results into a
   token-budgeted context window of `ContextChunk`s.

Supporting pieces: **`RuleEngine`** (declarative routing rules), **`QueryCache`**
(`CacheStats`) for repeated queries, and a **`CostModel` / `CostTracker`** for
plan cost accounting.

## Where it fits

```
kowitodb-index (retrieval)
  └─ kowitodb-planner  ← you are here (intent → plan → rerank → context)
       └─ kowitodb-server (engine + gRPC)  →  kowitodb (CLI)
```

For the full feature tour, quickstart, and SDKs, see the
**[project README](https://github.com/kowito/KowitoDB#readme)**.

## License

[MIT](https://github.com/kowito/KowitoDB/blob/main/LICENSE)
