# KowitoDB — AI Knowledge Operating System

**The PostgreSQL of AI.** An open-source database that automatically plans retrieval, manages memory, optimizes context, and executes AI queries — eliminating the need to stitch together vector databases, caches, graph databases, and retrieval pipelines.

```python
# Instead of:
embedding() → vector.search() → sql.query() → graph.search() → rerank() → build_context() → llm()

# Just write:
db.ask("Which enterprise customers renewed after Series A?")
```

## Architecture

```
                    SDK (Python, TS, Go, Rust)
                           │
                      gRPC / Flight
                           │
              ┌────────────────────────┐
              │   AI Query Planner      │
              │  · Intent Detection     │
              │  · Retrieval Optimizer  │
              │  · Context Optimizer    │
              │  · Cost Optimizer       │
              └────────────────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         │        │        │        │         │
      Vector  Keyword   Graph   Memory     SQL
         │        │        │        │         │
         └─────────────────┼─────────────────┘
                           │
              ┌────────────────────────┐
              │  Arrow Execution Engine │
              │     (DataFusion)        │
              └────────────────────────┘
                           │
                    Storage Layer
                 (Object Storage / SSD)
```

## Why KowitoDB?

| Problem | KowitoDB Solution |
|---------|-------------------|
| "I stitch together 5+ systems for RAG" | One database: `ai.ask()` handles everything |
| "Vector search alone gives shallow results" | Multi-index: semantic + keyword + graph + metadata + time |
| "My retrieval is a fixed pipeline" | AI Query Planner chooses the best strategy per query |
| "50k tokens of context is expensive" | Context Optimizer deduplicates, summarizes, compresses |
| "I pay for unused LLM capacity" | Built-in cost optimization and caching |

## Quick Start

### Build from source

```bash
# Prerequisites: Rust 1.75+
git clone https://github.com/kowitodb/kowitodb.git
cd kowitodb

# Build
cargo build --release

# Start the server
cargo run --release -- serve \
  --addr 127.0.0.1:50051 \
  --storage-path ./data/storage \
  --index-path ./data/index
```

### Python SDK

```bash
pip install kowitodb

# Or from source:
cd sdk/python && pip install -e .
```

```python
from kowitodb import KowitoDBClient

db = KowitoDBClient("localhost:50051")

# Store knowledge
db.remember(
    "Acme Corp renewed their enterprise license in March 2024 after Series A funding of $15M.",
    keywords=["acme", "renewal", "series a", "enterprise"],
    metadata={"company": "Acme Corp", "stage": "series_a"},
    importance=0.9,
)

# Ask natural-language questions
response = db.ask("Which enterprise customers renewed after Series A?")

print(f"Intent: {response.detected_intent}")
print(f"Plan:\n{response.plan_explanation}")
for result in response.results:
    print(f"  [{result.relevance_score:.2f}] {result.content}")
```

## How It Works

### 1. Storage Layer
Every document is stored as a **Knowledge Object** containing:
- Raw content
- Embeddings (multiple models)
- Metadata (arbitrary key-value)
- Keywords (for full-text search)
- Relationships (for graph traversal)
- Version history
- Importance score

### 2. Index Layer
Five indexes maintained simultaneously:
- **Vector** (cosine similarity, upgrade path to HNSW)
- **Full-text** (Tantivy — BM25, tokenization, inverted index)
- **Metadata** (exact and substring matching)
- **Time** (range queries via BTreeMap)
- **Graph** (named relationships, upgrade path to graph DB)

### 3. AI Query Planner (The Moat)

```text
"What companies raised funding after OpenAI?"
          │
    Intent Detection
   (Factoid, Comparison, Temporal, Entity, Code, Summary, General)
          │
     Rule Engine
   "Contains date? → Time Index"
   "Contains compare? → Dual retrieval"
   "Company names? → Entity graph"
   "Source code? → Code embedding"
          │
   Execution Plan
   [VectorSearch] → [KeywordSearch] → [GraphTraverse] → [Merge] → [Rerank] → [BuildContext]
```

## Project Structure

```
kowitodb/
├── kowitodb-core/       # Knowledge Object types, errors
├── kowitodb-storage/    # Sled-backed persistent storage
├── kowitodb-index/      # Multi-index layer (vector, fulltext, metadata, time)
├── kowitodb-planner/    # AI Query Planner (intent detection, rule engine, plans)
├── kowitodb-server/     # gRPC server (tonic)
├── kowitodb/            # CLI binary
├── proto/               # Protobuf definitions
├── sdk/python/          # Python client SDK
└── examples/            # Demo scripts
```

## Phase Roadmap

| Phase | Deliverable | Status |
|-------|-------------|--------|
| 1 | Storage + Vector Search MVP | ✅ Implemented |
| 2 | Multi-index Retrieval | ✅ Implemented |
| 3 | Query Planner (rule-based) | ✅ Implemented |
| 4 | Retrieval Optimizer (learned) | Planned |
| 5 | Context Optimizer | Planned |
| 6 | Agent Memory | Planned |
| 7 | Distributed Cluster | Planned |
| 8 | AI Runtime | Planned |

## Tech Stack

- **Language**: Rust
- **Query Engine**: Apache DataFusion (upgrade path)
- **Memory Format**: Apache Arrow
- **Full-text**: Tantivy (Lucene-compatible)
- **Storage**: Sled (embedded DB, upgrade path to Lance)
- **RPC**: gRPC + Protocol Buffers
- **Async Runtime**: Tokio
- **Serialization**: Serde + JSON

## License

MIT
